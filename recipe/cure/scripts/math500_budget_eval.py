"""MATH500 budget comparison: training-start checkpoint vs CURE global_step_560.

Three strategies share one pool of generations per problem:

* parallel: 8 independent answers. Budgets use the first 1/2/4/8.
* iterative: trajectory 0 runs the native critique -> hint-guided regen loop,
  at most 7 rounds. Smaller budgets read a prefix of that loop.
* multi: trajectories 0-3 each run at most one critique -> regen round.
  Trajectory 0's first round is the same generation as iterative round 0.

Stop rule matches validation, not the training GT filter: regenerate unless the
critique explicitly says ``ERROR_FOUND: false`` (and does not also say true).
Gold is used only after generation, for correctness and oracle_best_of_n.

oracle_best_of_n is an offline upper bound (any candidate correct). It is not
a deployable selector. Majority ignores unparsed answers. Ties break to the
earliest candidate in generation order, never by the gold answer.

Token totals are the logical cost of executing that strategy and budget, even
when the generations were reused from a longer run. Selection rules on the
same candidate set share one cost. A stopping critique is billed even if no
regeneration follows. Later rounds are not billed to a shorter budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

WANDB_CONFIG = (
    REPO_ROOT
    / "wandb/run-20260929_182914-08tgscmo/files/config.yaml"
)
RL_MERGED = (
    REPO_ROOT
    / "checkpoints/PAG/qwen1p5b_cure/global_step_560/hf_merged"
)
MATH500 = (
    Path("/data/yuranli/LLM/2026.04/github_references/Policy-As-GenVerifier")
    / "datasets/math500.parquet"
)

# val_kwargs from the run that wrote checkpoints/PAG/qwen1p5b_cure.
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = -1
MAX_PROMPT_TOKENS = 4096
MAX_RESPONSE_TOKENS = 2048
N_INITIAL = 8
N_MULTI = 4
MAX_ITERS = 7
SEED = 42

PARALLEL_BUDGETS = (1, 2, 4, 8)
ITER_BUDGETS = ((1, 0), (2, 1), (4, 3), (8, 7))  # budget, max T
MULTI_BUDGETS = ((1, 1, 0), (2, 1, 1), (4, 2, 1), (8, 4, 1))  # budget, N, max T


def _read_train_model_path(config_path: Path) -> str:
    """The actor path recorded for this run, not a name-based guess."""
    text = config_path.read_text()
    needle = "path: "
    # actor_rollout_ref.model.path is the first model path in that config dump.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("path: /"):
            return stripped.split("path: ", 1)[1].strip()
    raise RuntimeError(f"No model path in {config_path}")


def _stage_seed(base_seed: int, problem_index: int, trajectory: int, round_idx: int, stage: str) -> int:
    payload = f"{base_seed}|{problem_index}|{trajectory}|{round_idx}|{stage}".encode()
    return zlib.crc32(payload) % (2**31 - 1)


def _truncate(token_ids: list[int], max_len: int, mode: str) -> list[int]:
    if len(token_ids) <= max_len:
        return list(token_ids)
    if mode == "right":
        return list(token_ids[:max_len])
    if mode == "middle":
        left = max_len // 2
        right = max_len - left
        return list(token_ids[:left]) + list(token_ids[-right:])
    raise ValueError(mode)


def _percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile. q is in [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100.0 * len(ordered)))
    return float(ordered[rank - 1])


def _score_answer(data_source: str, text: str, gold: str) -> dict:
    from recipe.cure.reward_score import reward_func

    result = reward_func(data_source, text, gold)
    pred = "" if result.get("pred") is None else str(result["pred"]).strip()
    correct = float(result["score"]) > 0.5
    return {
        "pred": pred,
        "correct": bool(correct),
        "parse_ok": bool(pred),
    }


def _verdict(text: str) -> str:
    """Same rule as RayTrainerCure._critique_verdict. Unparsed is not 'correct'."""
    lower = text.lower()
    says_wrong = "error_found: true" in lower
    says_correct = "error_found: false" in lower
    if says_wrong and not says_correct:
        return "wrong"
    if says_correct and not says_wrong:
        return "correct"
    return "none"


def _solution_for_critique(tokenizer, token_ids: list[int]) -> str:
    """Decode the way update_data_for_critique pastes a solution into the prompt."""
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    eos = tokenizer.eos_token or ""
    if eos:
        text = text.split(eos)[0]
    return text.strip()


def _solution_for_score(tokenizer, token_ids: list[int]) -> str:
    """Decode the way the naive reward manager scores an answer."""
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _chat_ids(tokenizer, user_content: str) -> tuple[str, list[int]]:
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        add_generation_prompt=True,
        tokenize=False,
    )
    ids = tokenizer(templated, add_special_tokens=False)["input_ids"]
    return templated, list(ids)


def _initial_ids(tokenizer, messages: list[dict]) -> tuple[str, list[int]]:
    templated = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    ids = tokenizer(templated, add_special_tokens=False)["input_ids"]
    return templated, list(ids)


def _extract_question(tokenizer, prompt_ids: list[int]) -> str:
    from verl.utils.cure_utils import _extract_user_question

    return _extract_user_question([prompt_ids], tokenizer)[0]


def _build_critique(tokenizer, question: str, solution_text: str) -> tuple[str, list[int], str]:
    from verl.utils.cure_utils import INSTRUCT_CRITIQUE_PROMPT

    formatted = INSTRUCT_CRITIQUE_PROMPT.format(question=question, solution=solution_text)
    templated, ids = _chat_ids(tokenizer, formatted)
    ids = _truncate(ids, MAX_PROMPT_TOKENS, "middle")
    fed = tokenizer.decode(ids, skip_special_tokens=False)
    return fed, ids, formatted


def _hint_from_critique(critique_text: str) -> str:
    from verl.utils.cure_utils import extract_high_level_hint

    if "HIGH_LEVEL_HINT" in critique_text or "ERROR_FOUND" in critique_text:
        return extract_high_level_hint(critique_text)
    return critique_text.split("<|im_end|>")[0].strip()


def _problem_span(text: str) -> str:
    import re

    match = re.search(r"Problem:\n(.*?)AI Solution:\n", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _build_regen(tokenizer, critique_formatted: str, critique_text: str) -> tuple[str, list[int], str, str]:
    from verl.utils.cure_utils import GUIDED_REGENERATION_PROMPT

    question = _problem_span(critique_formatted)
    hint = _hint_from_critique(critique_text)
    formatted = GUIDED_REGENERATION_PROMPT.format(question=question, critique=hint)
    templated, ids = _chat_ids(tokenizer, formatted)
    ids = _truncate(ids, MAX_PROMPT_TOKENS, "right")
    fed = tokenizer.decode(ids, skip_special_tokens=False)
    return fed, ids, formatted, hint


def _pack_solution(tokenizer, data_source: str, gold: str, prompt_text: str, prompt_ids: list[int], output_ids: list[int]) -> dict:
    pasted = _solution_for_critique(tokenizer, output_ids)
    scored = _solution_for_score(tokenizer, output_ids)
    judged = _score_answer(data_source, scored, gold)
    return {
        "input_text": prompt_text,
        "output_text": tokenizer.decode(output_ids, skip_special_tokens=False),
        "solution_text": pasted,
        "scored_text": scored,
        "input_tokens": len(prompt_ids),
        "output_tokens": len(output_ids),
        **judged,
    }


def _generate(llm, sampling_cls, prompt_ids: list[list[int]], seeds: list[int]):
    params = [
        sampling_cls(
            n=1,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            max_tokens=MAX_RESPONSE_TOKENS,
            seed=seed,
        )
        for seed in seeds
    ]
    return llm.generate(prompt_token_ids=prompt_ids, sampling_params=params, use_tqdm=False)


def _output_ids(request) -> list[int]:
    completion = request.outputs[0]
    ids = completion.token_ids
    return list(ids)


def _worker(gpu_id: int, shard: list[dict], model_path: str, base_seed: int, out_path: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=MAX_PROMPT_TOKENS + MAX_RESPONSE_TOKENS,
        trust_remote_code=True,
        seed=base_seed,
    )
    prepared = []
    for row in shard:
        messages = row["messages"]
        prompt_text, prompt_ids = _initial_ids(tokenizer, messages)
        prompt_ids = _truncate(prompt_ids, MAX_PROMPT_TOKENS, "right")
        question = _extract_question(tokenizer, prompt_ids)
        user = next(m["content"] for m in messages if m["role"] == "user")
        if question.strip() != user.strip():
            raise RuntimeError(
                f"Question extraction mismatch on {row['problem_id']}: {question[:80]!r} vs {user[:80]!r}"
            )
        prepared.append({**row, "init_text": prompt_text, "init_ids": prompt_ids, "question": question})

    # One request per trajectory so each answer has its own seed.
    flat_ids = []
    flat_seeds = []
    owners = []
    for row_i, row in enumerate(prepared):
        for traj in range(N_INITIAL):
            flat_ids.append(row["init_ids"])
            flat_seeds.append(_stage_seed(base_seed, row["problem_index"], traj, 0, "initial"))
            owners.append(row_i)
    init_requests = _generate(llm, SamplingParams, flat_ids, flat_seeds)
    initials: list[list[dict]] = [[] for _ in prepared]
    for request, row_i in zip(init_requests, owners):
        row = prepared[row_i]
        output_ids = _output_ids(request)
        fed_len = len(request.prompt_token_ids) if request.prompt_token_ids is not None else len(row["init_ids"])
        packed = _pack_solution(
            tokenizer, row["data_source"], row["gold"], row["init_text"], list(request.prompt_token_ids or row["init_ids"]), output_ids
        )
        if packed["input_tokens"] != fed_len:
            packed["input_tokens"] = fed_len
        initials[row_i].append(packed)

    # rounds[row][traj] = list of {verdict, critique, regen or None}
    rounds: list[list[list[dict]]] = [[[] for _ in range(N_MULTI)] for _ in prepared]
    stop_reasons = [["max_rounds" for _ in range(N_MULTI)] for _ in prepared]

    def run_round(pairs: list[tuple[int, int]], round_idx: int) -> list[tuple[int, int]]:
        if not pairs:
            return []
        critique_ids = []
        critique_text_in = []
        formatted = []
        for row_i, traj in pairs:
            current = initials[row_i][traj]["solution_text"] if round_idx == 0 else rounds[row_i][traj][-1]["regen"]["solution_text"]
            fed, ids, formatted_prompt = _build_critique(tokenizer, prepared[row_i]["question"], current)
            critique_ids.append(ids)
            critique_text_in.append(fed)
            formatted.append(formatted_prompt)
        requests = _generate(
            llm,
            SamplingParams,
            critique_ids,
            [
                _stage_seed(base_seed, prepared[row_i]["problem_index"], traj, round_idx, "critique")
                for row_i, traj in pairs
            ],
        )
        continue_pairs = []
        regen_inputs = []
        for (row_i, traj), request, fed, formatted_prompt, prompt_ids in zip(
            pairs, requests, critique_text_in, formatted, critique_ids
        ):
            output_ids = _output_ids(request)
            critique_raw = tokenizer.decode(output_ids, skip_special_tokens=False)
            verdict = _verdict(critique_raw)
            prompt_ids_fed = list(request.prompt_token_ids or prompt_ids)
            critique_rec = {
                "input_text": tokenizer.decode(prompt_ids_fed, skip_special_tokens=False),
                "output_text": critique_raw,
                "input_tokens": len(prompt_ids_fed),
                "output_tokens": len(output_ids),
                "verdict": verdict,
            }
            record = {"verdict": verdict, "critique": critique_rec, "regen": None, "hint": ""}
            rounds[row_i][traj].append(record)
            if verdict == "correct":
                stop_reasons[row_i][traj] = "error_found_false"
                continue
            regen_fed, regen_ids, regen_formatted, hint = _build_regen(tokenizer, formatted_prompt, critique_raw)
            record["hint"] = hint
            if "AI Solution:" in regen_formatted:
                raise RuntimeError("Regen prompt includes the AI Solution block; that is not the native template.")
            regen_inputs.append((row_i, traj, regen_fed, regen_ids, len(rounds[row_i][traj]) - 1))
            continue_pairs.append((row_i, traj))
        if not regen_inputs:
            return []
        regen_requests = _generate(
            llm,
            SamplingParams,
            [item[3] for item in regen_inputs],
            [
                _stage_seed(base_seed, prepared[row_i]["problem_index"], traj, round_idx, "regen")
                for row_i, traj, _, _, _ in regen_inputs
            ],
        )
        for (row_i, traj, regen_fed, regen_ids, round_slot), request in zip(regen_inputs, regen_requests):
            output_ids = _output_ids(request)
            prompt_ids_fed = list(request.prompt_token_ids or regen_ids)
            packed = _pack_solution(
                tokenizer,
                prepared[row_i]["data_source"],
                prepared[row_i]["gold"],
                tokenizer.decode(prompt_ids_fed, skip_special_tokens=False),
                prompt_ids_fed,
                output_ids,
            )
            rounds[row_i][traj][round_slot]["regen"] = packed
        return continue_pairs

    # Round 0 on the four multi-chain trajectories. Trajectory 0 is also the iterative chain.
    active = run_round([(row_i, traj) for row_i in range(len(prepared)) for traj in range(N_MULTI)], 0)
    active = [(row_i, traj) for row_i, traj in active if traj == 0]
    for round_idx in range(1, MAX_ITERS):
        active = run_round(active, round_idx)
        active = [(row_i, traj) for row_i, traj in active if traj == 0]

    out_lines = []
    for row_i, row in enumerate(prepared):
        chains = []
        for traj in range(N_MULTI):
            chains.append({"rounds": rounds[row_i][traj], "stop_reason": stop_reasons[row_i][traj]})
        out_lines.append(
            {
                "problem_index": row["problem_index"],
                "problem_id": row["problem_id"],
                "data_source": row["data_source"],
                "gold": row["gold"],
                "question": row["question"],
                "initials": initials[row_i],
                "chains": chains,
            }
        )
    Path(out_path).write_text("\n".join(json.dumps(line, ensure_ascii=False) for line in out_lines) + "\n")
    print(f"[gpu {gpu_id}] wrote {len(out_lines)} problems to {out_path}", flush=True)


def _sum_tokens(records: list[dict]) -> tuple[int, int]:
    return (
        sum(int(r["input_tokens"]) for r in records),
        sum(int(r["output_tokens"]) for r in records),
    )


def _history(problem: dict, max_t: int) -> list[dict]:
    sols = [problem["chains"][0]["initial_slot"]]
    for rnd in problem["chains"][0]["rounds"][:max_t]:
        if rnd["regen"] is not None:
            sols.append(rnd["regen"])
    return sols


def _terminal(chain: dict, initial: dict, max_t: int) -> dict:
    sol = initial
    for rnd in chain["rounds"][:max_t]:
        if rnd["regen"] is not None:
            sol = rnd["regen"]
    return sol


def _stop_for_budget(chain: dict, max_t: int) -> str:
    if max_t <= 0:
        return "no_refinement"
    executed = chain["rounds"][:max_t]
    if not executed:
        return "no_refinement"
    last = executed[-1]
    if last["regen"] is None and last["verdict"] == "correct":
        return "error_found_false"
    if len(executed) < max_t:
        return chain["stop_reason"]
    return "max_rounds"


def _select_one(candidate: dict) -> tuple[bool, bool]:
    return bool(candidate["correct"]), not bool(candidate["parse_ok"])


def _select_majority(candidates: list[dict]) -> tuple[bool, bool]:
    buckets: dict[str, list[int]] = {}
    for index, candidate in enumerate(candidates):
        if not candidate["parse_ok"]:
            continue
        buckets.setdefault(candidate["pred"], []).append(index)
    if not buckets:
        return False, True
    best = max(buckets, key=lambda pred: (len(buckets[pred]), -buckets[pred][0]))
    chosen = candidates[buckets[best][0]]
    return bool(chosen["correct"]), not bool(chosen["parse_ok"])


def _select_oracle(candidates: list[dict]) -> tuple[bool, bool]:
    return any(bool(c["correct"]) for c in candidates), all(not c["parse_ok"] for c in candidates)


def _cost_from_parts(initials: list[dict], critiques: list[dict], regens: list[dict]) -> dict:
    ini_in, ini_out = _sum_tokens(initials)
    cri_in, cri_out = _sum_tokens(critiques)
    reg_in, reg_out = _sum_tokens(regens)
    total_in = ini_in + cri_in + reg_in
    total_out = ini_out + cri_out + reg_out
    return {
        "n_solutions": len(initials) + len(regens),
        "n_critiques": len(critiques),
        "n_refinements": len(regens),
        "initial_input_tokens": ini_in,
        "initial_output_tokens": ini_out,
        "critique_input_tokens": cri_in,
        "critique_output_tokens": cri_out,
        "regen_input_tokens": reg_in,
        "regen_output_tokens": reg_out,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_tokens": total_in + total_out,
    }


def _iter_cost(problem: dict, max_t: int) -> dict:
    critiques = []
    regens = []
    for rnd in problem["chains"][0]["rounds"][:max_t]:
        critiques.append(rnd["critique"])
        if rnd["regen"] is not None:
            regens.append(rnd["regen"])
    return _cost_from_parts([problem["initials"][0]], critiques, regens)


def _multi_cost(problem: dict, n_traj: int, max_t: int) -> dict:
    critiques = []
    regens = []
    for traj in range(n_traj):
        for rnd in problem["chains"][traj]["rounds"][:max_t]:
            critiques.append(rnd["critique"])
            if rnd["regen"] is not None:
                regens.append(rnd["regen"])
    return _cost_from_parts(problem["initials"][:n_traj], critiques, regens)


def _multi_terminals(problem: dict, n_traj: int, max_t: int) -> list[dict]:
    sols = []
    for traj in range(n_traj):
        if max_t <= 0:
            sols.append(problem["initials"][traj])
        else:
            sols.append(_terminal(problem["chains"][traj], problem["initials"][traj], max_t))
    return sols


def _attach_initial_slots(problem: dict) -> dict:
    """Point chain 0's history at the same objects as the independent initials."""
    for traj, chain in enumerate(problem["chains"]):
        chain["initial_slot"] = problem["initials"][traj]
    return problem


def evaluate_problem(problem: dict, checkpoint: str) -> list[dict]:
    problem = _attach_initial_slots(problem)
    rows = []

    def emit(strategy, budget, n, t, rule, correct, selected_fail, candidates, cost, stop_reason):
        rows.append(
            {
                "checkpoint": checkpoint,
                "problem_id": problem["problem_id"],
                "problem_index": problem["problem_index"],
                "strategy": strategy,
                "budget": budget,
                "n": n,
                "t": t,
                "selection_rule": rule,
                "correct": bool(correct),
                "selected_parse_fail": bool(selected_fail),
                "candidate_parse_fail": sum(not c["parse_ok"] for c in candidates),
                "n_candidates": len(candidates),
                "stop_reason": stop_reason,
                **cost,
            }
        )

    for budget in PARALLEL_BUDGETS:
        cands = problem["initials"][:budget]
        cost = _cost_from_parts(cands, [], [])
        if budget == 1:
            correct, fail = _select_one(cands[0])
            emit("parallel", budget, budget, "", "single", correct, fail, cands, cost, "no_refinement")
        else:
            correct, fail = _select_majority(cands)
            emit("parallel", budget, budget, "", "majority", correct, fail, cands, cost, "no_refinement")
            correct, fail = _select_oracle(cands)
            emit("parallel", budget, budget, "", "oracle_best_of_n", correct, fail, cands, cost, "no_refinement")

    for budget, max_t in ITER_BUDGETS:
        cost = _iter_cost(problem, max_t)
        hist = [problem["initials"][0]]
        for rnd in problem["chains"][0]["rounds"][:max_t]:
            if rnd["regen"] is not None:
                hist.append(rnd["regen"])
        stop = _stop_for_budget(problem["chains"][0], max_t)
        terminal = hist[-1]
        if budget == 1:
            correct, fail = _select_one(terminal)
            emit("iterative", budget, 1, "", "terminal", correct, fail, hist, cost, stop)
        else:
            correct, fail = _select_one(terminal)
            emit("iterative", budget, 1, max_t, "terminal", correct, fail, hist, cost, stop)
            correct, fail = _select_majority(hist)
            emit("iterative", budget, 1, max_t, "history_majority", correct, fail, hist, cost, stop)
            correct, fail = _select_oracle(hist)
            emit("iterative", budget, 1, max_t, "oracle_best_of_n", correct, fail, hist, cost, stop)

    for budget, n_traj, max_t in MULTI_BUDGETS:
        cost = _multi_cost(problem, n_traj, max_t)
        terminals = _multi_terminals(problem, n_traj, max_t)
        stops = [_stop_for_budget(problem["chains"][traj], max_t) for traj in range(n_traj)]
        stop = stops[0] if len(set(stops)) == 1 else "|".join(stops)
        if budget == 1:
            correct, fail = _select_one(terminals[0])
            emit("multi", budget, 1, "", "terminal", correct, fail, terminals, cost, stop)
        else:
            correct, fail = _select_majority(terminals)
            emit("multi", budget, n_traj, max_t, "terminal_majority", correct, fail, terminals, cost, stop)
            correct, fail = _select_oracle(terminals)
            emit("multi", budget, n_traj, max_t, "oracle_best_of_n", correct, fail, terminals, cost, stop)
    return rows


def flatten_calls(problem: dict, checkpoint: str) -> list[dict]:
    calls = []
    for traj, sol in enumerate(problem["initials"]):
        calls.append(
            {
                "checkpoint": checkpoint,
                "problem_id": problem["problem_id"],
                "problem_index": problem["problem_index"],
                "trajectory": traj,
                "round": 0,
                "stage": "initial",
                "input_text": sol["input_text"],
                "output_text": sol["output_text"],
                "pred": sol["pred"],
                "correct": sol["correct"],
                "parse_ok": sol["parse_ok"],
                "input_tokens": sol["input_tokens"],
                "output_tokens": sol["output_tokens"],
                "total_tokens": sol["input_tokens"] + sol["output_tokens"],
                "verdict": "",
                "stop_reason": "",
                "seed": _stage_seed(SEED, problem["problem_index"], traj, 0, "initial"),
            }
        )
    for traj, chain in enumerate(problem["chains"]):
        for round_idx, rnd in enumerate(chain["rounds"]):
            critique = rnd["critique"]
            calls.append(
                {
                    "checkpoint": checkpoint,
                    "problem_id": problem["problem_id"],
                    "problem_index": problem["problem_index"],
                    "trajectory": traj,
                    "round": round_idx,
                    "stage": "critique",
                    "input_text": critique["input_text"],
                    "output_text": critique["output_text"],
                    "pred": "",
                    "correct": "",
                    "parse_ok": "",
                    "input_tokens": critique["input_tokens"],
                    "output_tokens": critique["output_tokens"],
                    "total_tokens": critique["input_tokens"] + critique["output_tokens"],
                    "verdict": rnd["verdict"],
                    "stop_reason": "error_found_false" if rnd["verdict"] == "correct" else "",
                    "seed": _stage_seed(SEED, problem["problem_index"], traj, round_idx, "critique"),
                }
            )
            if rnd["regen"] is not None:
                regen = rnd["regen"]
                calls.append(
                    {
                        "checkpoint": checkpoint,
                        "problem_id": problem["problem_id"],
                        "problem_index": problem["problem_index"],
                        "trajectory": traj,
                        "round": round_idx,
                        "stage": "regen",
                        "input_text": regen["input_text"],
                        "output_text": regen["output_text"],
                        "pred": regen["pred"],
                        "correct": regen["correct"],
                        "parse_ok": regen["parse_ok"],
                        "input_tokens": regen["input_tokens"],
                        "output_tokens": regen["output_tokens"],
                        "total_tokens": regen["input_tokens"] + regen["output_tokens"],
                        "verdict": "",
                        "stop_reason": "",
                        "seed": _stage_seed(SEED, problem["problem_index"], traj, round_idx, "regen"),
                    }
                )
    return calls


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["checkpoint"], row["strategy"], row["budget"], row["n"], row["t"], row["selection_rule"])
        groups.setdefault(key, []).append(row)
    summary = []
    for key, items in groups.items():
        checkpoint, strategy, budget, n, t, rule = key
        totals = [r["total_tokens"] for r in items]
        n_problems = len(items)
        summary.append(
            {
                "checkpoint": checkpoint,
                "strategy": strategy,
                "budget": budget,
                "n": n,
                "t": t,
                "selection_rule": rule,
                "n_problems": n_problems,
                "accuracy": sum(bool(r["correct"]) for r in items) / n_problems,
                "selected_parse_fail_count": sum(bool(r["selected_parse_fail"]) for r in items),
                "candidate_parse_fail_count": sum(int(r["candidate_parse_fail"]) for r in items),
                "mean_n_solutions": sum(r["n_solutions"] for r in items) / n_problems,
                "mean_n_critiques": sum(r["n_critiques"] for r in items) / n_problems,
                "mean_n_refinements": sum(r["n_refinements"] for r in items) / n_problems,
                "mean_initial_input_tokens": sum(r["initial_input_tokens"] for r in items) / n_problems,
                "mean_initial_output_tokens": sum(r["initial_output_tokens"] for r in items) / n_problems,
                "mean_critique_input_tokens": sum(r["critique_input_tokens"] for r in items) / n_problems,
                "mean_critique_output_tokens": sum(r["critique_output_tokens"] for r in items) / n_problems,
                "mean_regen_input_tokens": sum(r["regen_input_tokens"] for r in items) / n_problems,
                "mean_regen_output_tokens": sum(r["regen_output_tokens"] for r in items) / n_problems,
                "mean_total_input_tokens": sum(r["total_input_tokens"] for r in items) / n_problems,
                "mean_total_output_tokens": sum(r["total_output_tokens"] for r in items) / n_problems,
                "mean_total_tokens": sum(totals) / n_problems,
                "median_total_tokens": _percentile(totals, 50),
                "p90_total_tokens": _percentile(totals, 90),
            }
        )
    summary.sort(key=lambda r: (r["checkpoint"], r["strategy"], r["budget"], r["selection_rule"]))
    return summary


def _sol(pred: str, correct: bool, in_tok=10, out_tok=20) -> dict:
    return {
        "pred": pred,
        "correct": correct,
        "parse_ok": bool(pred),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "input_text": "in",
        "output_text": pred or "bad",
        "solution_text": pred or "bad",
    }


def _critique(verdict: str, in_tok=30, out_tok=15) -> dict:
    return {
        "verdict": verdict,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "input_text": "critique-in",
        "output_text": f"ERROR_FOUND: {verdict}",
    }


def _round(verdict: str, regen: dict | None, cin=30, cout=15) -> dict:
    return {"verdict": verdict, "critique": _critique(verdict, cin, cout), "regen": regen, "hint": "h"}


def run_self_test() -> None:
    # Early stop after the first critique: higher budgets must not duplicate the answer.
    stopped = {
        "problem_id": "p",
        "problem_index": 0,
        "initials": [_sol("a", False, 11, 21), _sol("b", True), _sol("", False), _sol("a", False)],
        "chains": [
            {"rounds": [_round("correct", None, 100, 40)], "stop_reason": "error_found_false"},
            {"rounds": [_round("wrong", _sol("c", True, 50, 60))], "stop_reason": "max_rounds"},
            {"rounds": [_round("wrong", _sol("a", False, 50, 60))], "stop_reason": "max_rounds"},
            {"rounds": [_round("correct", None)], "stop_reason": "error_found_false"},
        ],
    }
    rows = { (r["strategy"], r["budget"], r["selection_rule"]): r for r in evaluate_problem(stopped, "base") }
    early = rows[("iterative", 8, "terminal")]
    assert early["correct"] is False
    assert early["n_solutions"] == 1
    assert early["n_critiques"] == 1
    assert early["n_refinements"] == 0
    assert early["total_input_tokens"] == 11 + 100
    assert early["stop_reason"] == "error_found_false"
    assert rows[("iterative", 2, "history_majority")]["n_candidates"] == 1
    assert rows[("iterative", 1, "terminal")]["n_critiques"] == 0
    assert rows[("iterative", 1, "terminal")]["total_tokens"] == 11 + 21

    # One regen, then a later critique would be round 2. Budget 2 must not bill it.
    # Build a chain that continues once more so the prefix cut is visible.
    cont = {
        "problem_id": "q",
        "problem_index": 1,
        "initials": [_sol("a", False, 5, 6)] + [_sol("z", False) for _ in range(7)],
        "chains": [
            {
                "rounds": [
                    _round("wrong", _sol("b", True, 7, 8), 9, 10),
                    _round("correct", None, 11, 12),
                ],
                "stop_reason": "error_found_false",
            }
        ]
        + [{"rounds": [], "stop_reason": "max_rounds"} for _ in range(3)],
    }
    rows = {(r["strategy"], r["budget"], r["selection_rule"]): r for r in evaluate_problem(cont, "base")}
    b2 = rows[("iterative", 2, "oracle_best_of_n")]
    assert b2["correct"] is True
    assert b2["n_solutions"] == 2
    assert b2["n_critiques"] == 1
    assert b2["n_refinements"] == 1
    assert b2["critique_input_tokens"] == 9
    assert b2["total_input_tokens"] == 5 + 9 + 7
    b4 = rows[("iterative", 4, "terminal")]
    assert b4["n_critiques"] == 2  # stopping critique is billed
    assert b4["n_refinements"] == 1
    assert b4["n_solutions"] == 2  # stopping does not add another vote
    assert b4["correct"] is True
    assert rows[("iterative", 4, "history_majority")]["n_candidates"] == 2

    # Majority tie uses the earliest answer, not gold. Oracle is separate.
    tie = {
        "problem_id": "t",
        "problem_index": 2,
        "initials": [
            _sol("wrong", False),
            _sol("right", True),
            _sol("wrong", False),
            _sol("right", True),
        ]
        + [_sol("x", False) for _ in range(4)],
        "chains": [{"rounds": [], "stop_reason": "max_rounds"} for _ in range(4)],
    }
    rows = {(r["strategy"], r["budget"], r["selection_rule"]): r for r in evaluate_problem(tie, "base")}
    assert rows[("parallel", 4, "majority")]["correct"] is False  # "wrong" appears first
    assert rows[("parallel", 4, "oracle_best_of_n")]["correct"] is True
    assert rows[("parallel", 2, "majority")]["correct"] is False
    assert rows[("parallel", 1, "single")]["n_solutions"] == 1
    assert rows[("parallel", 8, "majority")]["n_solutions"] == 8

    # Multi terminal excludes the initial once a regen exists. N=1 equals terminal accuracy.
    multi = {
        "problem_id": "m",
        "problem_index": 3,
        "initials": [_sol("bad", False, 1, 2), _sol("bad2", False, 3, 4), _sol("good", True), _sol("good2", True)],
        "chains": [
            {"rounds": [_round("wrong", _sol("fixed", True, 8, 9))], "stop_reason": "max_rounds"},
            {"rounds": [_round("wrong", _sol("still", False, 8, 9))], "stop_reason": "max_rounds"},
            {"rounds": [_round("correct", None)], "stop_reason": "error_found_false"},
            {"rounds": [_round("wrong", _sol("ok", True, 8, 9))], "stop_reason": "max_rounds"},
        ],
    }
    rows = {(r["strategy"], r["budget"], r["selection_rule"]): r for r in evaluate_problem(multi, "base")}
    assert rows[("multi", 2, "terminal_majority")]["correct"] is True
    assert rows[("multi", 2, "oracle_best_of_n")]["correct"] is True
    assert rows[("multi", 2, "terminal_majority")]["n_solutions"] == 2  # initial + regen
    assert rows[("multi", 2, "terminal_majority")]["n_candidates"] == 1
    # N=2 terminals: fixed (correct), still (wrong). Majority earliest is fixed.
    assert rows[("multi", 4, "terminal_majority")]["correct"] is True
    # N=4 terminals: fixed, still, good (initial, stopped), ok. votes: each once, earliest fixed is correct.
    assert rows[("multi", 8, "oracle_best_of_n")]["correct"] is True
    # Budget 1 does not bill the critique.
    assert rows[("multi", 1, "terminal")]["n_critiques"] == 0
    assert rows[("multi", 1, "terminal")]["correct"] is False
    # Unparsed majority abstains and is a selected parse failure. Denominator still includes it.
    unparsed = {
        "problem_id": "u",
        "problem_index": 4,
        "initials": [_sol("", False) for _ in range(8)],
        "chains": [{"rounds": [], "stop_reason": "max_rounds"} for _ in range(4)],
    }
    rows = {(r["strategy"], r["budget"], r["selection_rule"]): r for r in evaluate_problem(unparsed, "base")}
    assert rows[("parallel", 2, "majority")]["correct"] is False
    assert rows[("parallel", 2, "majority")]["selected_parse_fail"] is True
    summary = summarize(evaluate_problem(unparsed, "base") + evaluate_problem(tie, "base"))
    parallel = [r for r in summary if r["strategy"] == "parallel" and r["budget"] == 1 and r["selection_rule"] == "single"]
    assert parallel[0]["n_problems"] == 2
    print("self-test ok", flush=True)


def _load_problems(limit: int) -> list[dict]:
    import pandas as pd

    frame = pd.read_parquet(MATH500)
    if limit:
        frame = frame.iloc[:limit]
    problems = []
    for index, row in frame.iterrows():
        prompt = row["prompt"]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        messages = [{"role": m["role"], "content": m["content"]} for m in prompt]
        reward = row["reward_model"]
        if hasattr(reward, "item"):
            reward = reward.item()
        problems.append(
            {
                "problem_index": int(row["extra_info"]["index"]) if isinstance(row["extra_info"], dict) else int(index),
                "problem_id": str(row["unique_id"]),
                "data_source": str(row["data_source"]),
                "gold": str(reward["ground_truth"]),
                "messages": messages,
            }
        )
    return problems


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(problems: list[dict], checkpoint: str, out_dir: Path) -> list[dict]:
    problems = sorted(problems, key=lambda row: row["problem_index"])
    per_problem = []
    calls = []
    for problem in problems:
        per_problem.extend(evaluate_problem(problem, checkpoint))
        calls.extend(flatten_calls(problem, checkpoint))
    _write_jsonl(out_dir / "per_problem.jsonl", per_problem)
    _write_jsonl(out_dir / "calls.jsonl", calls)
    return per_problem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--models", nargs="+", default=["base", "rl"], choices=["base", "rl"])
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs/math500_budget_eval"),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return

    base_path = _read_train_model_path(WANDB_CONFIG)
    model_paths = {"base": base_path, "rl": str(RL_MERGED)}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "wandb_config": str(WANDB_CONFIG),
        "base_model_path": base_path,
        "rl_checkpoint": str(RL_MERGED),
        "rl_fsdp_checkpoint": str(REPO_ROOT / "checkpoints/PAG/qwen1p5b_cure/global_step_560"),
        "dataset": str(MATH500),
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "max_response_tokens": MAX_RESPONSE_TOKENS,
        "seed": args.seed,
        "n_initial": N_INITIAL,
        "max_iterative_rounds": MAX_ITERS,
        "stop_rule": "regen unless critique verdict == correct (ERROR_FOUND: false and not true). Gold is not a stop signal.",
        "oracle_best_of_n": "at least one candidate correct; not a deployable selector",
        "majority_tie": "earliest candidate in generation order; unparsed answers abstain",
        "budget_definition": "max solution count N*(1+T). Critique tokens are reported separately and are not part of Budget.",
        "sampling_source": "actor_rollout_ref.rollout.val_kwargs of wandb run 08tgscmo",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    problems = _load_problems(args.limit)
    all_rows = []
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    for name in args.models:
        model_path = model_paths[name]
        if not Path(model_path).exists():
            raise FileNotFoundError(model_path)
        print(f"[eval] {name} {model_path} problems={len(problems)}", flush=True)
        tmp_dir = out_dir / name
        tmp_dir.mkdir(parents=True, exist_ok=True)
        processes = []
        shard_paths = []
        n_gpus = min(args.num_gpus, len(problems))
        for gpu_id in range(n_gpus):
            shard = [row for i, row in enumerate(problems) if i % n_gpus == gpu_id]
            shard_path = tmp_dir / f".shard{gpu_id}.jsonl"
            shard_paths.append(shard_path)
            if not shard:
                continue
            proc = ctx.Process(
                target=_worker,
                args=(gpu_id, shard, model_path, args.seed, str(shard_path)),
            )
            proc.start()
            processes.append(proc)
        for proc in processes:
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError(f"worker failed with exit code {proc.exitcode}")
        merged = []
        for shard_path in shard_paths:
            if not shard_path.exists():
                continue
            for line in shard_path.read_text().splitlines():
                if line.strip():
                    merged.append(json.loads(line))
        if len(merged) != len(problems):
            raise RuntimeError(f"{name}: expected {len(problems)} problems, got {len(merged)}")
        _write_jsonl(tmp_dir / "problems.jsonl", merged)
        all_rows.extend(_aggregate(merged, name, tmp_dir))
    summary = summarize(all_rows)
    _write_csv(out_dir / "summary.csv", summary)
    for row in summary:
        print(
            f"{row['checkpoint']:4} {row['strategy']:10} B={row['budget']:<2} {row['selection_rule']:20} "
            f"acc={row['accuracy']:.4f} sols={row['mean_n_solutions']:.2f} "
            f"tokens={row['mean_total_tokens']:.0f} parse_fail={row['selected_parse_fail_count']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
