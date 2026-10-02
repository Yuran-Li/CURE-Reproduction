"""Estimate CURE's difficulty bins on a PAG parquet with the shared answer reward.

For each problem, draw K solutions from the initial checkpoint at the training
temperature and set Acc(x) = (1/K) * sum_i r_i. Bins:

    zero    Acc == 0
    partial 0 < Acc < 1
    full    Acc == 1

CURE's critique sampler needs a non-trivial zero + partial mass. Report the
three probabilities before reading a CURE vs PAG comparison.
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from recipe.cure.reward_score import reward_func


def _messages(prompt):
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    return [{"role": m["role"], "content": m["content"]} for m in prompt]


def _score_value(result):
    if isinstance(result, dict):
        return float(result["score"])
    return float(result)


def _worker(gpu_id, prompts, answers, sources, k, temperature, max_tokens, model_path, out_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    texts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in prompts
    ]
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=max_tokens + 4096,
        trust_remote_code=True,
    )
    params = SamplingParams(n=k, temperature=temperature, top_p=1.0, max_tokens=max_tokens)
    outputs = llm.generate(texts, params)
    rows = []
    for sample, answer, source, output in zip(prompts, answers, sources, outputs):
        rewards = []
        for completion in output.outputs:
            result = reward_func(source, completion.text, answer)
            rewards.append(_score_value(result))
        acc = sum(rewards) / len(rewards)
        if acc == 0.0:
            bin_name = "zero"
        elif acc == 1.0:
            bin_name = "full"
        else:
            bin_name = "partial"
        rows.append({"acc": acc, "bin": bin_name, "rewards": rewards, "data_source": source})
    Path(out_path).write_text(json.dumps(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Optional prefix of the parquet. 0 means all rows.")
    args = parser.parse_args()

    frame = pd.read_parquet(args.parquet)
    if args.limit:
        frame = frame.iloc[: args.limit]
    prompts = [_messages(p) for p in frame["prompt"].tolist()]
    answers = [row["ground_truth"] for row in frame["reward_model"].tolist()]
    sources = frame["data_source"].tolist()

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    shards = args.num_gpus
    processes = []
    tmp_paths = []
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for gpu_id in range(shards):
        sl = slice(gpu_id, None, shards)
        tmp = out_dir / f".pass_rate_gpu{gpu_id}.json"
        tmp_paths.append(tmp)
        if not prompts[sl]:
            tmp.write_text("[]")
            continue
        proc = ctx.Process(
            target=_worker,
            args=(
                gpu_id,
                prompts[sl],
                answers[sl],
                sources[sl],
                args.k,
                args.temperature,
                args.max_tokens,
                args.model_path,
                str(tmp),
            ),
        )
        proc.start()
        processes.append(proc)
    for proc in processes:
        proc.join()
        if proc.exitcode != 0:
            raise SystemExit(f"pass-rate worker failed with exit code {proc.exitcode}")

    rows = []
    for tmp in tmp_paths:
        rows.extend(json.loads(tmp.read_text()))
        tmp.unlink(missing_ok=True)

    n = len(rows)
    counts = {name: sum(r["bin"] == name for r in rows) for name in ("zero", "partial", "full")}
    summary = {
        "model_path": args.model_path,
        "parquet": args.parquet,
        "n": n,
        "k": args.k,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "counts": counts,
        "p_zero": counts["zero"] / n,
        "p_partial": counts["partial"] / n,
        "p_full": counts["full"] / n,
        "mean_acc": sum(r["acc"] for r in rows) / n,
    }
    payload = {"summary": summary, "rows": rows}
    Path(args.output).write_text(json.dumps(payload))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
