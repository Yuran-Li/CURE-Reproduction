# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Metrics related to the PPO trainer.
"""

from collections import defaultdict
from functools import partial
from typing import Any, Callable

import numpy as np
import torch

from verl import DataProto
from verl.utils.import_utils import deprecated


@deprecated("verl.utils.metric.reduce_metrics")
def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean of each list.

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its mean value.

    Example:
        >>> metrics = {"loss": [1.0, 2.0, 3.0], "accuracy": [0.8, 0.9, 0.7]}
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8}
    """
    from verl.utils.metric import reduce_metrics

    return reduce_metrics(metrics)


def _compute_response_info(batch: DataProto) -> dict[str, Any]:
    """
    Computes information about prompts and responses from a batch.

    This is an internal helper function that extracts masks and lengths for prompts and responses.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.

    Returns:
        A dictionary containing:
            - response_mask: Attention mask for the response tokens
            - prompt_length: Tensor of prompt lengths for each item in the batch
            - response_length: Tensor of response lengths for each item in the batch
    """
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
            - num_turns/mean, max, min: Statistics about the number of multi-turn conversations
    """
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["response_mask"].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    aborted_mask = (response_length == 0).bool()
    non_aborted_mask = ~aborted_mask

    non_aborted_sequence_score = sequence_score[non_aborted_mask]
    non_aborted_sequence_reward = sequence_reward[non_aborted_mask]

    score_mean = torch.mean(non_aborted_sequence_score).detach().item()
    score_max = torch.max(non_aborted_sequence_score).detach().item()
    score_min = torch.min(non_aborted_sequence_score).detach().item()

    reward_mean = torch.mean(non_aborted_sequence_reward).detach().item()
    reward_max = torch.max(non_aborted_sequence_reward).detach().item()
    reward_min = torch.min(non_aborted_sequence_reward).detach().item()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    # Aborted samples and non-aborted response length statistics
    # response_length_non_aborted/*: statistics computed on non-aborted samples only
    aborted_ratio = torch.mean(aborted_mask.float()).detach().item()

    non_aborted_response_length = response_length[non_aborted_mask]
    if non_aborted_response_length.numel() > 0:
        non_aborted_response_length_mean = torch.mean(non_aborted_response_length).detach().item()
        non_aborted_response_length_max = torch.max(non_aborted_response_length).detach().item()
        non_aborted_response_length_min = torch.min(non_aborted_response_length).detach().item()
        non_aborted_response_length_clip_ratio = (
            torch.mean(torch.eq(non_aborted_response_length, max_response_length).float()).detach().item()
        )
    else:
        raise ValueError("All samples are aborted, this should not happen.")

    metrics = {
        # score
        "critic/score/mean": score_mean,
        "critic/score/max": score_max,
        "critic/score/min": score_min,
        # reward
        "critic/rewards/mean": reward_mean,
        "critic/rewards/max": reward_max,
        "critic/rewards/min": reward_min,
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # response length (non-aborted only)
        # These statistics exclude aborted samples to avoid skew from zeros
        "response_length_non_aborted/mean": non_aborted_response_length_mean,
        "response_length_non_aborted/max": non_aborted_response_length_max,
        "response_length_non_aborted/min": non_aborted_response_length_min,
        "response_length_non_aborted/clip_ratio": non_aborted_response_length_clip_ratio,
        # aborted ratio
        # Fraction of samples whose response length is zero
        "response/aborted_ratio": aborted_ratio,
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # multi-turn conversation
    if "__num_turns__" in batch.non_tensor_batch:
        num_turns = batch.non_tensor_batch["__num_turns__"]
        metrics["num_turns/min"] = int(num_turns.min())
        metrics["num_turns/max"] = int(num_turns.max())
        metrics["num_turns/mean"] = float(num_turns.mean())

    if "tool_call_counts" in batch.non_tensor_batch:
        tool_call_counts = batch.non_tensor_batch["tool_call_counts"]
        metrics["tool_call_counts/min"] = int(tool_call_counts.min())
        metrics["tool_call_counts/max"] = int(tool_call_counts.max())
        metrics["tool_call_counts/mean"] = float(tool_call_counts.mean())

    extra_info = batch.non_tensor_batch['extra_info']
    reward_keys = [key for key in batch.non_tensor_batch.keys() if key.endswith('type_reward')]
    for reward_key in reward_keys:
        reward_values = batch.non_tensor_batch[reward_key]
        reward_name = reward_key.replace('type_', '')

        metrics.update({
            f'critic/{reward_name}/mean': float(np.mean(reward_values)),
        })
        unique_values, counts = np.unique(reward_values, return_counts=True)
        # Skip per-value counts when reward values look continuous.
        if len(unique_values) < 3:
            for value, count in zip(unique_values, counts):
                metrics[f'critic/{reward_name}/value_{value:.1f}_count'] = int(count)

    last_resp_idx = (response_length.long() - 1).clamp(min=0)
    batch_indices = torch.arange(advantages.shape[0], device=advantages.device)
    last_token_adv = advantages[batch_indices, last_resp_idx]

    # Count uids whose rewards are all zero; when task_type exists, only solving/replay rows are included.
    uid_list = batch.non_tensor_batch['uid']
    uid2rewards = defaultdict(list)
    for i, (uid, reward_value) in enumerate(zip(uid_list, sequence_reward)):
        if "task_type" in batch.non_tensor_batch:
            if batch.non_tensor_batch['task_type'][i] not in ['problem_solving','experience_replay']:
                continue
        uid2rewards[uid].append(reward_value.item())
    num_uid_all_zero_reward = 0
    for uid, rewards in uid2rewards.items():
        if all(r == 0.0 for r in rewards):
            num_uid_all_zero_reward += 1
    metrics['critic/non_pass_problem_count'] = num_uid_all_zero_reward

    # Metrics by task type.
    if "task_type" in batch.non_tensor_batch:
        task_type_list = batch.non_tensor_batch['task_type']
        # Count the reward values based on task_type
        task_type_counts = {}
        task_type_rewards = {}
        task_type_advantages = {}
        for task_type, reward_value, adv_value in zip(task_type_list, sequence_reward, last_token_adv):
            if task_type not in task_type_counts:
                task_type_counts[task_type] = 0
                task_type_rewards[task_type] = []
                task_type_advantages[task_type] = []
            task_type_counts[task_type] += 1
            task_type_rewards[task_type].append(reward_value.item())
            task_type_advantages[task_type].append(adv_value.item())
        # Calculate the mean for each task_type
        for task_type in task_type_counts.keys():
            metrics[f'critic/{task_type}_count'] = task_type_counts[task_type]
            metrics[f'critic/{task_type}_reward_mean'] = float(np.mean(task_type_rewards[task_type]))
            metrics[f'critic/{task_type}_adv_mean'] = float(np.mean(task_type_advantages[task_type]))


    return metrics


def _outcome_rewards(token_level_scores: torch.Tensor, attention_mask: torch.Tensor, prompt_length: int) -> torch.Tensor:
    """Reward placed on the last valid response token. Empty responses score 0."""
    response_mask = attention_mask[:, prompt_length:]
    lengths = response_mask.sum(dim=-1).long()
    index = (lengths - 1).clamp(min=0)
    gathered = token_level_scores.gather(1, index.unsqueeze(1)).squeeze(1)
    return torch.where(lengths > 0, gathered, torch.zeros_like(gathered))


def compute_cure_rollout_metrics(
    batch: DataProto,
    tokenizer=None,
    force_error_conclusion: bool = False,
    prefix: str = "multiturn/",
    include_multiturn: bool = True,
    include_grpo: bool = True,
) -> dict[str, Any]:
    """PAG-comparable rollout stats, plus GRPO group stats that a critic would not show.

    Keys under ``prefix`` match PAG's validation logger (``val/multiturn/``):
    turn_1_accuracy, final_acc, turn_2_accuracy_selection,
    i_to_c_*, c_to_i_*, and verify_* when the critique verdict is the model's.

    ``final_acc`` replaces each refined solution with the mean accuracy of its
    guided regenerations and leaves every other solution at its turn-1 score.
    ``i_to_c_rate`` / ``c_to_i_rate`` are computed on individual regenerations.
    ``*_rate_gt`` divides those counts by the number of turn-1 solutions, same
    denominator PAG uses. With ``n_refine`` > 1 that ratio can exceed 1.

    Verify confusion uses PAG's convention (positive = "says the solution is
    correct"): TP/TN are agreements, FP is "says correct" on a wrong solution,
    FN is "says wrong" on a correct solution. ``critique_force_error_conclusion``
    writes the label from the answer reward, so those counts are not logged.
    """
    if "task_type" not in batch.non_tensor_batch:
        return {}

    task_types = np.asarray(batch.non_tensor_batch["task_type"])
    prompt_length = batch.batch["prompts"].shape[-1]
    outcome = _outcome_rewards(
        batch.batch["token_level_scores"],
        batch.batch["attention_mask"],
        prompt_length,
    ).detach().float().cpu()

    solve_idx = np.flatnonzero(task_types == "problem_solving")
    critique_idx = np.flatnonzero(task_types == "critique")
    refine_idx = np.flatnonzero(task_types == "refine")
    metrics: dict[str, Any] = {}

    if solve_idx.size > 0:
        solve_scores = outcome[solve_idx]
        if include_multiturn:
            metrics[f"{prefix}turn_1_accuracy"] = float((solve_scores > 0.5).float().mean())
        if include_grpo:
            metrics.update(_grpo_solve_group_metrics(solve_scores, np.asarray(batch.non_tensor_batch["uid"])[solve_idx]))

    if not include_multiturn:
        return metrics

    if critique_idx.size == 0 or refine_idx.size == 0:
        if f"{prefix}turn_1_accuracy" in metrics:
            metrics[f"{prefix}final_acc"] = metrics[f"{prefix}turn_1_accuracy"]
        return metrics

    crit_ids = np.asarray(batch.non_tensor_batch["critique_id"])
    reward_models = batch.non_tensor_batch["reward_model"]
    refine_by_critique: dict[Any, list[int]] = defaultdict(list)
    for row in refine_idx.tolist():
        refine_by_critique[crit_ids[row]].append(row)

    final_sum = float(outcome[solve_idx].sum()) if solve_idx.size else 0.0
    n_solves = int(solve_idx.size)
    refine_scores: list[float] = []
    i_to_c = c_to_i = 0
    n_wrong_refined = n_correct_refined = 0

    for row in critique_idx.tolist():
        children = refine_by_critique.get(crit_ids[row], [])
        if not children:
            continue
        original_correct = float(reward_models[row]["ground_truth"]) > 0.5
        child_scores = [float(outcome[child]) for child in children]
        refine_scores.extend(child_scores)
        final_sum += float(np.mean(child_scores)) - float(reward_models[row]["ground_truth"])
        for score in child_scores:
            child_correct = score > 0.5
            if not original_correct:
                n_wrong_refined += 1
                i_to_c += int(child_correct)
            else:
                n_correct_refined += 1
                c_to_i += int(not child_correct)

    if n_solves > 0:
        metrics[f"{prefix}final_acc"] = final_sum / n_solves
    if refine_scores:
        metrics[f"{prefix}turn_2_accuracy_selection"] = float(np.mean(refine_scores))
    if n_wrong_refined > 0:
        metrics[f"{prefix}i_to_c_rate"] = i_to_c / n_wrong_refined
        metrics[f"{prefix}i_to_c_count"] = i_to_c
        if n_solves > 0:
            metrics[f"{prefix}i_to_c_rate_gt"] = i_to_c / n_solves
    if n_correct_refined > 0:
        metrics[f"{prefix}c_to_i_rate"] = c_to_i / n_correct_refined
        metrics[f"{prefix}c_to_i_count"] = c_to_i
        if n_solves > 0:
            metrics[f"{prefix}c_to_i_rate_gt"] = c_to_i / n_solves

    if not force_error_conclusion and tokenizer is not None:
        for key, value in _critique_verify_metrics(batch, tokenizer, critique_idx, prompt_length).items():
            metrics[f"{prefix}{key.removeprefix('multiturn/')}"] = value
    return metrics


def compute_pag_aligned_val_metrics(
    turn1_correct: np.ndarray,
    verify_agree: np.ndarray,
    revised: np.ndarray,
    turn2_correct: np.ndarray,
    prefix: str = "val/multiturn/",
) -> dict[str, float]:
    """Aggregate one generate-verify-regen trajectory per sample, the way PAG val does.

    ``verify_agree`` is 1 when the verifier's label matches the turn-1 answer, which is
    what PAG stores in ``verify_accuracies``. A sample is revised only when the verifier
    does not say the answer is correct. ``final_acc`` keeps turn 1 unless that sample
    was revised, in which case it uses that one regeneration.
    """
    turn1_correct = np.asarray(turn1_correct, dtype=bool)
    verify_agree = np.asarray(verify_agree, dtype=bool)
    revised = np.asarray(revised, dtype=bool)
    turn2_correct = np.asarray(turn2_correct, dtype=bool)
    n = int(turn1_correct.shape[0])
    if n == 0:
        return {}

    final = turn1_correct.astype(np.float64)
    final[revised] = turn2_correct[revised].astype(np.float64)
    answer_correct = turn1_correct
    agrees = verify_agree
    true_positive = int(np.sum(agrees & answer_correct))
    false_positive = int(np.sum(~agrees & ~answer_correct))
    false_negative = int(np.sum(~agrees & answer_correct))
    true_negative = int(np.sum(agrees & ~answer_correct))
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    recall_negative = true_negative / (true_negative + false_positive) if (true_negative + false_positive) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    metrics = {
        f"{prefix}turn_1_accuracy": float(answer_correct.mean()),
        f"{prefix}final_acc": float(final.mean()),
        f"{prefix}verify_1_accuracy": float(agrees.mean()),
        f"{prefix}verify_TP": true_positive,
        f"{prefix}verify_FP": false_positive,
        f"{prefix}verify_FN": false_negative,
        f"{prefix}verify_TN": true_negative,
        f"{prefix}verify_precision": precision,
        f"{prefix}verify_recall": recall,
        f"{prefix}verify_recall_negative": recall_negative,
        f"{prefix}verify_f1": f1,
    }
    if np.any(revised):
        metrics[f"{prefix}turn_2_accuracy_selection"] = float(turn2_correct[revised].mean())
        revised_wrong = revised & ~answer_correct
        revised_right = revised & answer_correct
        i_to_c = int(np.sum(revised_wrong & turn2_correct))
        c_to_i = int(np.sum(revised_right & ~turn2_correct))
        n_wrong = int(revised_wrong.sum())
        n_right = int(revised_right.sum())
        if n_wrong > 0:
            metrics[f"{prefix}i_to_c_rate"] = i_to_c / n_wrong
            metrics[f"{prefix}i_to_c_count"] = i_to_c
            metrics[f"{prefix}i_to_c_rate_gt"] = i_to_c / n
        if n_right > 0:
            metrics[f"{prefix}c_to_i_rate"] = c_to_i / n_right
            metrics[f"{prefix}c_to_i_count"] = c_to_i
            metrics[f"{prefix}c_to_i_rate_gt"] = c_to_i / n
    return metrics


def _grpo_solve_group_metrics(solve_scores: torch.Tensor, uids: np.ndarray) -> dict[str, float]:
    """Within-prompt reward spread. GRPO's baseline is this group mean; a zero std sends advantage to 0."""
    grouped: dict[Any, list[float]] = defaultdict(list)
    for uid, score in zip(uids.tolist(), solve_scores.tolist()):
        grouped[uid].append(float(score))

    zero_pass = partial_pass = full_pass = 0
    stds: list[float] = []
    for scores in grouped.values():
        mean_acc = float(np.mean(scores))
        stds.append(float(np.std(scores)))
        if mean_acc <= 0.0:
            zero_pass += 1
        elif mean_acc >= 1.0:
            full_pass += 1
        else:
            partial_pass += 1

    n_groups = max(len(grouped), 1)
    zero_std = sum(std <= 1e-8 for std in stds)
    return {
        "grpo/solve_group_count": float(len(grouped)),
        "grpo/solve_zero_pass_frac": zero_pass / n_groups,
        "grpo/solve_partial_pass_frac": partial_pass / n_groups,
        "grpo/solve_full_pass_frac": full_pass / n_groups,
        "grpo/solve_group_std_mean": float(np.mean(stds)) if stds else 0.0,
        "grpo/solve_zero_std_frac": zero_std / n_groups,
    }


def _critique_verify_metrics(
    batch: DataProto,
    tokenizer,
    critique_idx: np.ndarray,
    prompt_length: int,
) -> dict[str, float]:
    """Score a free ERROR_FOUND verdict the same way PAG scores GenRM."""
    attention = batch.batch["attention_mask"]
    responses = batch.batch["responses"]
    reward_models = batch.non_tensor_batch["reward_model"]
    true_positive = false_positive = false_negative = true_negative = 0
    judged = 0

    for row in critique_idx.tolist():
        valid_len = int(attention[row, prompt_length:].sum().item())
        if valid_len <= 0:
            continue
        text = tokenizer.decode(responses[row, :valid_len].detach().cpu().tolist(), skip_special_tokens=False)
        lower = text.lower()
        says_wrong = "error_found: true" in lower
        says_correct = "error_found: false" in lower
        if says_wrong == says_correct:
            continue
        answer_correct = float(reward_models[row]["ground_truth"]) > 0.5
        # Agreement bit, matching PAG's genrm_score in the confusion matrix.
        agrees = says_correct == answer_correct
        judged += 1
        if agrees and answer_correct:
            true_positive += 1
        elif (not agrees) and (not answer_correct):
            false_positive += 1
        elif (not agrees) and answer_correct:
            false_negative += 1
        else:
            true_negative += 1

    if judged == 0:
        return {}

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    recall_negative = true_negative / (true_negative + false_positive) if (true_negative + false_positive) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "multiturn/verify_1_accuracy": (true_positive + true_negative) / judged,
        "multiturn/verify_TP": true_positive,
        "multiturn/verify_FP": false_positive,
        "multiturn/verify_FN": false_negative,
        "multiturn/verify_TN": true_negative,
        "multiturn/verify_precision": precision,
        "multiturn/verify_recall": recall,
        "multiturn/verify_recall_negative": recall_negative,
        "multiturn/verify_f1": f1,
    }


def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    """
    Computes timing metrics for different processing stages in PPO training.

    This function calculates both raw timing metrics (in seconds) and per-token timing metrics
    (in milliseconds) for various processing stages like generation, reference computation,
    value computation, advantage computation, and model updates.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage

    Note:
        Different stages use different token counts for normalization:
        - "gen" uses only response tokens
        - Other stages ("ref", "values", "adv", "update_critic", "update_actor") use all tokens
          (prompt + response)
    """
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: dict[str, float], n_gpus: int) -> dict[str, Any]:
    """
    Computes throughput metrics for PPO training.

    This function calculates performance metrics related to token processing speed,
    including the total number of tokens processed, time per step, and throughput
    (tokens per second per GPU).

    Args:
        batch: A DataProto object containing batch data with meta information about token counts.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.

    Returns:
        A dictionary containing:
            - perf/total_num_tokens: Total number of tokens processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Tokens processed per second per GPU

    Note:
        The throughput is calculated as total_tokens / (time * n_gpus) to normalize
        across different GPU counts.
    """
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Performs bootstrap resampling to estimate statistics of metrics.

    This function uses bootstrap resampling to estimate the mean and standard deviation
    of metrics computed by the provided reduction functions on random subsets of the data.

    Args:
        data: List of data points to bootstrap from.
        subset_size: Size of each bootstrap sample.
        reduce_fns: List of functions that compute a metric from a subset of data.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A list of tuples, where each tuple contains (mean, std) for a metric
        corresponding to each reduction function in reduce_fns.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> reduce_fns = [np.mean, np.max]
        >>> bootstrap_metric(data, 3, reduce_fns)
        [(3.0, 0.5), (4.5, 0.3)]  # Example values
    """
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(float(np.mean(lst)), float(np.std(lst))) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate a value based on majority voting.

    This function identifies the most common value for a specified vote key
    in the data, then returns the corresponding value for that majority vote.

    Args:
        data: List of dictionaries, where each dictionary contains both vote_key and val_key.
        vote_key: The key in each dictionary used for voting/counting.
        val_key: The key in each dictionary whose value will be returned for the majority vote.

    Returns:
        The value associated with the most common vote.

    Example:
        >>> data = [
        ...     {"pred": "A", "val": 0.9},
        ...     {"pred": "B", "val": 0.8},
        ...     {"pred": "A", "val": 0.7}
        ... ]
        >>> calc_maj_val(data, vote_key="pred", val_key="val")
        0.9  # Returns the first "val" for the majority vote "A"
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(
    data_sources: list[str], sample_uids: list[str], infos_dict: dict[str, list[Any]], seed: int = 42
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.

    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.

    Args:
        data_sources: List of data source identifiers for each sample.
        sample_uids: List of sample uids corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }

        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)

    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_uids = ["uid1", "uid1", "uid2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_uids, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # Group metrics by data source, prompt and variable
    data_src2uid2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        uid = sample_uids[sample_idx]
        var2vals = data_src2uid2var2vals[data_source][uid]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2uid2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, uid2var2vals in data_src2uid2var2vals.items():
        for uid, var2vals in uid2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue

                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = float(np.mean(var_vals))

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = float(np.std(var_vals))

                    ns = []
                    n = 2
                    while n < n_resps:
                        ns.append(n)
                        n *= 2
                    ns.append(n_resps)

                    for n in ns:
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(
                            data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed
                        )
                        metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                        metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                        if var2vals.get("pred", None) is not None:
                            vote_data = [
                                {"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"], strict=True)
                            ]
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2uid2var2metric[data_source][uid][var_name] = metric

    # Aggregate metrics across uids
    data_src2var2metric2uid_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, uid2var2metric in data_src2uid2var2metric.items():
        for uid, var2metric in uid2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2uid_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2uid_vals in data_src2var2metric2uid_vals.items():
        for var_name, metric2uid_vals in var2metric2uid_vals.items():
            for metric_name, uid_vals in metric2uid_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = float(np.mean(uid_vals))

    return data_src2var2metric2val
