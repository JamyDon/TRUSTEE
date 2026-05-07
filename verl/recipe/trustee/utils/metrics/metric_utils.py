from typing import Any

import numpy as np
import torch

from verl.protocol import DataProto
from verl.trainer.ppo.metric_utils import _compute_response_info


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
        if valid_returns.numel() > 1:
            return_diff_var = torch.var(valid_returns - valid_values)
            return_var = torch.var(valid_returns)
        else:
            return_diff_var = torch.tensor(0.0)
            return_var = torch.tensor(1e-5)

    user_turns = batch.non_tensor_batch["user_turns"]
    tool_turns = batch.non_tensor_batch["tool_turns"]
    final_reward = batch.non_tensor_batch["final_reward"]
    agent_aborted = batch.non_tensor_batch["aborted"]
    abort_cause_tool = batch.non_tensor_batch.get("abort_cause_tool", np.array([]))
    abort_cause_user = batch.non_tensor_batch.get("abort_cause_user", np.array([]))
    abort_cause_verifier = batch.non_tensor_batch.get("abort_cause_verifier", np.array([]))
    abort_cause_tool_timeout = batch.non_tensor_batch.get("abort_cause_tool_timeout", np.array([]))
    abort_cause_tool_parse = batch.non_tensor_batch.get("abort_cause_tool_parse", np.array([]))
    abort_cause_tool_validation = batch.non_tensor_batch.get("abort_cause_tool_validation", np.array([]))
    abort_cause_tool_exception = batch.non_tensor_batch.get("abort_cause_tool_exception", np.array([]))
    abort_cause_user_timeout = batch.non_tensor_batch.get("abort_cause_user_timeout", np.array([]))
    abort_cause_user_parse = batch.non_tensor_batch.get("abort_cause_user_parse", np.array([]))
    abort_cause_user_validation = batch.non_tensor_batch.get("abort_cause_user_validation", np.array([]))
    abort_cause_verifier_timeout = batch.non_tensor_batch.get("abort_cause_verifier_timeout", np.array([]))
    abort_cause_verifier_parse = batch.non_tensor_batch.get("abort_cause_verifier_parse", np.array([]))
    abort_cause_verifier_validation = batch.non_tensor_batch.get("abort_cause_verifier_validation", np.array([]))
    stop_response_length = batch.non_tensor_batch["stop_response_length"]
    stop_assistant_turns = batch.non_tensor_batch["stop_assistant_turns"]
    stop_user_turns = batch.non_tensor_batch["stop_user_turns"]
    stop_tool_turns = batch.non_tensor_batch["stop_tool_turns"]
    stop_aborted = batch.non_tensor_batch["stop_aborted"]
    stop_user_terminated = batch.non_tensor_batch["stop_user_terminated"]
    format_ok = batch.non_tensor_batch.get("format_ok", np.array([]))
    assistant_response_turns = batch.non_tensor_batch.get("assistant_response_turns", np.array([]))
    assistant_tool_call_turns = batch.non_tensor_batch.get("assistant_tool_call_turns", np.array([]))
    if "user_rewards_sum" in batch.non_tensor_batch:
        user_rewards_sum = np.sum(batch.non_tensor_batch["user_rewards_sum"])
    if "tool_rewards_sum" in batch.non_tensor_batch:
        tool_rewards_sum = np.sum(batch.non_tensor_batch["tool_rewards_sum"])

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
        non_aborted_response_length_mean = 0.0
        non_aborted_response_length_max = 0.0
        non_aborted_response_length_min = 0.0
        non_aborted_response_length_clip_ratio = 0.0

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
        "critic/advantages/mean": torch.mean(valid_adv).detach().item() if valid_adv.numel() > 0 else 0.0,
        "critic/advantages/max": torch.max(valid_adv).detach().item() if valid_adv.numel() > 0 else 0.0,
        "critic/advantages/min": torch.min(valid_adv).detach().item() if valid_adv.numel() > 0 else 0.0,
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item() if valid_returns.numel() > 0 else 0.0,
        "critic/returns/max": torch.max(valid_returns).detach().item() if valid_returns.numel() > 0 else 0.0,
        "critic/returns/min": torch.min(valid_returns).detach().item() if valid_returns.numel() > 0 else 0.0,
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item() if valid_values.numel() > 0 else 0.0,
                "critic/values/max": torch.max(valid_values).detach().item() if valid_values.numel() > 0 else 0.0,
                "critic/values/min": torch.min(valid_values).detach().item() if valid_values.numel() > 0 else 0.0,
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
        metrics["num_turns/min"] = num_turns.min()
        metrics["num_turns/max"] = num_turns.max()
        metrics["num_turns/mean"] = num_turns.mean()

    if "tool_call_counts" in batch.non_tensor_batch:
        tool_call_counts = batch.non_tensor_batch["tool_call_counts"]
        metrics["tool_call_counts/min"] = tool_call_counts.min()
        metrics["tool_call_counts/max"] = tool_call_counts.max()
        metrics["tool_call_counts/mean"] = tool_call_counts.mean()

    if len(user_turns) > 0:
        metrics["agent/user_turns/min"] = user_turns.min()
        metrics["agent/user_turns/max"] = user_turns.max()
        metrics["agent-core/user_turns/mean"] = user_turns.mean()
    if len(tool_turns) > 0:
        metrics["agent/tool_turns/min"] = tool_turns.min()
        metrics["agent/tool_turns/max"] = tool_turns.max()
        metrics["agent-core/tool_turns/mean"] = tool_turns.mean()
    if len(final_reward) > 0:
        metrics["agent/final_reward/min"] = final_reward.min()
        metrics["agent/final_reward/max"] = final_reward.max()
        metrics["agent-core/final_reward/mean"] = final_reward.mean()
        metrics["agent/final_reward/std"] = final_reward.std()
        metrics["agent/final_reward/median"] = np.median(final_reward)
    if len(agent_aborted) > 0:
        metrics["agent-core/aborted/mean"] = agent_aborted.mean()
    if len(abort_cause_tool) > 0:
        metrics["agent/abort_cause/tool"] = abort_cause_tool.mean()
    if len(abort_cause_user) > 0:
        metrics["agent/abort_cause/user"] = abort_cause_user.mean()
    if len(abort_cause_verifier) > 0:
        metrics["agent/abort_cause/verifier"] = abort_cause_verifier.mean()
    if len(abort_cause_tool_timeout) > 0:
        metrics["agent/abort_cause/tool_timeout"] = abort_cause_tool_timeout.mean()
    if len(abort_cause_tool_parse) > 0:
        metrics["agent/abort_cause/tool_parse"] = abort_cause_tool_parse.mean()
    if len(abort_cause_tool_validation) > 0:
        metrics["agent/abort_cause/tool_validation"] = abort_cause_tool_validation.mean()
    if len(abort_cause_tool_exception) > 0:
        metrics["agent/abort_cause/tool_exception"] = abort_cause_tool_exception.mean()
    if len(abort_cause_user_timeout) > 0:
        metrics["agent/abort_cause/user_timeout"] = abort_cause_user_timeout.mean()
    if len(abort_cause_user_parse) > 0:
        metrics["agent/abort_cause/user_parse"] = abort_cause_user_parse.mean()
    if len(abort_cause_user_validation) > 0:
        metrics["agent/abort_cause/user_validation"] = abort_cause_user_validation.mean()
    if len(abort_cause_verifier_timeout) > 0:
        metrics["agent/abort_cause/verifier_timeout"] = abort_cause_verifier_timeout.mean()
    if len(abort_cause_verifier_parse) > 0:
        metrics["agent/abort_cause/verifier_parse"] = abort_cause_verifier_parse.mean()
    if len(abort_cause_verifier_validation) > 0:
        metrics["agent/abort_cause/verifier_validation"] = abort_cause_verifier_validation.mean()
    if len(stop_response_length) > 0:
        metrics["agent/stop_reason/response_length"] = stop_response_length.mean()
    if len(stop_assistant_turns) > 0:
        metrics["agent/stop_reason/assistant_turns"] = stop_assistant_turns.mean()
    if len(stop_user_turns) > 0:
        metrics["agent/stop_reason/user_turns"] = stop_user_turns.mean()
    if len(stop_tool_turns) > 0:
        metrics["agent/stop_reason/tool_turns"] = stop_tool_turns.mean()
    if len(stop_aborted) > 0:
        metrics["agent/stop_reason/aborted"] = stop_aborted.mean()
    if len(stop_user_terminated) > 0:
        metrics["agent/stop_reason/user_terminated"] = stop_user_terminated.mean()
    if len(format_ok) > 0:
        metrics["agent-core/format_ok/mean"] = format_ok.mean()
    if len(assistant_response_turns) > 0:
        metrics["agent-core/assistant_response_turns/mean"] = assistant_response_turns.mean()
    if len(assistant_tool_call_turns) > 0:
        metrics["agent-core/assistant_tool_call_turns/mean"] = assistant_tool_call_turns.mean()
    if user_turns.sum() > 0:
        metrics["agent-core/user_rewards/mean"] = user_rewards_sum / user_turns.sum()
    if tool_turns.sum() > 0:
        metrics["agent-core/tool_rewards/mean"] = tool_rewards_sum / tool_turns.sum()

    return metrics