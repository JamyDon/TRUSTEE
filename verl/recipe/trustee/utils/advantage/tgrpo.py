from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from verl.trainer.config import AlgoConfig


def compute_tgrpo_advantage(
    outcome_rewards: torch.Tensor,
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for TGRPO, operating only on Token-level reward
    (with only one scalar reward for each response).

    Args:
        outcome_rewards: `(torch.Tensor)`
            shape is (bs)
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    mode = config.get("adv_kwargs", {}).get("mode", "relu")
    scores = outcome_rewards

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if response_mask[i].sum() == 0:
                continue
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if response_mask[i].sum() == 0:
                scores[i] = 0.0
            elif norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        # scores = scores.unsqueeze(-1) * response_mask

        if mode == "sign":
            relu_mask = (token_level_rewards * scores.unsqueeze(-1)) > 0
            scores = scores.unsqueeze(-1) * response_mask * relu_mask
        elif mode == "sign-linear":
            positive_mask = (scores.unsqueeze(-1) > 0).float()
            negative_mask = (scores.unsqueeze(-1) < 0).float()
            positive_token_level_contribution = (token_level_rewards + 1) * 0.5
            negative_token_level_contribution = (1 - token_level_rewards) * 0.5
            scores = scores.unsqueeze(-1) * response_mask * positive_mask * positive_token_level_contribution + scores.unsqueeze(-1) * response_mask * negative_mask * negative_token_level_contribution
        elif mode == "relu":
            relu_mask = (token_level_rewards * scores.unsqueeze(-1)) > 0
            scores = scores.unsqueeze(-1) * response_mask * relu_mask * token_level_rewards
        elif mode == "linear":
            scores = scores.unsqueeze(-1) * response_mask * token_level_rewards
        elif mode == "uniform":
            scores = scores.unsqueeze(-1) * response_mask
        else:
            raise NotImplementedError(f"mode {mode} not implemented")
        
    return scores, scores