import torch


def compute_agent_score(
    response_mask: torch.Tensor,
    tool_rewards: torch.Tensor,
    user_rewards: torch.Tensor,
    final_reward: int,
    gamma: float = 0.99,
    mode: str = "sparse",
) -> torch.Tensor:
    """
    Compute agent score (i.e., return for each token in the response).

    Args:
        response_mask [response_length]: The mask of the response.
        tool_rewards [response_length]: The rewards for the tool calls.
        user_rewards [response_length]: The rewards for the user messages.
        final_reward [1]: The final reward for the agent.
        gamma: The discount factor for future rewards.
        mode: The mode of the reward score.
            "step": Step-level discounted score. [0.98, 0.98, 0.98, 0, 0, 0, 0.94, 0.94, 0.94, 0.94]
            "token": Token-level discounted score. [0.98, 0.99, 1.0, 0, 0, 0, 0.84, 0.94, 0.99, 1.0]
            "sparse": Sparse reward score. [0, 0, 0.98, 0, 0, 0, 0, 0, 0, 0.94]
    """

    # Normalize the rewards to [-1, 1]
    reward_tensor = (tool_rewards + user_rewards) * response_mask * 1.0    # [response_length]
    final_reward = final_reward * 1.0
    
    if mode == "step":
        # compute step masks
        step_masks = compute_step_masks(response_mask)  # [num_steps, response_length]
        
        # add final_reward to the last step
        reward_tensor = reward_tensor + final_reward * step_masks[-1]
        
        # compute step-level rewards: average rewards for each step
        step_rewards = (reward_tensor.unsqueeze(0) * step_masks).sum(dim=1) / step_masks.sum(dim=1)  # [num_steps]
        
        # compute step-level score with gamma decay (backwards)
        step_score = torch.zeros_like(step_rewards)
        step_score[-1] = step_rewards[-1]
        for i in range(len(step_rewards) - 2, -1, -1):
            step_score[i] = step_rewards[i] + gamma * step_score[i + 1]
        
        # broadcast step score to all tokens in each step
        score = (step_score.unsqueeze(1) * step_masks).sum(dim=0)  # [response_length]
        
    elif mode == "token":  # token-level
        # add final_reward to the last token (last position where response_mask == 1)
        mask_indices = (response_mask == 1).nonzero(as_tuple=True)[0]
        
        if len(mask_indices) == 0:
            return torch.zeros_like(reward_tensor)
        
        reward_tensor[mask_indices[-1]] += final_reward
        
        # compute token-level score with gamma decay (backwards through masked positions only)
        score = torch.zeros_like(reward_tensor)
        
        # Start from the last masked position
        score[mask_indices[-1]] = reward_tensor[mask_indices[-1]]
        
        # Work backwards through masked positions only
        for i in range(len(mask_indices) - 2, -1, -1):
            curr_idx = mask_indices[i]
            next_idx = mask_indices[i + 1]
            score[curr_idx] = reward_tensor[curr_idx] + gamma * score[next_idx]

    elif mode == "sparse":     # sparse rewards
        # compute step masks
        step_masks = compute_step_masks(response_mask)  # [num_steps, response_length]
        
        # Safety check for empty response to avoid errors in subsequent operations
        if step_masks.size(0) == 0:
            return torch.zeros_like(reward_tensor)

        # compute step-level rewards: average rewards for each step
        # Since input rewards are shared (smeared) across the step, averaging recovers the scalar reward value.
        step_rewards = (reward_tensor.unsqueeze(0) * step_masks).sum(dim=1) / step_masks.sum(dim=1)  # [num_steps]
        
        # add final_reward to the last step's reward
        step_rewards[-1] += final_reward
        
        # Initialize the output tensor with zeros
        score = torch.zeros_like(reward_tensor)
        
        # Identify the index of the last token for each step.
        # Logic: Multiply mask by position indices. Since tokens in a step are contiguous, 
        # the max value will be the index of the last token in that step.
        indices = torch.arange(response_mask.size(0), device=response_mask.device).unsqueeze(0) # [1, response_length]
        # (step_masks * indices) gives indices where mask is 1, and 0 elsewhere.
        # argmax(dim=1) picks the largest index for each step row.
        last_token_indices = (step_masks * indices).argmax(dim=1) # [num_steps]
        
        # Scatter the step_rewards to the computed last token positions
        score.scatter_(0, last_token_indices, step_rewards)

    elif mode == "tgrpo":
        score = reward_tensor
        
    return score


def compute_step_masks(response_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the step masks, where each step is a contiguous sequence of tokens with response_mask == 1.

    Args:
        response_mask [response_length]: The mask of the response [0,0,1,1,0,0,1,1,0,0].

    score:
        step_masks [num_steps, response_length]: The step masks [[0,0,1,1,0,0,0,0,0,0], [0,0,0,0,0,0,1,1,0,0]].
    """
    # Find where sequences start (transition from 0 to 1)
    prev_mask = torch.cat([torch.tensor([0], device=response_mask.device, dtype=response_mask.dtype), response_mask[:-1]])
    starts = (response_mask == 1) & (prev_mask == 0)    # [0,0,1,0,0,0,1,0,0,0]
    
    # Assign step IDs to each position (0 for non-masked, 1,2,3,... for each step)
    step_ids = torch.cumsum(starts.int(), dim=0) * response_mask.int()    # [0,0,1,1,1,1,2,2,2,2]
    
    # Count number of steps
    num_steps = step_ids.max().item() if step_ids.max() > 0 else 0    # 2
    
    if num_steps == 0:
        return torch.zeros((0, len(response_mask)), device=response_mask.device, dtype=response_mask.dtype)
    
    # Create masks using broadcasting: each row corresponds to one step
    step_indices = torch.arange(1, num_steps + 1, device=response_mask.device).unsqueeze(1)    # [1, 2]
    step_masks = (step_ids.unsqueeze(0) == step_indices) & response_mask.unsqueeze(0).bool()    # [[0,0,1,1,0,0,0,0,0,0], [0,0,0,0,0,0,1,1,0,0]]
    
    return step_masks.to(response_mask.dtype)