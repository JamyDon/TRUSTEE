from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.workers.config.rollout import MultiTurnConfig as BaseMultiTurnConfig


@dataclass
class MultiTurnConfig(BaseMultiTurnConfig):
    _mutable_fields = {"max_assistant_turns", "max_user_turns"}

    enable: bool = False
    max_assistant_turns: Optional[int] = None
    tool_config_path: Optional[str] = None
    max_user_turns: Optional[int] = None
    max_tool_turns: Optional[int] = None
    max_parallel_calls: int = 1
    max_tool_response_length: int = 256
    tool_response_truncate_side: str = "middle"
    interaction_config_path: Optional[str] = None
    use_inference_chat_template: bool = False
    tokenization_sanity_check_mode: str = "strict"
    format: str = "hermes"
    num_repeat_rollouts: Optional[int] = None
    enable_tool_reward: bool = True
    enable_user_reward: bool = True
    require_thinking: bool = True  # set False for models that don't use <think>...</think> (e.g. Llama-3.1)