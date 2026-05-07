# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import asyncio
import json
import logging
import os
import re
from enum import Enum
from typing import Any
from uuid import uuid4

import numpy as np
import ray
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from recipe.trustee.simulator import Simulator
from verl.experimental.agent_loop.agent_loop import (AgentLoopBase,
                                                     AgentLoopManager,
                                                     AgentLoopOutput,
                                                     AgentLoopWorkerBase,
                                                     AsyncLLMServerManager,
                                                     DictConfigWrap,
                                                     RayResourcePool,
                                                     RayWorkerGroup,
                                                     _InternalAgentLoopOutput,
                                                     get_trajectory_info,
                                                     register)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.protocol import DataProto
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_op
from verl.utils.transferqueue_utils import tqbridge

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    WAITING_FOR_TOOL_RESPONSE = "waiting_for_tool_response"
    WAITING_FOR_USER_RESPONSE = "waiting_for_user_response"
    TERMINATED = "terminated"
    ABORTED = "aborted"


class SimulatorAgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    tool may need to access full history state. User can store any tool session data in `extra_fields`."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        metrics: dict[str, Any],
        request_id: str,
    ):
        self.messages = messages
        self.tool_schemas = tool_schemas
        self.metrics = metrics
        self.request_id = request_id

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_rewards: list[float] = []
        self.tool_rewards_for_logging: list[float] = []
        self.user_rewards_for_logging: list[float] = []
        self.tool_turns = 0
        self.user_turns = 0
        self.assistant_turns = 0
        self.stop_reason = "not_terminated"

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []

        # Per-rollout turn limits (set from kwargs in run(); None means use class-level defaults)
        self.max_user_turns: int | None = None
        self.max_assistant_turns: int | None = None
        self.max_tool_turns: int | None = None

        # Format tracking: False if any turn breaks <think>/<think> or tool call format
        self.format_ok = True
        # Assistant turn type counters
        self.assistant_response_turns = 0
        self.assistant_tool_call_turns = 0

        # Abort cause: "none" | "tool_timeout" | "tool_parse" | "tool_validation" | "tool_exception"
        #              | "user_timeout" | "user_parse" | "user_validation"
        #              | "verifier_timeout" | "verifier_parse" | "verifier_validation"
        self.abort_cause = "none"

        # Extra fields for dynamic addition, e.g., tool session data
        self.extra_fields: dict[str, Any] = {}


@register("simulator_agent")
class SimulatorAgentLoop(AgentLoopBase):
    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        config = trainer_config.config

        # Initialize tools from config file
        self.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        self.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        self.max_tool_turns = config.actor_rollout_ref.rollout.multi_turn.max_tool_turns
        self.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        self.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        self.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        self.tool_parser = ToolParser.get_tool_parser(
            config.actor_rollout_ref.rollout.multi_turn.format, self.tokenizer
        )
        self.tool_parser_name = config.actor_rollout_ref.rollout.multi_turn.format
        self.enable_tool_reward = config.actor_rollout_ref.rollout.multi_turn.enable_tool_reward
        self.enable_user_reward = config.actor_rollout_ref.rollout.multi_turn.enable_user_reward
        self.require_thinking = config.actor_rollout_ref.rollout.multi_turn.require_thinking
        self.strip_thinking = config.simulator.strip_thinking

        # Ablation flags
        self.ablate_user_intent = getattr(config.simulator, 'ablate_user_intent', False)
        self.ablate_user_persona = getattr(config.simulator, 'ablate_user_persona', False)
        self.ablate_expected_tool_calls = getattr(config.simulator, 'ablate_expected_tool_calls', False)

        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        self.response_length = config.actor_rollout_ref.rollout.response_length

        self.simulator = Simulator(config.simulator)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        tool_schemas = list(kwargs["tools"])

        metrics = {}
        request_id = uuid4().hex

        # Create EnvAgentData instance to encapsulate all state
        agent_data = SimulatorAgentData(
            messages=messages,
            tool_schemas=tool_schemas,
            metrics=metrics,
            request_id=request_id,
        )
        agent_data.user_query = messages[0]["content"]
        # Per-rollout turn limits must be provided by the curriculum during training
        is_validate = bool(kwargs.get("validate", False))
        for key in ("max_user_turns", "max_assistant_turns", "max_tool_turns"):
            if kwargs.get(key) is None and not is_validate:
                raise ValueError(f"Curriculum must provide '{key}' for each sample, but it is missing.")
        agent_data.max_user_turns = int(kwargs["max_user_turns"]) if kwargs.get("max_user_turns") is not None else self.max_user_turns
        agent_data.max_assistant_turns = int(kwargs["max_assistant_turns"]) if kwargs.get("max_assistant_turns") is not None else self.max_assistant_turns
        agent_data.max_tool_turns = int(kwargs["max_tool_turns"]) if kwargs.get("max_tool_turns") is not None else self.max_tool_turns
        # Task fields (from curriculum task generator; optional for non-curriculum data)
        agent_data.expected_tool_calls = kwargs.get("expected_tool_calls") or []
        agent_data.user_intent = kwargs.get("user_intent") or ""
        agent_data.user_persona = kwargs.get("user_persona") or ""
        agent_data.num_criteria = int(kwargs.get("num_criteria", 5))

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED and state != AgentState.ABORTED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.WAITING_FOR_TOOL_RESPONSE:
                state = await self._handle_waiting_for_tool_response_state(agent_data)
            elif state == AgentState.WAITING_FOR_USER_RESPONSE:
                state = await self._handle_waiting_for_user_response_state(agent_data)
            else:
                logger.error(f"Invalid state: {state.value}")
                state = AgentState.TERMINATED

        # Finalize output
        final_reward, final_status = await self.simulator.reward(
            agent_data.messages,
            agent_data.user_query,
            agent_data.tool_schemas,
            expected_tool_calls=[] if self.ablate_expected_tool_calls else (getattr(agent_data, "expected_tool_calls", None) or []),
            user_intent="" if self.ablate_user_intent else (getattr(agent_data, "user_intent", "") or ""),
            user_persona="" if self.ablate_user_persona else (getattr(agent_data, "user_persona", "") or ""),
            num_criteria=getattr(agent_data, "num_criteria", 5),
        )
        if final_status.startswith("aborted"):
            state = AgentState.ABORTED
            subcause = final_status[len("aborted_"):] if "_" in final_status else "unknown"
            agent_data.abort_cause = f"verifier_{subcause}"

        # Format penalty
        # if not agent_data.format_ok:
        #     final_reward = -1.0
        # if agent_data.format_ok and agent_data.assistant_tool_call_turns > 0:
        #     final_reward = max(-1.0, final_reward)
        # elif agent_data.format_ok or agent_data.assistant_tool_call_turns > 0:
        #     # final_reward = max(-0.75, final_reward)
        #     final_reward = -1.5
        # else:
        #     final_reward = -2.0

        # No-tool penalty: agent must make at least one tool call turn
        if agent_data.assistant_tool_call_turns == 0:
            final_reward = -1.0

        # If the agent is aborted, set the response mask to 0
        if state == AgentState.ABORTED:
            agent_data.response_mask = [0] * len(agent_data.response_mask)

        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + agent_data.tool_turns + 1,
            metrics=agent_data.metrics,
            extra_fields={},
        )
        output.extra_fields.update(
            {
                "tool_rewards": agent_data.tool_rewards[: self.response_length],
                "user_rewards": agent_data.user_rewards[: self.response_length],
                "tool_rewards_sum": sum(agent_data.tool_rewards_for_logging),
                "user_rewards_sum": sum(agent_data.user_rewards_for_logging),
                "user_turns": agent_data.user_turns,
                "assistant_turns": agent_data.assistant_turns,
                "tool_turns": agent_data.tool_turns,
                "final_reward": final_reward,
                "aborted": 1 if state == AgentState.ABORTED else 0,
                "abort_cause_tool": 1 if agent_data.abort_cause.startswith("tool") else 0,
                "abort_cause_user": 1 if agent_data.abort_cause.startswith("user") else 0,
                "abort_cause_verifier": 1 if agent_data.abort_cause.startswith("verifier") else 0,
                "abort_cause_tool_timeout": 1 if agent_data.abort_cause == "tool_timeout" else 0,
                "abort_cause_tool_parse": 1 if agent_data.abort_cause == "tool_parse" else 0,
                "abort_cause_tool_validation": 1 if agent_data.abort_cause == "tool_validation" else 0,
                "abort_cause_tool_exception": 1 if agent_data.abort_cause == "tool_exception" else 0,
                "abort_cause_user_timeout": 1 if agent_data.abort_cause == "user_timeout" else 0,
                "abort_cause_user_parse": 1 if agent_data.abort_cause == "user_parse" else 0,
                "abort_cause_user_validation": 1 if agent_data.abort_cause == "user_validation" else 0,
                "abort_cause_verifier_timeout": 1 if agent_data.abort_cause == "verifier_timeout" else 0,
                "abort_cause_verifier_parse": 1 if agent_data.abort_cause == "verifier_parse" else 0,
                "abort_cause_verifier_validation": 1 if agent_data.abort_cause == "verifier_validation" else 0,
                "stop_response_length": 1 if agent_data.stop_reason == "response length" else 0,
                "stop_assistant_turns": 1 if agent_data.stop_reason == "assistant turns" else 0,
                "stop_user_turns": 1 if agent_data.stop_reason == "user turns" else 0,
                "stop_tool_turns": 1 if agent_data.stop_reason == "tool turns" else 0,
                "stop_aborted": 1 if agent_data.stop_reason == "aborted" else 0,
                "stop_user_terminated": 1 if agent_data.stop_reason == "user terminated" else 0,
                "format_ok": 1 if agent_data.format_ok else 0,
                "assistant_response_turns": agent_data.assistant_response_turns,
                "assistant_tool_call_turns": agent_data.assistant_tool_call_turns,
            }
        )
        # Add simulator token stats for this rollout
        token_stats = self.simulator.get_token_stats()
        output.extra_fields["simulator_input_tokens"] = token_stats["input_tokens_step"]
        output.extra_fields["simulator_output_tokens"] = token_stats["output_tokens_step"]
        assert "tool_rewards" in output.extra_fields, "tool_rewards should be in extra_fields"
        assert "user_rewards" in output.extra_fields, "user_rewards should be in extra_fields"
        assert len(agent_data.tool_rewards) == len(agent_data.response_mask), f"({agent_data.request_id}) tool_rewards length {len(agent_data.tool_rewards)} should be equal to response_mask length {len(agent_data.response_mask)}"
        assert len(agent_data.user_rewards) == len(agent_data.response_mask), f"({agent_data.request_id}) user_rewards length {len(agent_data.user_rewards)} should be equal to response_mask length {len(agent_data.response_mask)}"
        assert agent_data.stop_reason in ["response length", "assistant turns", "user turns", "tool turns", "aborted", "user terminated"], f"({agent_data.request_id}) stop_reason should be one of ['response length', 'assistant turns', 'user turns', 'tool turns', 'aborted', 'user terminated'], got {agent_data.stop_reason}"
        
        return output

    async def _handle_pending_state(self, agent_data: SimulatorAgentData) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=agent_data.tool_schemas,
        )
        agent_data.prompt_ids = prompt_ids
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: SimulatorAgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
            )

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # Add assistant message to messages
        assistant_message = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
        )
        
        # Format check: every assistant turn must contain <think> ... </think>
        # Disabled for models that don't use chain-of-thought tags (e.g. Llama-3.1).
        if self.require_thinking and ("<think>" not in assistant_message or "</think>" not in assistant_message):
            agent_data.format_ok = False

        # Extract tool calls
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)
        should_call = self.tool_parser.tool_call_start_token in assistant_message \
            or self.tool_parser.tool_call_end_token in assistant_message
        
        # Update assistant turn type counters
        # NOTE: for Llama-3.1, tool call tokens (<|python_tag|>, <|eom_id|>) are special tokens
        # stripped by skip_special_tokens=True in the decode above, so should_call is False even
        # when tool calls were made. Fall back to checking agent_data.tool_calls (which uses
        # skip_special_tokens=False internally) to ensure assistant_tool_call_turns is counted.
        if should_call or bool(agent_data.tool_calls):
            agent_data.assistant_tool_call_turns += 1
        else:
            agent_data.assistant_response_turns += 1

        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            agent_data.tool_rewards += [0.0] * len(agent_data.response_ids)
            agent_data.user_rewards += [0.0] * len(agent_data.response_ids)
            agent_data.stop_reason = "response length"
            return AgentState.TERMINATED
        if agent_data.max_assistant_turns and agent_data.assistant_turns > agent_data.max_assistant_turns:
            agent_data.tool_rewards += [0.0] * len(agent_data.response_ids)
            agent_data.user_rewards += [0.0] * len(agent_data.response_ids)
            agent_data.stop_reason = "assistant turns"
            return AgentState.TERMINATED
        if agent_data.max_user_turns and agent_data.user_turns > agent_data.max_user_turns:
            agent_data.tool_rewards += [0.0] * len(agent_data.response_ids)
            agent_data.user_rewards += [0.0] * len(agent_data.response_ids)
            agent_data.stop_reason = "user turns"
            return AgentState.TERMINATED
        if agent_data.max_tool_turns and agent_data.tool_turns > agent_data.max_tool_turns:
            agent_data.tool_rewards += [0.0] * len(agent_data.response_ids)
            agent_data.user_rewards += [0.0] * len(agent_data.response_ids)
            agent_data.stop_reason = "tool turns"
            return AgentState.TERMINATED

        # Strip <think>...</think> blocks so user/tool/rewarder are not noised by reasoning
        if self.strip_thinking:
            content_for_messages = re.sub(r"<think>.*?</think>", "", assistant_message, flags=re.DOTALL).strip()
        else:
            content_for_messages = assistant_message
        
        add_messages.append({"role": "assistant", "content": content_for_messages})
        agent_data.messages.extend(add_messages)

        # Determine next state
        if agent_data.tool_calls or should_call:
            agent_data.tool_rewards += ["TO_BE_REPLACED"] * len(agent_data.response_ids)
            agent_data.user_rewards += [0.0] * len(agent_data.response_ids)
            return AgentState.WAITING_FOR_TOOL_RESPONSE
        else:
            agent_data.tool_rewards += [0.0] * len(agent_data.response_ids)
            agent_data.user_rewards += ["TO_BE_REPLACED"] * len(agent_data.response_ids)
            return AgentState.WAITING_FOR_USER_RESPONSE

    async def _handle_waiting_for_tool_response_state(self, agent_data: SimulatorAgentData) -> AgentState:
        """Handle the waiting for tool response state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        agent_data.tool_turns += 1

        # Format check: tool call tokens were present but no valid tool calls were parsed → malformed
        if not agent_data.tool_calls[: self.max_parallel_calls]:
            agent_data.format_ok = False

        tasks = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data))

        with simple_timer("tool_calls", agent_data.metrics):
            responses: list[tuple[str, float, str]] = await asyncio.gather(*tasks)

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        tool_rewards = []
        status = "normal"
        for tool_response_text, tool_reward, tool_status in responses:
            # Create message from tool response
            message = {"role": "tool", "content": tool_response_text or "Error: Tool call could not be resolved"}

            add_messages.append(message)

            if self.enable_tool_reward:
                tool_rewards.append(tool_reward)
            
            if tool_status.startswith("aborted"):
                status = "aborted"
                subcause = tool_status[len("aborted_"):] if "_" in tool_status else "unknown"
                agent_data.abort_cause = f"tool_{subcause}"
                
        if self.enable_tool_reward:
            if not tool_rewards:
                tool_reward = -1.0
                assert add_messages == [], "add_messages should be empty if tool rewards are enabled and no tool rewards are returned"
                add_messages.append({"role": "tool", "content": "Error: Tool call is malformed or could not be parsed"})
            else:
                tool_reward = sum(tool_rewards) / len(tool_rewards)
        else:
            tool_reward = None
            if len(agent_data.tool_calls[: self.max_parallel_calls]) == 0:
                add_messages.append({"role": "tool", "content": "Error: Tool call is malformed or could not be parsed"})

        agent_data.messages.extend(add_messages)

        # print("="*30 + "add_messages" + "="*30 + f"\n{add_messages}")

        try:
            if not add_messages:
                # This indicates that the agent fails to call tools with the correct format.
                print("[WARNING] Tool response is empty, using default error message")
                add_messages = [{"role": "tool", "content": "Error: Tool call is malformed or could not be parsed"}]
            response_ids = await self.apply_chat_template(
                add_messages,
                remove_system_prompt=True,
            )
        except Exception as e:
            print(f"[Exception] Error when applying chat template: {e}")
            print(f"add_messages: {add_messages}")
            response_ids = []
            status = "aborted"
            agent_data.abort_cause = "tool_exception"

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length \
            or status == "aborted":
            if self.enable_tool_reward:
                for i in range(len(agent_data.tool_rewards)):
                    if agent_data.tool_rewards[i] == "TO_BE_REPLACED":
                        agent_data.tool_rewards[i] = tool_reward
            agent_data.stop_reason = "aborted" if status == "aborted" else "response length"
            return AgentState.ABORTED if status == "aborted" else AgentState.TERMINATED
            
        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        if self.enable_tool_reward:
            for i in range(len(agent_data.tool_rewards)):
                if agent_data.tool_rewards[i] == "TO_BE_REPLACED":
                    agent_data.tool_rewards[i] = tool_reward
            agent_data.tool_rewards_for_logging.append(tool_reward)
            agent_data.tool_rewards += [0.0] * len(response_ids)
        if self.enable_user_reward:
            agent_data.user_rewards += [0.0] * len(response_ids)

        return AgentState.GENERATING

    async def _handle_waiting_for_user_response_state(self, agent_data: SimulatorAgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""

        (
            should_terminate_sequence,
            user_responses,
            user_reward,
            user_status,
        ) = await self.simulator.call_user(
            agent_data.messages,
            agent_data.messages[-1]["content"],
            agent_data.user_query,
            agent_data.tool_schemas,
            user_intent="" if self.ablate_user_intent else (getattr(agent_data, "user_intent", "") or ""),
            user_persona="" if self.ablate_user_persona else (getattr(agent_data, "user_persona", "") or ""),
        )
        agent_data.user_turns += 1

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": user_responses}]
        agent_data.messages.extend(add_messages)

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        response_ids = await self.apply_chat_template(
            add_messages,
            remove_system_prompt=True,
        )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length \
            or user_status.startswith("aborted"):
            if self.enable_user_reward:
                for i in range(len(agent_data.user_rewards)):
                    if agent_data.user_rewards[i] == "TO_BE_REPLACED":
                        agent_data.user_rewards[i] = user_reward
            if user_status.startswith("aborted"):
                subcause = user_status[len("aborted_"):] if "_" in user_status else "unknown"
                agent_data.abort_cause = f"user_{subcause}"
            agent_data.stop_reason = "aborted" if user_status.startswith("aborted") else "response length"
            return AgentState.ABORTED if user_status.startswith("aborted") else AgentState.TERMINATED

        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)

        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        if self.enable_user_reward:
            for i in range(len(agent_data.user_rewards)):
                if agent_data.user_rewards[i] == "TO_BE_REPLACED":
                    agent_data.user_rewards[i] = user_reward
            agent_data.user_rewards_for_logging.append(user_reward)
            agent_data.user_rewards += [0.0] * len(response_ids)
        if self.enable_tool_reward:
            agent_data.tool_rewards += [0.0] * len(response_ids)

        # Check termination condition
        if should_terminate_sequence:
            agent_data.stop_reason = "user terminated"
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, agent_data: SimulatorAgentData
    ) -> tuple[str, float]:
        """Call tool and return tool response."""
        tool_name = tool_call.name

        def _check_tool_in_schema(tool_name: str, tool_schemas: list[dict[str, Any]]) -> bool:
            """Check if tool is in schema."""
            return any(tool.get("function", {}).get("name", None) == tool_name for tool in tool_schemas)

        if not _check_tool_in_schema(tool_name, agent_data.tool_schemas):
            tool_response_text = f"Error: Tool {tool_name} not found"
            tool_reward = -1.0
            tool_status = "normal"
        else:
            tool_args = json.loads(tool_call.arguments)
            tool_call_text = json.dumps({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": tool_args
                }
            }, indent=2)
            interaction_history = agent_data.messages
            tool_response_text, tool_reward, tool_status = await self.simulator.call_tool(
                interaction_history,
                tool_call_text,
                agent_data.tool_schemas,
                user_intent="" if self.ablate_user_intent else (getattr(agent_data, "user_intent", "") or ""),
            )

        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        return tool_response_text, tool_reward, tool_status


@ray.remote
class SimulatorAgentLoopWorker(AgentLoopWorkerBase):
    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], reward_router_address: str = None):
        # Import the module to ensure SimulatorAgentLoop is registered
        import recipe.trustee.agent_loop  # noqa: F401
        super().__init__(config, server_handles, reward_router_address)
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        # Accumulated token counters across all training steps for this worker
        self._simulator_input_tokens_total: int = 0
        self._simulator_output_tokens_total: int = 0

    @tqbridge()
    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=batch.meta_info["temperature"] if "temperature" in batch.meta_info else config.temperature,
            top_p=batch.meta_info["top_p"] if "top_p" in batch.meta_info else config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        is_validate = batch.meta_info.get("validate", False)
        if is_validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature
        batch.non_tensor_batch["validate"] = np.array([is_validate] * len(batch), dtype=object)

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker

        # For n rollouts per sample, we trace all n rollouts for selected samples
        # Note: This sampling happens per-worker, so total traces = max_samples_per_worker * num_workers * n
        if max_samples_per_worker is not None:
            unique_sample_indices = np.unique(index)
            if max_samples_per_worker < len(unique_sample_indices):
                selected_samples = set(
                    np.random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist()
                )
                traced_indices = set(i for i in range(len(batch)) if index[i] in selected_samples)
            else:
                traced_indices = set(range(len(batch)))
        else:
            traced_indices = set(range(len(batch)))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        tasks = []
        for i in range(len(batch)):
            trace_this_sample = i in traced_indices
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(sampling_params, trajectory_info[i], trace=trace_this_sample, **kwargs)
                )
            )
        outputs = await asyncio.gather(*tasks)

        output = self._postprocess(outputs)

        return output

    async def _agent_loop_postprocess(self, output, **kwargs) -> _InternalAgentLoopOutput:
        """Perform post-processing operations on the output of each individual agent loop."""
        assert "user_rewards" in output.extra_fields, "user_rewards should be in extra_fields"
        assert "tool_rewards" in output.extra_fields, "tool_rewards should be in extra_fields"
        output.extra_fields["raw_prompt"] = kwargs["raw_prompt"]

        # Some AgentLoop may have already computed the reward score, e.g SWE-agent.

        # NOTE: consistent with the legacy batch version of generate_sequences that existed in the
        # deprecated vLLM SPMD rollout implementation.
        # prompt_ids: left padded with zeros (e.g., [0,0,0,0,1,2,3,4])
        # response_ids: right padded with zeros (e.g., [5,6,7,8,0,0,0,0])
        # input_ids: concatenation of prompt + response
        # Mask:
        # For example, if the prompt is [1,2,3,4] and the response is [5,6,7,(tool start)8,9(tool end),10,11,12]
        # - prompt_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [0,0,0,0,1,1,1,1]
        # - response_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
        # attention_mask: concatenation of prompt_attention_mask and response_attention_mask
        #   e.g., [0,0,0,0,1,1,1,1(prompt),1,1,1,1,1,1,1,1,1,1,1,0,0,0,0(response)]
        # - response_mask: 1s for LLM generated tokens, 0 for tool response/padding tokens
        #   e.g., [1,1,1,1,1,1,1,(tool start),0,0(tool end),1,1,0,0,0,0]
        # - position_ids: sequential positions for tokens, starting at 0
        #   e.g., [0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,0,0,0]

        # TODO(wuxibin): remove padding and use tensordict.
        self.tokenizer.padding_side = "left"

        prompt_output = self.tokenizer.pad(
            {"input_ids": output.prompt_ids},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if prompt_output["input_ids"].dim() == 1:
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
            prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

        self.tokenizer.padding_side = "right"

        response_output = self.tokenizer.pad(
            {"input_ids": output.response_ids},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if response_output["input_ids"].dim() == 1:
            response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
            response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

        response_mask_output = self.tokenizer.pad(
            {"input_ids": output.response_mask},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        if response_mask_output["input_ids"].dim() == 1:
            response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

        assert "tool_rewards" in output.extra_fields, "tool_rewards should be in extra_fields"
        assert "user_rewards" in output.extra_fields, "user_rewards should be in extra_fields"

        if "tool_rewards" in output.extra_fields:
            tool_rewards = torch.tensor(output.extra_fields["tool_rewards"])
            tool_rewards = self.tokenizer.pad(
                {"input_ids": tool_rewards},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.response_length,
                return_tensors="pt",
                return_attention_mask=False,
            )
            if tool_rewards["input_ids"].dim() == 1:
                tool_rewards["input_ids"] = tool_rewards["input_ids"].unsqueeze(0)
            output.extra_fields["tool_rewards"] = tool_rewards["input_ids"]

        if "user_rewards" in output.extra_fields:
            user_rewards = torch.tensor(output.extra_fields["user_rewards"])
            user_rewards = self.tokenizer.pad(
                {"input_ids": user_rewards},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.response_length,
                return_tensors="pt",
                return_attention_mask=False,
            )
            if user_rewards["input_ids"].dim() == 1:
                user_rewards["input_ids"] = user_rewards["input_ids"].unsqueeze(0)
            output.extra_fields["user_rewards"] = user_rewards["input_ids"]

        response_logprobs = None
        if output.response_logprobs is not None:
            pad_size = self.config.actor_rollout_ref.rollout.response_length - len(output.response_logprobs)
            response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

        response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
        attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
        input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

        routed_experts = None
        if output.routed_experts is not None:
            total_length = input_ids.shape[1]
            length, layer_num, topk_num = output.routed_experts.shape
            experts_tensor = torch.from_numpy(output.routed_experts)
            routed_experts = torch.zeros(1, total_length, layer_num, topk_num, dtype=experts_tensor.dtype)

            # Calculate start position: left padding means original prompt starts at the end
            start_pos = prompt_output["input_ids"].shape[1] - len(output.prompt_ids)
            end_pos = min(start_pos + length, total_length)

            # Add boundary checks for robustness
            if start_pos < 0 or end_pos > total_length:
                raise ValueError(
                    f"Invalid position range: start_pos={start_pos}, end_pos={end_pos}, total_length={total_length}"
                )

            routed_experts[:, start_pos:end_pos] = experts_tensor.unsqueeze(0)

        multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
        position_ids = self._compute_position_ids(input_ids, attention_mask, multi_modal_inputs)
        await self._compute_score(
            output,
            prompts=prompt_output["input_ids"],
            responses=response_output["input_ids"],
            attention_mask=attention_mask,
            input_ids=input_ids,
            position_ids=position_ids,
            kwargs=kwargs,
        )

        assert "user_rewards" in output.extra_fields, "user_rewards should be in extra_fields"
        assert "tool_rewards" in output.extra_fields, "tool_rewards should be in extra_fields"

        return _InternalAgentLoopOutput(
            prompt_ids=prompt_output["input_ids"],
            response_ids=response_output["input_ids"],
            input_ids=input_ids,
            position_ids=position_ids,
            response_mask=response_mask,
            attention_mask=attention_mask,
            response_logprobs=response_logprobs,
            routed_experts=routed_experts,
            multi_modal_inputs=multi_modal_inputs,
            multi_modal_data=output.multi_modal_data,
            reward_score=output.reward_score,
            num_turns=output.num_turns,
            metrics=output.metrics,
            extra_fields=output.extra_fields,
        )

    def _postprocess(self, inputs: list[_InternalAgentLoopOutput]) -> DataProto:
        """Process the padded outputs from _run_agent_loop and combine them into a batch."""
        # Convert lists back to tensors and stack them to create a batch.
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)
        if inputs[0].routed_experts is not None:
            optional_outputs["routed_experts"] = torch.cat([input.routed_experts for input in inputs], dim=0)
        if "tool_rewards" in inputs[0].extra_fields:
            optional_outputs["tool_rewards"] = torch.cat([input.extra_fields["tool_rewards"] for input in inputs], dim=0)
        if "user_rewards" in inputs[0].extra_fields:
            optional_outputs["user_rewards"] = torch.cat([input.extra_fields["user_rewards"] for input in inputs], dim=0)
        optional_outputs["final_reward"] = torch.tensor([[input.extra_fields["final_reward"]] for input in inputs])

        assert "user_rewards" in optional_outputs, "user_rewards should be in optional_outputs"
        assert "tool_rewards" in optional_outputs, "tool_rewards should be in optional_outputs"
        assert "final_reward" in optional_outputs, "final_reward should be in optional_outputs"

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                # position_ids: [bsz, 3, prompt_length + response_length] or [bsz, prompt_length + response_length]
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            prompt_length = prompt_ids.size(1)
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }

        # add reward_extra_info to non_tensor_batch
        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = list(reward_extra_infos[0].keys())
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        # Add multi_modal_inputs to non_tensor_batch if any samples have them
        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]
        # Collect extra fields from all inputs and convert them to np.ndarray
        extra_fields = {}
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields)
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
            extra_fields[key] = temp_arr

        non_tensor_batch.update(extra_fields)

        # Aggregate simulator token stats across all samples in this batch
        input_tokens_step = sum(
            int(input.extra_fields.get("simulator_input_tokens", 0)) for input in inputs
        )
        output_tokens_step = sum(
            int(input.extra_fields.get("simulator_output_tokens", 0)) for input in inputs
        )
        self._simulator_input_tokens_total += input_tokens_step
        self._simulator_output_tokens_total += output_tokens_step

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={
                "metrics": metrics,
                "reward_extra_keys": reward_extra_keys,
                "simulator_input_tokens_step": input_tokens_step,
                "simulator_output_tokens_step": output_tokens_step,
                "simulator_input_tokens_total": self._simulator_input_tokens_total,
                "simulator_output_tokens_total": self._simulator_output_tokens_total,
            },
        )


class SimulatorAgentLoopManager(AgentLoopManager):
    """Agent loop manager that manages a group of agent loop workers."""

    def __init__(
        self, config: DictConfig, worker_group: RayWorkerGroup = None, rm_resource_pool: RayResourcePool = None
    ):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): trainer config.
            worker_group (RayWorkerGroup): ActorRolloutRef worker group for hybrid mode; None for standalone mode.
            rm_resource_pool (RayResourcePool): Resource pool for reward model (Standalone mode).
        """
        self.agent_loop_workers_class = SimulatorAgentLoopWorker
        super().__init__(config, worker_group, rm_resource_pool)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences and aggregate simulator token stats across all workers."""
        import ray

        self.wake_up()
        if self.reward_model_manager:
            self.reward_model_manager.wake_up()

        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )

        # Aggregate simulator token stats across all workers BEFORE DataProto.concat
        # (concat requires non-metric meta_info keys to be identical across outputs)
        input_tokens_step = sum(out.meta_info.pop("simulator_input_tokens_step", 0) for out in outputs)
        output_tokens_step = sum(out.meta_info.pop("simulator_output_tokens_step", 0) for out in outputs)
        input_tokens_total = sum(out.meta_info.pop("simulator_input_tokens_total", 0) for out in outputs)
        output_tokens_total = sum(out.meta_info.pop("simulator_output_tokens_total", 0) for out in outputs)

        output = DataProto.concat(outputs)
        self.sleep()
        if self.reward_model_manager:
            self.reward_model_manager.sleep()

        # Pop per-worker metrics and compute timing (same as base class)
        metrics = [out.meta_info.pop("metrics") for out in outputs]
        timing = self._performance_metrics(metrics, output)

        output.meta_info = {
            "timing": timing,
            **outputs[0].meta_info,
            "simulator_input_tokens_step": input_tokens_step,
            "simulator_output_tokens_step": output_tokens_step,
            "simulator_input_tokens_total": input_tokens_total,
            "simulator_output_tokens_total": output_tokens_total,
        }
        return output