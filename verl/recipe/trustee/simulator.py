import asyncio
import json
import re
from typing import Any

import httpx
from openai import AsyncOpenAI

from recipe.trustee.utils.templates.tool_simulation import \
    get_tool_simulation_prompt
from recipe.trustee.utils.templates.user_simulation import \
    get_user_simulation_prompt
from recipe.trustee.utils.templates.verifier_simulation import \
    get_verifier_simulation_prompt
from verl.experimental.agent_loop.agent_loop import DictConfigWrap


class Simulator:
    def __init__(self, config: DictConfigWrap):
        self.config = config
        self.url = config.url
        self.api_key = config.api_key
        self.model = config.model
        self.max_tokens = config.max_tokens
        self.timeout = config.timeout
        self.frequency_penalty = config.frequency_penalty
        self.enable_thinking = config.enable_thinking
        self.verbose = config.verbose
        self.max_retries = getattr(config, 'max_retries', 3)
        self.max_retries_verifier = getattr(config, 'max_retries_verifier', 5)  # Higher for verifier
        self.model_size = getattr(config, 'model_size', 'medium')  # 'small', 'medium', 'large'

        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.url)

        # Token usage tracking (per-step and accumulated totals)
        self._input_tokens_step: int = 0
        self._output_tokens_step: int = 0
        self._input_tokens_total: int = 0
        self._output_tokens_total: int = 0

    async def _call_llm(
        self, 
        prompt: str, 
        max_tokens: int, 
        timeout: int, 
        enable_thinking_override: bool | None = None
    ) -> str:
        """Call LLM with optional thinking override (for disabling thinking in verifier)."""
        # Use override if provided, otherwise use config
        enable_thinking = enable_thinking_override if enable_thinking_override is not None else self.enable_thinking
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
            timeout=timeout,
            frequency_penalty=self.frequency_penalty,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            }
        )
        # Track token usage
        if response.usage is not None:
            self._input_tokens_step += response.usage.prompt_tokens
            self._output_tokens_step += response.usage.completion_tokens
            self._input_tokens_total += response.usage.prompt_tokens
            self._output_tokens_total += response.usage.completion_tokens
        return response.choices[0].message.content.strip()

    def reset_step_tokens(self) -> None:
        """Reset per-step token counters (call at the start of each rollout step)."""
        self._input_tokens_step = 0
        self._output_tokens_step = 0

    def get_token_stats(self) -> dict[str, int]:
        """Return current token usage stats (per-step and accumulated totals)."""
        return {
            "input_tokens_step": self._input_tokens_step,
            "output_tokens_step": self._output_tokens_step,
            "input_tokens_total": self._input_tokens_total,
            "output_tokens_total": self._output_tokens_total,
        }

    def _find_json_in_text(self, text: str) -> str | None:
        """
        Robustly find the first complete JSON object {...} in text.
        Returns the JSON string if found, None otherwise.
        Handles truncated JSON by finding the longest valid JSON block.
        """
        # Try to find {...} patterns
        depth = 0
        start_idx = -1
        
        for i, char in enumerate(text):
            if char == '{':
                if depth == 0:
                    start_idx = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start_idx >= 0:
                    # Found a complete JSON object
                    candidate = text[start_idx:i+1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        # Invalid JSON, keep searching
                        start_idx = -1
        
        return None

    def _repair_json(self, text: str) -> str | None:
        """
        Attempt to repair truncated or malformed JSON.
        Strategies:
        1. Extract just the reward field using regex (highest priority — always safe)
        2. Find the last {...} block and try to close it with matching braces
        3. Close truncated string values then add closing braces
        """
        # Strategy 1: Extract reward field directly via regex (works even with truncated text)
        # Matches: "reward": -1 / "reward": 0 / "reward": 1 (or floats)
        reward_match = re.search(r'"reward"\s*:\s*(-?[01](?:\.\d+)?)', text)
        if reward_match:
            try:
                reward_val = float(reward_match.group(1))
                if -1 <= reward_val <= 1:
                    return json.dumps({"reward": reward_val})
            except (ValueError, TypeError):
                pass

        # Strategy 2: Find last opening brace and try to close it with matching braces
        last_brace_idx = text.rfind('{')
        if last_brace_idx >= 0:
            candidate = text[last_brace_idx:]
            open_count = candidate.count('{') - candidate.count('}')
            if open_count > 0:
                repaired = candidate + '}' * open_count
                try:
                    json.loads(repaired)
                    return repaired
                except json.JSONDecodeError:
                    pass

        # Strategy 3: Close truncated string value then close object.
        # Handles: {"reward": 1, "reasoning": "The agent successfully... (no closing " or })
        last_brace_idx = text.rfind('{')
        if last_brace_idx >= 0:
            candidate = text[last_brace_idx:]
            open_count = candidate.count('{') - candidate.count('}')
            if open_count > 0:
                # Close any open string by appending a quote, then close all open braces
                repaired = candidate + '"' + '}' * open_count
                try:
                    json.loads(repaired)
                    return repaired
                except json.JSONDecodeError:
                    pass

        return None

    def _extract_json(self, content: str) -> dict:
        """
        Robustly extract JSON from response with multiple fallback strategies.
        Handles:
        - Markdown code blocks (```json...```, ```...```)
        - Thinking blocks (<think>...</think>)
        - Commentary before/after JSON
        - Truncated JSON
        - Missing fields
        """
        original_content = content
        
        # Step 1: Strip markdown code blocks
        if '```json' in content:
            try:
                content = content.split('```json')[1].split('```')[0].strip()
            except IndexError:
                pass
        elif '```' in content:
            try:
                content = content.split('```')[1].split('```')[0].strip()
            except IndexError:
                pass
        
        # Step 2: Strip thinking blocks
        if '</think>' in content:
            try:
                content = content.split('</think>', 1)[1].strip()
                # After stripping thinking, check for code fences again
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0].strip()
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0].strip()
            except IndexError:
                pass
        elif '<think>' in content:
            # Truncated thinking block: <think> opened but never closed.
            # Any JSON must appear after the thinking section, but since the response
            # was cut off mid-think there is no JSON to recover — return empty to
            # trigger the repair/fallback path later.
            content = ""
        
        # Step 3: Try direct JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Step 4: Find JSON object in text (handles surrounding commentary)
        json_str = self._find_json_in_text(content)
        if json_str:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # Step 5: Try to find JSON in original content if not in stripped version
        if content != original_content:
            json_str = self._find_json_in_text(original_content)
            if json_str:
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
        
        # Step 6: Attempt repair for truncated JSON
        repaired = self._repair_json(content)
        if repaired:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
        
        # Step 7: Last resort - try repairing original
        if content != original_content:
            repaired = self._repair_json(original_content)
            if repaired:
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass
        
        # All strategies failed
        raise json.JSONDecodeError(
            f"Could not extract valid JSON from content. Tried direct parse, finding objects, and repair.",
            content,
            0
        )

    def _get_model_criteria_count(self, base_criteria: int) -> int:
        """
        Reduce num_criteria based on model size for verifier complexity.
        Qwen3-8B should use fewer criteria due to token pressure.
        """
        if self.model_size == 'small':  # 8B models
            return min(base_criteria, 2)
        elif self.model_size == 'medium':  # 14B models
            return min(base_criteria, 3)
        else:  # 'large' (32B+)
            return base_criteria

    async def call_tool(
        self,
        interaction_history: list[dict[str, Any]],
        tool_call_text: str,
        tool_schemas: list[dict[str, Any]],
        user_intent: str = "",
    ) -> tuple[str, float, str]:
        """Call tool and return tool response and reward."""
        prompt = get_tool_simulation_prompt(
            interaction_history, tool_call_text, tool_schemas, user_intent=user_intent
        )

        for attempt in range(self.max_retries):
            try:
                raw_response = await self._call_llm(prompt, self.max_tokens.tool, self.timeout.tool)
            except (asyncio.TimeoutError, httpx.TimeoutException, TimeoutError) as e:
                if attempt == self.max_retries - 1:
                    if self.verbose:
                        print(f"call_tool timed out after {self.max_retries} attempts: {e}")
                    return "Error: Failed to simulate tool execution", -1.0, "aborted_timeout"
                await asyncio.sleep(0.5)
                continue
            except Exception as e:
                if attempt == self.max_retries - 1:
                    if self.verbose:
                        print(f"call_tool LLM call failed after {self.max_retries} attempts: {e}")
                    return "Error: Failed to simulate tool execution", -1.0, "aborted_timeout"
                await asyncio.sleep(0.5)
                continue

            try:
                response = self._extract_json(raw_response)
            except (json.JSONDecodeError, ValueError) as e:
                if attempt == self.max_retries - 1:
                    if self.verbose:
                        print(f"call_tool JSON parse failed after {self.max_retries} attempts: {e}")
                    return "Error: Failed to simulate tool execution", -1.0, "aborted_parse"
                await asyncio.sleep(0.5)
                continue

            try:
                # Validate response structure
                if not isinstance(response, dict):
                    raise ValueError("Response must be a dict")
                if "result" not in response:
                    raise ValueError("Missing 'result' field")
                if "reward" not in response:
                    raise ValueError("Missing 'reward' field")

                tool_response_text = str(response["result"])
                tool_reward = float(response["reward"])

                # Validate reward range (-1, 0, 1)
                if not (-1 <= tool_reward <= 1):
                    tool_reward = max(-1.0, min(1.0, tool_reward))

                return tool_response_text, tool_reward, "normal"

            except (ValueError, TypeError, KeyError) as e:
                if attempt == self.max_retries - 1:
                    if self.verbose:
                        print(f"call_tool validation failed after {self.max_retries} attempts: {e}")
                    return "Error: Failed to simulate tool execution", -1.0, "aborted_validation"
                await asyncio.sleep(0.5)

    async def call_user(
        self,
        interaction_history: list[dict[str, Any]],
        user_message: str,
        user_query: str,
        tool_schemas: list[dict[str, Any]],
        user_intent: str = "",
        user_persona: str = "",
    ) -> tuple[bool, str, float, str]:
        """Call user and return termination flag, user response, reward, and status."""
        prompt = get_user_simulation_prompt(
            interaction_history, user_message, user_query, tool_schemas,
            user_intent=user_intent, user_persona=user_persona,
        )
        
        for attempt in range(self.max_retries):
            try:
                raw_response = await self._call_llm(prompt, self.max_tokens.user, self.timeout.user)
            except (asyncio.TimeoutError, httpx.TimeoutException, TimeoutError) as e:
                if attempt == self.max_retries - 1:
                    if self.verbose:
                        print(f"call_user timed out after {self.max_retries} attempts: {e}")
                    return False, "I need more time to think about this.", 0.0, "aborted_timeout"
                await asyncio.sleep(0.5)
                continue
            except Exception as e:
                if attempt == self.max_retries - 1:
                    if self.verbose:
                        print(f"call_user LLM call failed after {self.max_retries} attempts: {e}")
                    return False, "I need more time to think about this.", 0.0, "aborted_timeout"
                await asyncio.sleep(0.5)
                continue

            try:
                response = self._extract_json(raw_response)
            except (json.JSONDecodeError, ValueError) as e:
                if attempt == self.max_retries - 1:
                    if self.verbose:
                        print(f"call_user JSON parse failed after {self.max_retries} attempts: {e}")
                    return False, "I need more time to think about this.", 0.0, "aborted_parse"
                await asyncio.sleep(0.5)
                continue

            try:
                # Validate response structure
                if not isinstance(response, dict):
                    raise ValueError("Response must be a dict")
                if "response" not in response:
                    raise ValueError("Missing 'response' field")
                if "reward" not in response:
                    raise ValueError("Missing 'reward' field")

                user_response_text = str(response["response"])
                user_reward = float(response["reward"])
                # Validate reward range (-1, 0, 1)
                if not (-1 <= user_reward <= 1):
                    user_reward = max(-1.0, min(1.0, user_reward))

                should_terminate_sequence = user_response_text == ""

                return should_terminate_sequence, user_response_text, user_reward, "normal"

            except (ValueError, TypeError, KeyError) as e:
                if attempt == self.max_retries - 1:
                    if self.verbose:
                        print(f"call_user validation failed after {self.max_retries} attempts: {e}")
                    return False, "I need more time to think about this.", 0.0, "aborted_validation"
                await asyncio.sleep(0.5)

    async def reward(
        self,
        interaction_history: list[dict[str, Any]],
        user_query: str,
        tool_schemas: list[dict[str, Any]],
        expected_tool_calls: list[str] | None = None,
        user_intent: str = "",
        user_persona: str = "",
        num_criteria: int = 5,
    ) -> tuple[float, str]:
        """
        Verify the agent's performance; reward in [-1, 0, 1] based on num_criteria.
        Enhanced with:
        - Higher retry count for verifier (max_retries_verifier instead of max_retries)
        - Thinking disabled for verifier to reduce token pressure
        - Model-size aware criteria reduction
        """
        # Reduce criteria based on model size
        num_criteria = self._get_model_criteria_count(num_criteria)
        
        prompt = get_verifier_simulation_prompt(
            interaction_history, user_query, tool_schemas,
            expected_tool_calls=expected_tool_calls,
            user_intent=user_intent, user_persona=user_persona,
            num_criteria=num_criteria,
        )
        
        # Use higher retry count for verifier
        verifier_max_retries = self.max_retries_verifier

        for attempt in range(verifier_max_retries):
            try:
                raw_response = await self._call_llm(
                    prompt,
                    self.max_tokens.verifier,
                    self.timeout.verifier,
                )
            except (asyncio.TimeoutError, httpx.TimeoutException, TimeoutError) as e:
                if attempt == verifier_max_retries - 1:
                    if self.verbose:
                        print(f"reward timed out after {verifier_max_retries} attempts: {e}")
                    return 0.0, "aborted_timeout"
                await asyncio.sleep(0.5)
                continue
            except Exception as e:
                if attempt == verifier_max_retries - 1:
                    if self.verbose:
                        print(f"reward LLM call failed after {verifier_max_retries} attempts: {e}")
                    return 0.0, "aborted_timeout"
                await asyncio.sleep(0.5)
                continue

            try:
                response = self._extract_json(raw_response)
            except (json.JSONDecodeError, ValueError) as e:
                if attempt == verifier_max_retries - 1:
                    if self.verbose:
                        print(f"reward JSON parse failed after {verifier_max_retries} attempts: {e}")
                    return 0.0, "aborted_parse"
                await asyncio.sleep(0.5)
                continue

            try:
                if not isinstance(response, dict) or "reward" not in response:
                    raise ValueError("Response must be a dict with 'reward'")
                final_reward = float(response["reward"])
                # Verifier returns -1, 0, or 1
                if not (-1 <= final_reward <= 1):
                    final_reward = max(-1.0, min(1.0, final_reward))
                return final_reward, "normal"

            except (ValueError, TypeError, KeyError) as e:
                if attempt == verifier_max_retries - 1:
                    if self.verbose:
                        print(f"reward validation failed after {verifier_max_retries} attempts: {e}")
                    return 0.0, "aborted_validation"
                await asyncio.sleep(0.5)
