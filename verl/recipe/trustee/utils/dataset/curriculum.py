import asyncio
import json
import random
from math import floor
from typing import Any, Dict, List

from openai import AsyncOpenAI

from recipe.trustee.utils.templates.system_prompt import get_system_prompt
from recipe.trustee.utils.templates.task_generation import \
    get_task_generation_prompt


def load_curriculum_dataset(data_files: List[str] | str) -> Dict:
    """Load categorized tools from JSON files.
    
    Args:
        data_files: List of file paths to categorized_tools.json files
        
    Returns:
        dict: Dictionary with categories as keys and toolsets as values
    """
    if isinstance(data_files, str):
        data_files = [data_files]
    dataset = {}
    for data_file in data_files:
        with open(data_file, "r") as f:
            data = json.load(f)
            dataset.update(data)
    return dataset


class CurriculumDataLoader:
    """DataLoader for curriculum learning that dynamically generates training data.
    
    This dataloader samples tools from categorized_tools.json and generates user queries
    dynamically with adaptive difficulty based on training metrics. The difficulty
    increases when the agent performs well and decreases when it struggles.
    """
    
    def __init__(
        self,
        config,
        dataset,
        batch_size,
        total_training_steps,
        tokenizer=None,
        max_prompt_length=4096,
        collate_fn=None,
        format: str = "hermes",
    ):
        """Initialize curriculum dataloader.
        
        Args:
            config: config.data.curriculum configuration object
            dataset: Categorized tools dictionary from categorized_tools.json
            batch_size: Number of samples per batch
            total_training_steps: Total number of training steps
            tokenizer: Tokenizer for length checking (optional)
            max_prompt_length: Maximum prompt length in tokens
        """
        self.config_curriculum = config
        self.dataset = dataset  # categorized_tools.json

        self.batch_size = batch_size
        self.total_training_steps = total_training_steps
        self.current_step = 0
        
        # Tokenizer for length checking
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.collate_fn = collate_fn
        self.format = format  # tool-call format ("hermes" or "llama-3.1")

        # Difficulty level 1-100; when agent performs well, level += step (m). Only place that manages level.
        self.difficulty_level = float(getattr(config.difficulty, "level_initial", 1))
        self.difficulty_level_max = float(getattr(config.difficulty, "level_max", 100))
        self.difficulty_step = float(getattr(config.difficulty, "step", 1))

        # Multipliers: single dict; all derived values computed here from level + self.difficulty_multipliers
        default_mult = {"num_tools": 0.3, "expected_tool_calls": 0.2, "user_persona": 0.02, "query_ambiguity": 0.02, "expected_turns": 0.02, "num_criteria": 0.04, "system_prompt": 0.03, "max_user_turns": 0.1, "max_assistant_turns": 0.2, "max_tool_turns": 0.1}
        mult = getattr(config.difficulty, "multipliers", None)
        if mult is None:
            self.difficulty_multipliers = dict(default_mult)
        else:
            self.difficulty_multipliers = {k: float(mult.get(k, default_mult[k])) for k in default_mult}
        self.min_turns = int(getattr(config.difficulty, "min_turns", 1))
        self.reward_threshold_upper = float(getattr(config, "reward_threshold_upper", 0.5))
        self.reward_threshold_lower = float(getattr(config, "reward_threshold_lower", 0.0))
        self.soft_curriculum_epsilon = float(getattr(config, "epsilon", 0.0))
        # Aspects listed here are always computed at level_max (pinned at max difficulty)
        self.fixed_aspects = set(getattr(config.difficulty, "fixed_aspects", []) or [])

        # Initialize AsyncOpenAI client for query generation
        client_kwargs = {}
        if config.get("api_base"):
            client_kwargs["base_url"] = config.get("api_base")
        api_key = config.get("api_key")
        if api_key is None:
            import os
            api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        client_kwargs["api_key"] = api_key
        
        self.llm_client = AsyncOpenAI(**client_kwargs)
        self.llm_model = config.get("model", "gpt-4o-mini")
        self.llm_max_retries = config.get("max_retries", 5)
        self.llm_timeout = config.get("timeout", 120)
        self.llm_max_tokens = config.get("max_tokens", 8192)
        
        # Concurrency control
        self.llm_concurrency = config.get("concurrency", 32)
        self.semaphore = asyncio.Semaphore(self.llm_concurrency)

        # Curriculum update settings
        self.update_frequency = config.get("update_frequency", 1)  # update every N steps
        self.steps_since_update = 0

        # Preprocess dataset for efficient sampling
        self._preprocess_dataset()
        
        print(f"[Curriculum] Difficulty level: {self.difficulty_level} (max {self.difficulty_level_max}, step +{self.difficulty_step})")
        print(f"[Curriculum] difficulty_multipliers: {self.difficulty_multipliers}")
        print(f"[Curriculum] LLM concurrency: {self.llm_concurrency}")

    def __len__(self):
        return self.total_training_steps

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_step >= self.total_training_steps:
            raise StopIteration

        batch = self.get_batch()
        self.current_step += 1

        return batch

    def _preprocess_dataset(self):
        """Preprocess the categorized tools for efficient sampling."""
        self.categories = list(self.dataset.keys())
        self.category_to_toolsets = {}
        self.toolset_to_tools = {}

        for category, toolsets in self.dataset.items():
            self.category_to_toolsets[category] = list(toolsets.keys())
            for toolset_name, tools in toolsets.items():
                key = f"{category}/{toolset_name}"
                self.toolset_to_tools[key] = tools

    def get_batch(self) -> List[Dict]:
        """Generate a batch of data by sampling tools and generating queries.
        
        Returns:
            List[Dict]: List of training samples with query, tools, and metadata
        """
        # Run async batch generation in a synchronous context
        return asyncio.run(self._get_batch_async())

    async def _get_batch_async(self) -> List[Dict]:
        """Async implementation of batch generation for concurrent LLM calls."""
        batch_data = []
        attempts = 0
        max_attempts = self.config_curriculum.max_attempts
        
        while len(batch_data) < self.batch_size and attempts < max_attempts:
            attempts += 1
            num_needed = self.batch_size - len(batch_data)
            
            # Generate 2x the needed samples to account for failures
            num_to_generate = num_needed * 2 if attempts > 1 else num_needed
            
            # Prepare batch tasks
            tasks = [self._generate_single_sample() for _ in range(num_to_generate)]
            
            # Execute all tasks concurrently
            new_samples = await asyncio.gather(*tasks)
            
            # Filter out None values (failed generations)
            valid_samples = []
            for sample in new_samples:
                if sample is not None:
                    valid_samples.extend(sample)
            
            if len(valid_samples) == 0 and attempts > 1:
                # If generation keeps failing, temporarily lower difficulty level
                print(f"[Curriculum] Warning: Generation failing, lowering difficulty_level temporarily")
                original_level = self.difficulty_level
                self.difficulty_level = max(1.0, self.difficulty_level - 10)
                tasks = [self._generate_single_sample() for _ in range(num_to_generate)]
                new_samples = await asyncio.gather(*tasks)
                valid_samples = []
                for sample in new_samples:
                    if sample is not None:
                        valid_samples.extend(sample)
                self.difficulty_level = original_level
            
            batch_data.extend(valid_samples)
            
            # if len(valid_samples) > 0:
            #     print(f"[Curriculum] Generated {len(valid_samples)} valid samples, total: {len(batch_data)}/{self.batch_size}")
        
        # If we have too many samples, randomly select batch_size samples
        if len(batch_data) > self.batch_size:
            batch_data = random.sample(batch_data, self.batch_size)
        
        # Last resort: if still not enough, pad with simple queries
        while len(batch_data) < self.batch_size:
            print(f"[Curriculum] Warning: Padding with fallback samples ({len(batch_data)}/{self.batch_size})")
            tools = self._sample_tools()
            query = "Help me with the task."
            params = self._get_difficulty_params()
            batch_data.append(self._format_sample(query, tools, [], "", "", params["num_criteria"], None,
                                                   params["max_user_turns"], params["max_assistant_turns"], params["max_tool_turns"]))

        batch_data = self.collate_fn(batch_data)
        
        return batch_data

    def _get_difficulty_params(self) -> Dict[str, int]:
        """Single place for all difficulty-derived values. Uses difficulty_level and difficulty_multipliers only.
        With probability epsilon (config), soft curriculum randomizes: num_tools, num_expected_calls,
        user_persona_level, query_ambiguity_level, expected_turns_level. num_criteria, system_prompt_level,
        max_user_turns, max_assistant_turns, max_tool_turns are always fixed.

        Aspects listed in self.fixed_aspects are always computed at level_max (pinned at max difficulty).
        """
        m = self.difficulty_multipliers
        level = self.difficulty_level
        level_max = self.difficulty_level_max
        fa = self.fixed_aspects

        def L(aspect_key: str) -> float:
            """Return level_max if aspect is pinned (fixed), else current level."""
            return level_max if aspect_key in fa else level

        num_tools = max(1, floor(m["num_tools"] * L("num_tools")))
        num_expected_calls = max(1, floor(m["expected_tool_calls"] * L("expected_tool_calls")))
        user_persona_level = min(2, max(0, floor(m["user_persona"] * L("user_persona"))))
        query_ambiguity_level = min(2, max(0, floor(m["query_ambiguity"] * L("query_ambiguity"))))
        expected_turns_level = min(2, max(0, floor(m["expected_turns"] * L("expected_turns"))))
        if self.soft_curriculum_epsilon > 0 and random.random() < self.soft_curriculum_epsilon:
            max_tools = max(1, floor(m["num_tools"] * level_max))
            max_calls = max(1, floor(m["expected_tool_calls"] * level_max))
            if "num_tools" not in fa:
                num_tools = random.randint(1, max_tools)
            if "expected_tool_calls" not in fa:
                num_expected_calls = random.randint(1, max_calls)
            if "user_persona" not in fa:
                user_persona_level = random.randint(0, 2)
            if "query_ambiguity" not in fa:
                query_ambiguity_level = random.randint(0, 2)
            if "expected_turns" not in fa:
                expected_turns_level = random.randint(0, 2)
        return {
            "num_tools": num_tools,
            "num_expected_calls": num_expected_calls,
            "user_persona_level": user_persona_level,
            "query_ambiguity_level": query_ambiguity_level,
            "expected_turns_level": expected_turns_level,
            "num_criteria": min(5, max(1, 1 + floor(m["num_criteria"] * L("num_criteria")))),
            "system_prompt_level": min(3, max(0, floor(m["system_prompt"] * L("system_prompt")))),
            "max_user_turns": max(self.min_turns, floor(m["max_user_turns"] * L("max_user_turns"))),
            "max_assistant_turns": max(2 * self.min_turns, floor(m["max_assistant_turns"] * L("max_assistant_turns"))),
            "max_tool_turns": max(self.min_turns, floor(m["max_tool_turns"] * L("max_tool_turns"))),
        }

    async def _generate_single_sample(self) -> Dict:
        """Generate a single training sample via task generator (expected_tool_calls, user_intent, user_persona, first_user_query)."""
        # Sample tools: num_tools = max(1, 0.3 * difficulty_level)
        tools = self._sample_tools()

        params = self._get_difficulty_params()
        prompt = get_task_generation_prompt(
            tools=tools,
            num_expected_calls=params["num_expected_calls"],
            user_persona_level=params["user_persona_level"],
            query_ambiguity_level=params["query_ambiguity_level"],
            expected_turns_level=params["expected_turns_level"],
        )
        task_result = await self._call_llm_async_task(prompt)

        if task_result:
            query = task_result.get("first_user_query", "").strip()
            expected_tool_calls = task_result.get("expected_tool_calls", [])
            if not isinstance(expected_tool_calls, list):
                expected_tool_calls = []
            user_intent = task_result.get("user_intent", "")
            user_persona = task_result.get("user_persona", "")

            system_prompt = get_system_prompt(params["system_prompt_level"], format=self.format)
            if self.tokenizer is not None:
                try:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}
                    ]
                    token_ids = self.tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tools=tools
                    )
                    if len(token_ids) > self.max_prompt_length:
                        return None
                except Exception:
                    return None

            num_criteria = params["num_criteria"]
            if random.random() < 0.01:
                print("=" * 100)
                print(f"[Curriculum] difficulty_level={self.difficulty_level} num_criteria={num_criteria}")
                print(f"[Curriculum] first_user_query: {query[:200]}...")
                print("=" * 100)

            return [self._format_sample(query, tools, expected_tool_calls, user_intent, user_persona, num_criteria, system_prompt,
                                        params["max_user_turns"], params["max_assistant_turns"], params["max_tool_turns"])]
        return None

    def _format_sample(
        self,
        query: str,
        tools: List[Dict],
        expected_tool_calls: List[str] | None = None,
        user_intent: str = "",
        user_persona: str = "",
        num_criteria: int = 5,
        system_prompt: str | None = None,
        max_user_turns: int | None = None,
        max_assistant_turns: int | None = None,
        max_tool_turns: int | None = None,
    ) -> Dict:
        """Format sample with task fields for simulators and verifier."""
        import torch

        if system_prompt is None:
            system_prompt = get_system_prompt(self._get_difficulty_params()["system_prompt_level"], format=self.format)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        out = {
            "raw_prompt": messages,
            "tools": tools,
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            "index": random.randint(0, 1000000),
        }
        if expected_tool_calls is not None:
            out["expected_tool_calls"] = expected_tool_calls
        # Always set string task fields so collate_fn yields batch-aligned arrays (optional keys per row break DataProto).
        out["user_intent"] = user_intent or ""
        out["user_persona"] = user_persona or ""
        out["num_criteria"] = num_criteria
        if max_user_turns is not None:
            out["max_user_turns"] = max_user_turns
        if max_assistant_turns is not None:
            out["max_assistant_turns"] = max_assistant_turns
        if max_tool_turns is not None:
            out["max_tool_turns"] = max_tool_turns
        return out

    def _sample_tools(self) -> List[Dict]:
        """
        Sample tools within a single domain. Schema: dataset[domain][subdomain] = list of tools.
        1) Sample one domain. 2) Sample sub-domains (toolsets) until total tools >= num_tools or all selected.
        num_tools may be softened by epsilon in _get_difficulty_params.
        """
        params = self._get_difficulty_params()
        num_tools = params["num_tools"]
        domain = random.choice(self.categories)
        subdomain_names = list(self.category_to_toolsets[domain])
        random.shuffle(subdomain_names)
        pool = []
        for sub in subdomain_names:
            key = f"{domain}/{sub}"
            pool.extend(self.toolset_to_tools[key])
            if len(pool) >= num_tools:
                break
        num_take = min(num_tools, len(pool)) if pool else 0
        if num_take == 0:
            # Fallback: take one toolset from this domain
            if subdomain_names:
                key = f"{domain}/{subdomain_names[0]}"
                pool = self.toolset_to_tools[key]
                num_take = len(pool)
        sampled = random.sample(pool, num_take) if len(pool) > num_take else pool
        return self._strip_toolset_info(sampled)

    def _strip_toolset_info(self, tools: List[Dict]) -> List[Dict]:
        """Remove toolset metadata from tools to get clean tool definitions."""
        return [
            {k: v for k, v in tool.items() if k not in ["toolset_name", "toolset_description", "category"]}
            for tool in tools
        ]

    async def _call_llm_async(self, prompt: str) -> List[Dict]:
        """Call LLM asynchronously to generate queries with retry logic.
        
        Args:
            prompt: The prompt for query generation
            
        Returns:
            List of validated query dictionaries
        """
        async with self.semaphore:
            for attempt in range(self.llm_max_retries):
                try:
                    response = await self.llm_client.chat.completions.create(
                        model=self.llm_model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        timeout=self.llm_timeout,
                        max_completion_tokens=self.llm_max_tokens,
                    )
                    content = response.choices[0].message.content.strip()

                    # Extract JSON from markdown code block if present
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0].strip()

                    result = json.loads(content)

                    # Validate JSON structure
                    if not isinstance(result, list):
                        raise ValueError("Result must be a list")

                    valid_queries = []
                    for item in result:
                        if not isinstance(item, dict):
                            continue

                        # Query must be non-empty string
                        query = item.get("query", "").strip()
                        if not query:
                            continue

                        # Expected tools must be a list
                        expected_tools = item.get("expected_tools", [])
                        if not isinstance(expected_tools, list):
                            expected_tools = []

                        valid_queries.append({
                            "query": query,
                            "expected_tools": expected_tools
                        })

                    if valid_queries:
                        return valid_queries
                    else:
                        raise ValueError("No valid queries after validation")

                except Exception as e:
                    if attempt == self.llm_max_retries - 1:
                        print(f"[Curriculum] LLM call failed after {self.llm_max_retries} attempts: {e}")
                        return []
                    # Brief delay before retry
                    await asyncio.sleep(1)

        return []

    async def _call_llm_async_task(self, prompt: str) -> Dict | None:
        """Call LLM for task generator; returns single task dict or None."""
        async with self.semaphore:
            for attempt in range(self.llm_max_retries):
                try:
                    response = await self.llm_client.chat.completions.create(
                        model=self.llm_model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        timeout=self.llm_timeout,
                        max_completion_tokens=self.llm_max_tokens,
                    )
                    content = response.choices[0].message.content.strip()
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0].strip()
                    # Strip <think>...</think> block (emitted by thinking models like Qwen3)
                    # and extract the JSON that follows it.
                    if "</think>" in content:
                        content = content.split("</think>", 1)[1].strip()
                        # After stripping the thinking block, check again for code fences
                        if "```json" in content:
                            content = content.split("```json")[1].split("```")[0].strip()
                        elif "```" in content:
                            content = content.split("```")[1].split("```")[0].strip()
                    result = json.loads(content)
                    if not isinstance(result, dict):
                        raise ValueError("Task result must be a dict")
                    if not result.get("first_user_query", "").strip():
                        raise ValueError("first_user_query required")
                    return result
                except Exception as e:
                    if attempt == self.llm_max_retries - 1:
                        print(f"[Curriculum] Task LLM call failed after {self.llm_max_retries} attempts: {e}")
                        return None
                    await asyncio.sleep(1)
        return None

    def update(self, metrics: dict):
        """Update difficulty based on training metrics using curriculum learning strategy.
        
        The difficulty is adjusted based on the agent's performance:
        - Increase difficulty when all reward metrics are above upper thresholds
        - Decrease difficulty when any reward metric is below lower threshold
        
        Args:
            metrics: Dictionary of training metrics from the current batch
        """
        self.steps_since_update += 1
        
        # Only update every N steps to avoid being too reactive
        if self.steps_since_update < self.update_frequency:
            return
        
        self.steps_since_update = 0

        final_reward = metrics.get("agent-core/final_reward/mean", 0.0)
        should_increase = final_reward >= self.reward_threshold_upper
        should_decrease = final_reward <= self.reward_threshold_lower

        if should_increase:
            self.difficulty_level = min(
                self.difficulty_level + self.difficulty_step,
                self.difficulty_level_max,
            )
        elif should_decrease:
            self.difficulty_level = max(
                self.difficulty_level - self.difficulty_step,
                1.0,
            )
            
    def get_metrics(self) -> dict:
        """Log all derived difficulty levels (single source: _get_difficulty_params())."""
        p = self._get_difficulty_params()
        return {
            "curriculum/difficulty_level": self.difficulty_level,
            "curriculum/derived_num_tools": p["num_tools"],
            "curriculum/derived_num_expected_calls": p["num_expected_calls"],
            "curriculum/derived_user_persona_level": p["user_persona_level"],
            "curriculum/derived_query_ambiguity_level": p["query_ambiguity_level"],
            "curriculum/derived_expected_turns_level": p["expected_turns_level"],
            "curriculum/derived_num_criteria": p["num_criteria"],
            "curriculum/derived_system_prompt_level": p["system_prompt_level"],
            "curriculum/derived_max_user_turns": p["max_user_turns"],
            "curriculum/derived_max_assistant_turns": p["max_assistant_turns"],
            "curriculum/derived_max_tool_turns": p["max_tool_turns"],
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "current_step": self.current_step,
            "difficulty_level": self.difficulty_level,
            "steps_since_update": self.steps_since_update,
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.current_step = state_dict["current_step"]
        self.difficulty_level = state_dict.get("difficulty_level", self.difficulty_level)
        self.steps_since_update = state_dict.get("steps_since_update", self.steps_since_update)