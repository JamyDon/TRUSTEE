import json
import textwrap

# User persona: expertise only (decoupled from query ambiguity).
USER_PERSONA_DESCRIPTIONS = {
    0: "Expert: the user has extensive prior knowledge in the domain.",
    1: "Beginner: the user has some prior knowledge but may need guidance.",
    2: "Novice: the user has no prior knowledge and may need clarification on basics.",
}

# Query ambiguity: used only for first_user_query generation (how clear vs ambiguous the message is).
QUERY_AMBIGUITY_DESCRIPTIONS = {
    0: "Clear: the first user message should be specific, complete, and unambiguous.",
    1: "Somewhat ambiguous: the first user message may omit some details or be a little vague.",
    2: "Highly ambiguous: the first user message should be vague, incomplete, or missing key information that the agent may need to clarify.",
}

# Expected number of interaction turns for the task (0=short, 1=medium, 2=long).
EXPECTED_TURNS_DESCRIPTIONS = {
    0: "One or a couple of turns: the task should be resolvable in a short exchange (e.g. 1–2 back-and-forths).",
    1: "A few to several turns: the task should naturally involve some back-and-forth (e.g. clarification, multiple steps, a handful of exchanges).",
    2: "Many turns: the task should require an extended conversation (e.g. multi-step workflow, several clarifications, or many tool uses and replies).",
}


def get_task_generation_prompt(
    tools: list[dict],
    num_expected_calls: int,
    user_persona_level: int,
    query_ambiguity_level: int,
    expected_turns_level: int = 0,
) -> str:
    """
    Format the task generator prompt. All difficulty-derived values are passed in;
    curriculum is the single place that computes them from level + difficulty_multipliers.
    """
    tools_text = json.dumps(tools, indent=2)
    # Clamp for prompt safety (curriculum already passes valid values)
    num_expected_calls = max(1, num_expected_calls)
    user_persona_level = min(2, max(0, user_persona_level))
    query_ambiguity_level = min(2, max(0, query_ambiguity_level))
    expected_turns_level = min(2, max(0, expected_turns_level))

    persona_instruction = USER_PERSONA_DESCRIPTIONS[user_persona_level]
    ambiguity_instruction = QUERY_AMBIGUITY_DESCRIPTIONS[query_ambiguity_level]
    turns_instruction = EXPECTED_TURNS_DESCRIPTIONS[expected_turns_level]

    template = f"""
You are an expert at creating realistic tasks for agent training. Given a set of tools and difficulty settings, generate ONE task.

## Available Tools
{tools_text}

## Settings
- Expected number of distinct tool calls the agent should make: approximately {num_expected_calls}
- **User persona** (user's expertise; describe this in the task): {persona_instruction}
- **Query ambiguity** (for the first user message only): {ambiguity_instruction}
  User persona and query ambiguity are independent (e.g. an expert can ask vaguely; a novice can ask clearly).
- **Expected interaction turns**: {turns_instruction}

## Your Task
Produce a single task with:
1. **Expected turns**: The task as a whole should match **expected turns**: {turns_instruction} Design the goal and first message so the conversation length fits this.
2. **expected_tool_calls**: A list of tool names (from the available tools above) that the agent is expected to use, in a plausible order. Length approximately {num_expected_calls}.
3. **user_intent**: A clear description of what the user **ultimately** wants to achieve (the ground truth goal for evaluation). This may be richer or more specific than what the user says in the first message.
4. **user_persona**: A short description of the user's expertise level, matching: {persona_instruction}. Do NOT tie this to query clarity.
5. **first_user_query**: The user's **first message only**—how a real user would actually open the conversation. Keep it **natural**:
   - It need **not** state the full intent directly; the user might hint, ask a partial question, or express a vague need that the agent must clarify.
   - Match the **query ambiguity** level: {ambiguity_instruction}
   - Do not name tools; the message should lead naturally toward using the expected tools as the conversation unfolds.
   - Avoid sounding like a spec or a list of requirements; write as a real person would (e.g. "I'm trying to figure out..." or "Can you help with something?" rather than "I want you to do X, Y, and Z").

## Output Format (JSON only)
```json
{{
    "expected_tool_calls": ["tool_name_1", "tool_name_2", ...],
    "user_intent": "Description of what the user wants to achieve.",
    "user_persona": "Short description of user expertise (not query clarity).",
    "first_user_query": "The user's first message (ambiguity as specified above)."
}}
```

Generate the task now. Output only valid JSON.
"""
    return textwrap.dedent(template).strip()
