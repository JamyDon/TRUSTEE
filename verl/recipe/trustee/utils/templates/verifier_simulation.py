import json
import textwrap

VERIFIER_CRITERIA = [
    "1. Meet the user's intent: Did the agent fulfill what the user wanted by using the tools?",
    "2. Correctly use the expected tools: Did the agent use the expected tools appropriately and correctly?",
    "3. Free of hallucination: No invented facts, tool names, or outputs.",
    "4. No redundant or incorrect trials: No unnecessary or wrong tool calls or steps.",
    "5. Concise and efficient: Response and tool use are to-the-point and efficient.",
]


def get_verifier_simulation_prompt(
    interaction_history: list[dict],
    user_query: str,
    tools: list[dict],
    expected_tool_calls: list[str] | None = None,
    user_intent: str = "",
    user_persona: str = "",
    num_criteria: int = 5,
) -> str:
    """
    Generate a prompt for evaluating agent performance. Uses the first `num_criteria`
    criteria (1 + 0.04*x, 1..5). Reward scale: 1 (all met), 0 (most met), -1 (most not met).
    """
    interaction_history_text = json.dumps(interaction_history, indent=2)
    tools_text = json.dumps(tools, indent=2)
    criteria_to_use = VERIFIER_CRITERIA[:num_criteria]
    criteria_text = "\n".join(criteria_to_use)

    extra = ""
    if expected_tool_calls:
        extra += f"\n## Expected Tool Calls (should be used correctly)\n{json.dumps(expected_tool_calls, indent=2)}\n"
    if user_intent:
        extra += f"\n## User Intent (ground truth)\n{user_intent}\n"
    if user_persona:
        extra += f"\n## User Persona\n{user_persona}\n"

    template = f"""
Your task is to evaluate how well an assistant agent performed in helping a user accomplish their task.

## User's Original Query
{user_query}
{extra}
## Available Tools
{tools_text}

## Complete Interaction Trajectory
{interaction_history_text}

## Evaluation Task

Evaluate the assistant agent's overall performance using ONLY the following {num_criteria} key criteria (in order of importance):

{criteria_text}

### Reward Scale

Assign a final reward of **1**, **0**, or **-1**:

- **1**: All of the above {num_criteria} key criteria are met.
- **0**: Most of the above {num_criteria} key criteria are met.
- **-1**: Most of the above {num_criteria} key criteria are NOT met or the agent does not call the tools at all.

## Output Format

Return your evaluation in the following JSON format:

```json
{{
    "reward": 0,
    "reasoning": "Brief explanation (1-2 sentences)"
}}
```

**Important**:
- `reward` must be exactly one of: -1, 0, 1
- Output valid JSON only

Generate the evaluation now.
"""
    return textwrap.dedent(template).strip()
