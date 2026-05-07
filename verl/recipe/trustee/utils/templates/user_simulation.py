import json
import textwrap


def get_user_simulation_prompt(
    history: list[dict],
    agent_message: str,
    user_query: str,
    tools: list[dict],
    user_intent: str = "",
    user_persona: str = "",
) -> str:
    """
    Generate a prompt for simulating user responses.
    [user simulator] input: history, tools, user_intent, user_persona -> user_reward, user_response
    """
    history_text = json.dumps(history, indent=2)
    tools_text = json.dumps(tools, indent=2)
    extra = ""
    if user_intent:
        extra += f"\n## User Intent (what the user ultimately wants)\n{user_intent}\n"
    if user_persona:
        extra += f"\n## User Persona\n{user_persona}\n"

    template = f"""
You are simulating a human user interacting with an assistant agent to accomplish a task.

## Original Task / First User Message
{user_query}
{extra}

## Available Tools (for agent use)
{tools_text}

## Interaction History
{history_text}

## Agent's Latest Message
{agent_message}

## Your Task

As the user, respond to the agent's latest message and evaluate its quality.

### Response Guidelines

**Be Human-Like**
- Use natural, conversational language
- Be concise and to the point (typically 1-3 sentences)
- Show realistic reactions and emotions when appropriate
- Don't repeat the exact task; use your own words

**Provide Information Appropriately**
- Only provide information when the agent asks for it
- Don't volunteer all details at once

**Know When to End**
- If the task is successfully completed, respond with an empty string ""
- If the agent has clearly failed and cannot help, respond with an empty string ""
- Only end when it's genuinely appropriate to stop the conversation

**Be Realistic**
- React naturally to agent's actions (confusion, satisfaction, frustration, anger, etc.)
- Ask for clarification if the agent's response is unclear
- Point out errors or issues if the agent makes mistakes

### Reward Guidelines

Assign exactly one of **-1**, **0**, or **1** based on how well the agent's message serves the user's intent and moves the task forward:

**1** — Significant progress; fully meets the user's intent
- Clearly addresses what the user asked or needed at this step
- Moves the task meaningfully toward completion (e.g. correct information, right tool use, useful next step)
- Response is on-topic, clear, and appropriate

**0** — Little progress or suboptimal
- Partly relevant or only marginally helpful
- Some progress but slow, indirect, or with unnecessary detours
- Suboptimal choices (e.g. weaker tool use or incomplete answer) but not wrong

**-1** — Wrong, unclear, no progress, or off-topic
- Wrong answer, off-topic, or fails to address the user's intent
- Unclear or confusing so the user cannot act on it
- No real progress (e.g. repetition, irrelevant actions, or counterproductive steps)

First judge progress and intent alignment, then assign the single value {-1, 0, 1}.

## Output Format

Return your response in the following JSON format:

```json
{{
    "response": "Your message as the user",
    "reward": 0
}}
```

**Important**:
- `response` must be a string; use an empty string "" if the conversation should end
- `reward` must be exactly one of: -1, 0, 1
- Output valid JSON only (no extra text or comments)

Generate the user simulation now.
"""
    
    return textwrap.dedent(template).strip()
