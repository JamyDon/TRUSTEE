import json
import textwrap


def get_tool_simulation_prompt(
    history: list[dict],
    tool_call: dict | None,
    tools: list[dict],
    user_intent: str = "",
) -> str:
    """
    Generate a prompt for simulating tool execution.
    [tool simulator] input: history, tools, user_intent -> tool_reward, tool_response
    """
    history_text = json.dumps(history, indent=2)
    intent_block = f"\n## User Intent (task context)\n{user_intent}\n" if user_intent else ""
    
    # Handle malformed tool call
    if not tool_call:
        tool_call_text = "null (malformed or unparseable tool call)"
    else:
        tool_call_text = json.dumps(tool_call, indent=2)
    
    tools_text = json.dumps(tools, indent=2)
    
    template = f"""
You are an environment simulator for tool calling agents. Your task is to simulate realistic tool execution results based on the interaction history and the agent's tool call.

## Interaction History
{history_text}

## Agent's Tool Call
{tool_call_text}

## Available Tools
{tools_text}
{intent_block}
## Your Task
Simulate the execution of the agent's tool call and return:
1. **Tool Result**: The simulated execution output (or an error message if the call is invalid)
2. **Tool Reward**: An integer reward in {{-1, 0, 1}} evaluating how well this tool call serves the user's intent

## Tool Result Guidelines

### Validation
First, validate the tool call:
- **Parseable**: Is the tool call properly formatted and parseable? If the tool call is null or malformed, return an error like "Error: Tool call must be properly formatted in the expected structure"
- **Tool exists**: Is the tool name in the available tools?
- **Required arguments**: Are all required parameters provided?
- **Argument types**: Do argument types match specifications?
- **Argument values**: Are values realistic and contextually appropriate?

If invalid, return an error message explaining the issue.

### Simulation (for valid calls)
If valid, simulate realistic execution output:
- **Contextual**: Use information from history to generate relevant results
- **Realistic**: Output should mirror real tool behavior with concrete, specific data
- **Consistent**: Maintain consistency with previous interactions
- **Plausible**: Avoid placeholders or generic values; use realistic naming

Examples of good simulation:
- Instead of "file1.txt", use "ProjectReport_2024_Q1.pdf"
- Instead of "user@example.com", use "sarah.chen@techcorp.com"
- Instead of "ID: 123", use "ID: a7c3f91b-2e8d"

Examples of error messages:
- "Error: Tool call is malformed or could not be parsed"
- "Error: Tool 'send_email' not found in available tools"
- "Error: Missing required parameter 'recipient' for tool 'send_email'"
- "Error: Parameter 'amount' must be a number, got string"

## Tool Reward Guidelines

Assign exactly one of **-1**, **0**, or **1**:

**1** — Tool call fully serves the user's intent
- Valid and well-formed (correct tool, required arguments present and correctly typed)
- Clearly advances the user's goal; optimal or near-optimal choice at this step
- Arguments are accurate and contextually appropriate

**0** — Tool call is valid but suboptimal
- Correctly formatted and executable (tool exists, arguments valid)
- Only partly aligned with the user's intent, or a reasonable but not best choice
- May be redundant, slightly off-purpose, or use weaker arguments than possible

**-1** — Invalid or intent-mismatched
- Invalid: malformed, unparseable, wrong tool, missing or wrong-typed arguments, or will fail
- Or valid but not addressing the user's intent: wrong task, irrelevant tool, or counterproductive for the goal

First decide validity and intent alignment, then assign the single value {-1, 0, 1}. Malformed or unparseable tool calls are always -1.

## Output Format

Return your response in the following JSON format:

```json
{{
    "result": "The simulated tool execution output or error message",
    "reward": 0
}}
```

**Important**:
- `result` must be a string (success output or error message; may be JSON inside the string if needed)
- `reward` must be exactly one of: -1, 0, 1
- Output valid JSON only (no extra text or comments)

Generate the simulation now.
"""
    
    return textwrap.dedent(template).strip()
