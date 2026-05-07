import json
import textwrap


def get_system_prompt(difficulty: int, format: str = "hermes") -> str:
    """
    Generate a system prompt for the agent.

    Args:
        difficulty: Prompt verbosity level (0=detailed, 1=medium, 2=brief, 3=default).
        format: Tool call format. Use "llama-3.1" to omit Hermes-style ``<tool_call>``
            format instructions (Llama-3.1's native chat template handles those itself).
    """
    if format == "llama-3.1":
        # Levels 0 and 1 reference <tool_call>/<think> format explicitly, which
        # conflicts with Llama-3.1's native python_tag format. Use format-neutral
        # variants instead; levels 2/3 are already format-neutral.
        if difficulty == 0:
            return get_system_prompt_detailed_llama()
        elif difficulty == 1:
            return get_system_prompt_medium_llama()

    if difficulty == 0:
        return get_system_prompt_detailed()
    elif difficulty == 1:
        return get_system_prompt_medium()
    elif difficulty == 2:
        return get_system_prompt_brief()
    elif difficulty == 3:
        return get_system_prompt_default()
    else:
        raise ValueError(f"Invalid difficulty: {difficulty}")


def get_system_prompt_default() -> str:
    """
    Generate a system prompt for the agent.
    """
    template = f"""
You are a helpful assistant.
"""
    return textwrap.dedent(template).strip()


def get_system_prompt_brief() -> str:
    """
    Generate a system prompt for the agent.
    """
    template = f"""
You are a helpful multi-turn dialogue assistant capable of leveraging tool calls to solve user tasks.
"""
    return textwrap.dedent(template).strip()


def get_system_prompt_medium() -> str:
    """
    Generate a system prompt for the agent.
    """
    template = f"""
You are a helpful multi-turn dialogue assistant capable of leveraging tool calls to solve user tasks.

# Important Notes
1. You must always include the `<think>` field to outline your reasoning. Provide either calling tools or responding to the user. The former leads to an interaction with the tool environment, while the latter switches back to the user interface.
2. You may invoke multiple tool calls simultaneously in the `<tool_call>` field. You will get the tool feedback after calling the tools. You may call tools for multiple turns if needed.
3. Refer to the previous dialogue records in the history, including the user's queries, previous tool calls, tool feedback, and your responses.
"""
    return textwrap.dedent(template).strip()


def get_system_prompt_detailed() -> str:
    """
    Generate a system prompt for the agent.
    """
    template = f"""
You are a helpful multi-turn dialogue assistant capable of leveraging tool calls to solve user tasks.

# Steps for Each Turn
1. **Think:** Recall relevant context and analyze the current user goal.
2. **Decide on Tool Usage:** If a tool is needed, specify the tool and its parameters.
3. **Respond Appropriately:** If a response is needed, generate one while maintaining consistency across user queries.

# Output Format
If you need to call tools, use the following format:
```plaintext
<think> Your thoughts and reasoning </think>
<tool_call> Tool call JSON object(s) </tool_call>
```

If you need to respond to the user, use the following format:
```plaintext
<think> Your thoughts and reasoning </think>
Your response to the user.
```

# Important Notes
1. You must always include the `<think>` field to outline your reasoning. Provide either calling tools or responding to the user. The former leads to an interaction with the tool environment, while the latter switches back to the user interface.
2. You may invoke multiple tool calls simultaneously in the `<tool_call>` field. You will get the tool feedback after calling the tools. You may call tools for multiple turns if needed.
3. Refer to the previous dialogue records in the history, including the user's queries, previous tool calls, tool feedback, and your responses.
"""
    return textwrap.dedent(template).strip()


def get_system_prompt_detailed_llama() -> str:
    """
    Detailed system prompt for Llama-3.1 (native tool-call format).
    Omits ``<tool_call>`` / ``<think>`` format instructions since Llama-3.1's
    chat template handles tool calls natively via ``<|python_tag|>``.
    """
    template = f"""
You are a helpful multi-turn dialogue assistant capable of leveraging tool calls to solve user tasks.

# Steps for Each Turn
1. **Think:** Recall relevant context and analyze the current user goal.
2. **Decide on Tool Usage:** If a tool is needed, call the appropriate tool with the correct parameters.
3. **Respond Appropriately:** If no tool is needed, generate a response while maintaining consistency across user queries.

# Important Notes
1. Use the available tools whenever they would help fulfill the user's request.
2. You may invoke multiple tool calls in a single turn. You will receive tool feedback before your next response.
3. Refer to the previous dialogue records in the history, including the user's queries, previous tool calls, tool feedback, and your responses.
"""
    return textwrap.dedent(template).strip()


def get_system_prompt_medium_llama() -> str:
    """
    Medium system prompt for Llama-3.1 (native tool-call format).
    Omits ``<tool_call>`` / ``<think>`` format instructions.
    """
    template = f"""
You are a helpful multi-turn dialogue assistant capable of leveraging tool calls to solve user tasks.

# Important Notes
1. Use the available tools whenever they would help fulfill the user's request. You will get tool feedback after calling the tools and may call tools for multiple turns if needed.
2. Refer to the previous dialogue records in the history, including the user's queries, previous tool calls, tool feedback, and your responses.
"""
    return textwrap.dedent(template).strip()