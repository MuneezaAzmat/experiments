from uuid import uuid4

from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    ChatSystemMessage,
    ChatToolResultMessage,
    ChatUserMessage,
    text_content_block_from_string,
)
from agentdojo.functions_runtime import FunctionCall


def build_agentdojo_messages(
    system_message: str,
    user_query: str,
    call_log: list[dict],
    final_response: str,
) -> list[ChatMessage]:
    """Build an AgentDojo-compatible message list from NeMo's execution trace.

    Args:
        system_message: The system prompt used.
        user_query: The user's original query.
        call_log: List of dicts with function, args, result, error from bridged actions.
        final_response: The final text response from NeMo.

    Returns:
        List of ChatMessage in the order AgentDojo expects.
    """
    messages: list[ChatMessage] = []

    messages.append(ChatSystemMessage(
        role="system",
        content=[text_content_block_from_string(system_message)],
    ))

    messages.append(ChatUserMessage(
        role="user",
        content=[text_content_block_from_string(user_query)],
    ))

    for entry in call_log:
        call_id = str(uuid4())
        func_call = FunctionCall(
            function=entry["function"],
            args=dict(entry["args"]),
            id=call_id,
        )

        messages.append(ChatAssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[func_call],
        ))

        result_content = entry["result"] if entry["result"] else f"Error: {entry['error']}"
        messages.append(ChatToolResultMessage(
            role="tool",
            content=[text_content_block_from_string(result_content)],
            tool_call_id=call_id,
            tool_call=func_call,
            error=entry["error"],
        ))

    messages.append(ChatAssistantMessage(
        role="assistant",
        content=[text_content_block_from_string(final_response)],
        tool_calls=None,
    ))

    return messages
