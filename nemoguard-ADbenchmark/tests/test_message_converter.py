from agentdojo.types import (
    ChatAssistantMessage,
    ChatSystemMessage,
    ChatToolResultMessage,
    ChatUserMessage,
    TextContentBlock,
)
from agentdojo.functions_runtime import FunctionCall

from bridge.message_converter import (
    build_agentdojo_messages,
)


def test_build_messages_from_call_log_and_response():
    system_msg = "You are a helpful assistant."
    user_query = "Search my emails"
    call_log = [
        {
            "function": "search_emails",
            "args": {"query": "meeting"},
            "result": "Found 2 emails about meetings.",
            "error": None,
        },
    ]
    final_response = "I found 2 emails about meetings."

    messages = build_agentdojo_messages(system_msg, user_query, call_log, final_response)

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["tool_calls"] is not None
    assert messages[2]["tool_calls"][0].function == "search_emails"
    assert messages[3]["role"] == "tool"
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"][0]["content"] == final_response


def test_build_messages_no_tool_calls():
    messages = build_agentdojo_messages("sys", "hello", [], "Hi there!")

    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["tool_calls"] is None


def test_build_messages_multiple_tool_calls():
    call_log = [
        {"function": "search_emails", "args": {"query": "a"}, "result": "r1", "error": None},
        {"function": "send_email", "args": {"to": "b"}, "result": "r2", "error": None},
    ]
    messages = build_agentdojo_messages("sys", "q", call_log, "Done")

    # system, user, assistant(tool_call_1), tool_result_1, assistant(tool_call_2), tool_result_2, assistant(final)
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert len(assistant_msgs) == 3  # 2 with tool_calls + 1 final


def test_build_messages_with_error():
    call_log = [
        {"function": "bad_fn", "args": {"x": 1}, "result": None, "error": "ValueError: boom"},
    ]
    messages = build_agentdojo_messages("sys", "q", call_log, "Failed")

    tool_msg = [m for m in messages if m["role"] == "tool"][0]
    assert tool_msg["error"] == "ValueError: boom"
