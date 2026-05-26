import asyncio

from agentdojo.functions_runtime import FunctionsRuntime
from pydantic import BaseModel

from bridge.tool_adapter import bridge_tools, create_langchain_tool_schemas


class FakeEnv(BaseModel):
    counter: int = 0


def increment(counter: int, amount: int) -> str:
    """Increment a counter.

    :param counter: The current counter value.
    :param amount: The amount to increment by.
    """
    return f"New value: {counter + amount}"


def test_bridge_tools_creates_actions():
    runtime = FunctionsRuntime()
    runtime.register_function(increment)
    env = FakeEnv()

    actions, call_log = bridge_tools(runtime, env)

    assert "increment" in actions
    assert callable(actions["increment"])


def test_bridged_action_delegates_to_runtime():
    runtime = FunctionsRuntime()
    runtime.register_function(increment)
    env = FakeEnv()

    actions, call_log = bridge_tools(runtime, env)
    result = asyncio.run(actions["increment"](counter=5, amount=3))

    assert "New value: 8" in str(result)


def test_bridged_action_logs_calls():
    runtime = FunctionsRuntime()
    runtime.register_function(increment)
    env = FakeEnv()

    actions, call_log = bridge_tools(runtime, env)
    asyncio.run(actions["increment"](counter=5, amount=3))

    assert len(call_log) == 1
    assert call_log[0]["function"] == "increment"
    assert call_log[0]["args"] == {"counter": 5, "amount": 3}
    assert "New value: 8" in str(call_log[0]["result"])


def test_bridged_action_handles_errors():
    def bad_func(x: int) -> str:
        """A function that might fail.

        :param x: Input value.
        """
        raise ValueError("boom")

    runtime = FunctionsRuntime()
    runtime.register_function(bad_func)
    env = FakeEnv()

    actions, call_log = bridge_tools(runtime, env)
    result = asyncio.run(actions["bad_func"](x=1))

    assert "Error" in str(result)


def test_create_langchain_tool_schemas():
    runtime = FunctionsRuntime()
    runtime.register_function(increment)

    schemas = create_langchain_tool_schemas(runtime)

    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "increment"
    assert "description" in schemas[0]["function"]
    assert "parameters" in schemas[0]["function"]
