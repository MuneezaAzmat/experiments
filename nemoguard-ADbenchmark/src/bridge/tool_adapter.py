from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str


def bridge_tools(
    runtime: FunctionsRuntime,
    env,
) -> tuple[dict[str, callable], list[dict]]:
    """Bridge AgentDojo tools to NeMo action functions.

    Returns:
        Tuple of (actions_dict, call_log).
        actions_dict maps action names to async callables.
        call_log is a list that gets appended to on each call.
    """
    call_log: list[dict] = []
    actions: dict[str, callable] = {}

    for func_name, func in runtime.functions.items():
        param_names = set(func.parameters.model_fields.keys())

        async def _action(
            _fn=func_name,
            _params=param_names,
            **kwargs,
        ):
            tool_kwargs = {k: v for k, v in kwargs.items() if k in _params}
            result, error = runtime.run_function(env, _fn, tool_kwargs)

            if error:
                call_log.append({
                    "function": _fn,
                    "args": tool_kwargs,
                    "result": None,
                    "error": error,
                })
                return f"Error: {error}"

            result_str = tool_result_to_str(result)
            call_log.append({
                "function": _fn,
                "args": tool_kwargs,
                "result": result_str,
                "error": None,
            })
            return result_str

        _action.__name__ = func_name
        _action.__qualname__ = func_name
        _action.__doc__ = func.full_docstring
        actions[func_name] = _action

    return actions, call_log


def create_langchain_tool_schemas(runtime: FunctionsRuntime) -> list[dict]:
    """Convert AgentDojo Function objects to OpenAI-format tool schemas for bind_tools()."""
    schemas = []
    for func_name, func in runtime.functions.items():
        json_schema = func.parameters.model_json_schema()
        # Remove pydantic metadata keys that aren't part of OpenAI schema
        json_schema.pop("title", None)

        schemas.append({
            "type": "function",
            "function": {
                "name": func_name,
                "description": func.description,
                "parameters": json_schema,
            },
        })
    return schemas
