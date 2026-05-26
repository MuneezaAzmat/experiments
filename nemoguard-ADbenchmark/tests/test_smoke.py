"""Smoke test — requires Ollama running with qwen2.5:14b.

Run with: python3 -m pytest tests/test_smoke.py -v -s
Skip with: python3 -m pytest tests/ -v --ignore=tests/test_smoke.py
"""
import os
import pytest

import openai

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.llms.local_llm import LocalLLM
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor
from agentdojo.task_suite.load_suites import get_suite
from langchain_ollama import ChatOllama

from bridge.pipeline_element import RailCheckedToolsExecutor

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")

requires_ollama = pytest.mark.skipif(
    os.environ.get("SKIP_OLLAMA_TESTS", "0") == "1",
    reason="SKIP_OLLAMA_TESTS=1",
)


def _check_ollama():
    try:
        client = openai.OpenAI(api_key="EMPTY", base_url=f"{OLLAMA_BASE_URL}/v1")
        client.models.list()
        return True
    except Exception:
        return False


@requires_ollama
@pytest.mark.skipif(not _check_ollama(), reason="Ollama not running")
def test_baseline_single_task():
    suite = get_suite("v1.2.2", "workspace")
    client = openai.OpenAI(api_key="EMPTY", base_url=f"{OLLAMA_BASE_URL}/v1")
    llm = LocalLLM(client=client, model=OLLAMA_MODEL, temperature=0)

    pipeline = AgentPipeline([
        SystemMessage("You are a helpful workspace assistant."),
        InitQuery(),
        llm,
        ToolsExecutionLoop([ToolsExecutor(), llm]),
    ])

    user_task = suite.get_user_task_by_id("user_task_0")
    utility, security = suite.run_task_with_pipeline(
        agent_pipeline=pipeline,
        user_task=user_task,
        injection_task=None,
        injections={},
    )

    # We don't assert pass/fail — just that it runs without crashing
    assert isinstance(utility, bool)
    assert isinstance(security, bool)


@requires_ollama
@pytest.mark.skipif(not _check_ollama(), reason="Ollama not running")
def test_execution_rails_single_task():
    suite = get_suite("v1.2.2", "workspace")
    client = openai.OpenAI(api_key="EMPTY", base_url=f"{OLLAMA_BASE_URL}/v1")
    llm = LocalLLM(client=client, model=OLLAMA_MODEL, temperature=0)
    judge = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

    pipeline = AgentPipeline([
        SystemMessage("You are a helpful workspace assistant."),
        InitQuery(),
        llm,
        ToolsExecutionLoop([
            RailCheckedToolsExecutor(judge, check_tool_calls=True, check_tool_results=True),
            llm,
        ]),
    ])

    user_task = suite.get_user_task_by_id("user_task_0")
    utility, security = suite.run_task_with_pipeline(
        agent_pipeline=pipeline,
        user_task=user_task,
        injection_task=None,
        injections={},
    )

    assert isinstance(utility, bool)
    assert isinstance(security, bool)
