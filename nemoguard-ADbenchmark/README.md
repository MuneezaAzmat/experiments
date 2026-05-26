# nemoguard-ADbenchmark

Benchmarking NeMo Guardrails defenses against [AgentDojo](https://github.com/ethz-spylab/agentdojo) prompt injection attacks on agentic tool-use pipelines.

## What This Does

Runs an agentic workspace assistant (email, calendar, cloud drive) through AgentDojo's injection attack suite under three conditions:

| Condition | Description |
|-----------|-------------|
| **baseline** | No guardrails — standard AgentDojo pipeline |
| **execution_rails** | LLM-as-judge checks on tool calls and tool results |
| **dialog_execution_rails** | Hardened system prompt + LLM-as-judge tool checks |

Results measure the **utility vs. security tradeoff**: how much task completion (utility) is sacrificed to block injection attacks (security).

### Sample Results

| Metric | baseline | execution_rails | dialog_execution_rails |
|--------|----------|-----------------|------------------------|
| Utility | 38% | 12% | 0% |
| Security | 12% | 0% | 0% |

## Prerequisites

- [Ollama](https://ollama.com) running locally
- `ollama pull qwen2.5:14b`
- Python 3.10+

## Setup

```bash
pip install -e ".[dev]"
```

## Usage

Run all conditions:
```bash
python src/runner.py
```

Run a specific condition:
```bash
python src/runner.py baseline
python src/runner.py execution_rails
```

Generate comparison report:
```bash
python src/report.py
```

## Project Structure

```
src/
  runner.py              # Benchmark orchestrator
  report.py              # Results aggregation
  bridge/                # Adapters between AgentDojo and guardrail checks
    pipeline_element.py  # LLM-as-judge tool execution wrapper
    message_converter.py # Message format conversion
    tool_adapter.py      # AgentDojo tool bridging
  configs/               # NeMo Guardrails config files per condition
    baseline/
    execution_rails/
    dialog_execution_rails/
tests/                   # Unit and smoke tests
library_additions/       # Extra guardrail implementations (Colang flows, YARA rules)
results/                 # Benchmark outputs
```
