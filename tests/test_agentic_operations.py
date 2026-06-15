from __future__ import annotations

import pytest

import docetl
import docetl.operations.utils.api as api_mod
from docetl.operations.utils.openai_agents_runner import AgentResult


def test_tool_decorator_builds_json_schema() -> None:
    @docetl.tool
    def search_notes(query: str, limit: int = 3) -> list[str]:
        """Search notes by query."""
        return [query][:limit]

    assert search_notes.name == "search_notes"
    assert search_notes.description == "Search notes by query."
    assert search_notes.parameters["properties"]["query"] == {"type": "string"}
    assert search_notes.parameters["properties"]["limit"] == {"type": "integer"}
    assert search_notes.parameters["required"] == ["query"]


def test_frame_to_yaml_rejects_agent() -> None:
    frame = docetl.from_list([{"text": "hello"}]).map(
        prompt="Summarize {{ input.text }}",
        output={"schema": {"summary": "str"}},
        agent=docetl.Agent(),
    )
    with pytest.raises(ValueError, match="Python-only"):
        frame.to_yaml()


def test_agentic_map_operation(monkeypatch) -> None:
    def fake_run_openai_agent(**kwargs):
        assert kwargs["model"] == "azure/gpt-4o-mini"
        return AgentResult({"word_count": 3}, 0.01)

    monkeypatch.setattr(api_mod, "run_openai_agent", fake_run_openai_agent)

    @docetl.tool
    def count_words(text: str) -> dict[str, int]:
        """Count words in text."""
        return {"word_count": len(text.split())}

    rows = (
        docetl.from_list([{"text": "one two three"}])
        .map(
            prompt="Count words in {{ input.text }}",
            output={"schema": {"word_count": "int"}},
            model="azure/gpt-4o-mini",
            agent=docetl.Agent(tools=[count_words]),
        )
        .collect(max_threads=1)
    )

    assert rows == [{"text": "one two three", "word_count": 3}]


def test_agentic_filter_operation(monkeypatch) -> None:
    calls = iter([AgentResult({"keep": True}, 0.01), AgentResult({"keep": False}, 0.01)])

    def fake_run_openai_agent(**kwargs):
        return next(calls)

    monkeypatch.setattr(api_mod, "run_openai_agent", fake_run_openai_agent)

    rows = (
        docetl.from_list([{"text": "urgent ticket"}, {"text": "normal note"}])
        .filter(
            prompt="Keep urgent records: {{ input.text }}",
            output={"schema": {"keep": "bool"}},
            model="azure/gpt-4o-mini",
            agent=docetl.Agent(),
        )
        .collect(max_threads=1)
    )

    assert rows == [{"text": "urgent ticket"}]


def test_filter_rejects_agent_with_cascade() -> None:
    frame = docetl.from_list([{"text": "urgent"}]).filter(
        prompt="Keep urgent records: {{ input.text }}",
        output={"schema": {"keep": "bool"}},
        agent=docetl.Agent(),
        cascade={"proxy_model": "gpt-4o-mini", "target": 0.9, "label_budget": 1},
    )
    with pytest.raises(ValueError, match="agent cannot yet be combined with cascade"):
        frame.collect(max_threads=1)


def test_agentic_reduce_operation(monkeypatch) -> None:
    def fake_run_openai_agent(**kwargs):
        assert kwargs["op_type"] == "reduce"
        return AgentResult({"total": 7}, 0.01)

    monkeypatch.setattr(api_mod, "run_openai_agent", fake_run_openai_agent)

    rows = (
        docetl.from_list([{"group": "a", "value": 2}, {"group": "a", "value": 5}])
        .reduce(
            reduce_key="group",
            prompt="Sum values in {{ inputs }}",
            output={"schema": {"total": "int"}},
            model="azure/gpt-4o-mini",
            agent=docetl.Agent(),
        )
        .collect(max_threads=1)
    )

    assert rows[0]["group"] == "a"
    assert rows[0]["total"] == 7
