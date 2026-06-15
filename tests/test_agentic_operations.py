from __future__ import annotations

import os

import pytest

import docetl
import docetl.operations.utils.api as api_mod
from docetl.display import format_query_plan
from docetl.operations.utils.openai_agents_runner import (
    AgentExecutionError,
    AgentResult,
    _build_agent_instructions,
    _build_sdk_tools,
    _build_subagent_instructions,
    _coerce_final_output,
)


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


def test_agent_instructions_require_json_object() -> None:
    instructions = _build_agent_instructions(
        "system",
        docetl.Agent(),
        {"answer": "str"},
    )
    assert "exactly one JSON object" in instructions
    assert "no prose, Markdown, or code fences" in instructions
    assert '"answer": "str"' in instructions


def test_subagent_instructions_require_json_when_schema_is_set() -> None:
    specialist = docetl.Agent()
    instructions = _build_subagent_instructions(
        "system",
        specialist.as_tool(
            name="summarize",
            description="Summarize text.",
            output_schema={"summary": "str"},
        ),
    )
    assert "exactly one JSON object" in instructions
    assert '"summary": "str"' in instructions


def test_agent_string_output_is_not_wrapped_for_single_field_schema() -> None:
    with pytest.raises(AgentExecutionError, match="Could not parse"):
        _coerce_final_output("plain text", {"answer": "str"})


def test_agent_tools_are_visible_in_query_plan() -> None:
    @docetl.tool
    def lookup_city(city: str) -> dict[str, str]:
        """Lookup city metadata."""
        return {"city": city}

    frame = docetl.from_list([{"city": "Berkeley"}]).map(
        prompt="Lookup {{ input.city }}",
        output={"schema": {"city": "str"}},
        agent=docetl.Agent(tools=[lookup_city]),
    )
    runner = frame._build_runner()
    _, plan = format_query_plan(
        runner.last_op_container,
        runner.op_container_map,
        default_model=runner.default_model,
    )
    assert "agent tools" in plan
    assert "lookup_city" in plan


def test_subagent_tool_is_visible_in_query_plan() -> None:
    specialist = docetl.Agent(instructions="Summarize company evidence.")
    manager = docetl.Agent(
        tools=[
            specialist.as_tool(
                name="summarize_evidence",
                description="Summarize evidence for one company.",
            )
        ]
    )
    frame = docetl.from_list([{"company": "NVIDIA"}]).map(
        prompt="Research {{ input.company }}",
        output={"schema": {"summary": "str"}},
        agent=manager,
    )
    runner = frame._build_runner()
    _, plan = format_query_plan(
        runner.last_op_container,
        runner.op_container_map,
        default_model=runner.default_model,
    )
    assert "agent tools" in plan
    assert "summarize_evidence" in plan


def test_agent_tools_are_visible_in_progress_state() -> None:
    @docetl.tool
    def lookup_city(city: str) -> dict[str, str]:
        """Lookup city metadata."""
        return {"city": city}

    frame = docetl.from_list([{"city": "Berkeley"}]).filter(
        prompt="Keep {{ input.city }}",
        output={"schema": {"keep": "bool"}},
        agent=docetl.Agent(tools=[lookup_city]),
    )
    runner = frame._build_runner()
    operation = runner.list_pipeline_operations()[0]
    assert operation[-1] == ["lookup_city"]


def test_agent_as_tool_builds_sdk_function_tool() -> None:
    from agents import Agent as OpenAIAgent
    from agents import ModelSettings
    from agents.extensions.models.litellm_model import LitellmModel

    @docetl.tool
    def extract_numbers(text: str) -> list[int]:
        """Extract numbers from text."""
        return [int(token) for token in text.split() if token.isdigit()]

    specialist = docetl.Agent(
        tools=[extract_numbers],
        max_turns=4,
        max_tool_calls=3,
        instructions="Extract numeric evidence.",
    )
    manager = docetl.Agent(
        tools=[
            specialist.as_tool(
                name="extract_numeric_evidence",
                description="Extract numeric evidence from a note.",
            )
        ]
    )
    tools = _build_sdk_tools(
        agent=manager,
        counter={"count": 0},
        model="gpt-4o-mini",
        op_type="map",
        system_prompt="",
        litellm_completion_kwargs={},
        openai_agent_cls=OpenAIAgent,
        model_settings_cls=ModelSettings,
        litellm_model_cls=LitellmModel,
    )
    assert len(tools) == 1
    assert tools[0].name == "extract_numeric_evidence"
    assert tools[0].description == "Extract numeric evidence from a note."


def test_bash_tool_helper_builds_hosted_shell_tool() -> None:
    shell_tool = docetl.tools.bash(network="disabled", memory_limit="1g")
    assert shell_tool.name == "bash"
    assert shell_tool.environment["type"] == "container_auto"
    assert shell_tool.environment["network_policy"] == {"type": "disabled"}
    assert shell_tool.environment["memory_limit"] == "1g"


def test_agentic_map_retries_invalid_schema(monkeypatch) -> None:
    calls = iter([AgentResult({"wrong": "shape"}, 0.01), AgentResult({"count": 2}, 0.01)])

    def fake_run_openai_agent(**kwargs):
        return next(calls)

    monkeypatch.setattr(api_mod, "run_openai_agent", fake_run_openai_agent)

    rows = (
        docetl.from_list([{"text": "one two"}])
        .map(
            prompt="Count words in {{ input.text }}",
            output={"schema": {"count": "int"}},
            model="azure/gpt-4o-mini",
            agent=docetl.Agent(),
            num_retries_on_validate_failure=1,
        )
        .collect(max_threads=1)
    )

    assert rows == [{"text": "one two", "count": 2}]


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


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="needs OPENAI_API_KEY for live OpenAI Agents SDK run",
)
def test_live_openai_agent_with_subagent_tool_and_schema() -> None:
    @docetl.tool
    def count_words(text: str) -> dict[str, int]:
        """Count words in text."""
        return {"word_count": len(text.split())}

    specialist = docetl.Agent(
        tools=[count_words],
        max_turns=4,
        max_tool_calls=2,
        instructions=(
            "Always call count_words with the supplied text. Return exactly one "
            "JSON object with word_count."
        ),
    )
    manager = docetl.Agent(
        tools=[
            specialist.as_tool(
                name="count_words_specialist",
                description="Count words in supplied text.",
                output_schema={"word_count": "int"},
            )
        ],
        max_turns=6,
        max_tool_calls=3,
        instructions=(
            "Call count_words_specialist exactly once, then return the DocETL "
            "schema fields using that result."
        ),
    )

    rows = (
        docetl.from_list([{"text": "alpha beta gamma"}])
        .map(
            prompt="Count words in this text: {{ input.text }}",
            output={"schema": {"word_count": "int", "method": "str"}},
            model="gpt-4o-mini",
            agent=manager,
            validate=["output['word_count'] == 3"],
            num_retries_on_validate_failure=1,
        )
        .collect(max_threads=1)
    )

    assert rows[0]["text"] == "alpha beta gamma"
    assert rows[0]["word_count"] == 3
    assert isinstance(rows[0]["method"], str)
    assert rows[0]["method"]
