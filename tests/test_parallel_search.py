"""Exercise the real Agents SDK and MCP transport with deterministic responses."""

from __future__ import annotations

import ast
import asyncio
import json
from collections import defaultdict
from types import SimpleNamespace

import httpx
import pytest
from agents import Model, ModelResponse, Usage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

import docetl
from docetl.operations import get_operation
from docetl.operations.utils.openai_agents_runner import _run_openai_agent_async
from docetl.operations.utils.parallel_search import (
    PARALLEL_MCP_URL,
    ParallelSearchClient,
)


@pytest.fixture
def mcp_transport(monkeypatch):
    """Keep the production HTTP/MCP stack, replacing only network I/O."""
    import agents.mcp.server as sdk_server

    requests = []
    state = {"error": False, "malformed": False, "delay": 0.0, "cancelled": False}

    async def respond(request):
        message = json.loads(request.content)
        requests.append((request, message))
        method = message["method"]
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "initialize":
            result = {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "test-search", "version": "1"},
            }
        elif method == "tools/list":
            tools = ParallelSearchClient(session_id="unused").tools()
            result = {
                "tools": [
                    {
                        "name": name,
                        "description": tool.description,
                        "inputSchema": tool.parameters,
                    }
                    for name, tool in zip(["web_search", "web_fetch"], tools)
                ]
            }
        elif method == "tools/call":
            try:
                await asyncio.sleep(state["delay"])
            except asyncio.CancelledError:
                state["cancelled"] = True
                raise
            payload = {
                "results": [
                    {
                        "title": "DocETL",
                        "url": "https://docetl.org/",
                        "excerpts": ["Declarative data processing."],
                    }
                ]
            }
            result = {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "quota reached" if state["error"] else json.dumps(payload)
                        ),
                    }
                ],
                "isError": state["error"],
            }
            if state["malformed"]:
                result["content"][0]["text"] = "not JSON"
        else:
            raise AssertionError(method)
        return httpx.Response(
            200, json={"jsonrpc": "2.0", "id": message["id"], "result": result}
        )

    def factory(**kwargs):
        return httpx.AsyncClient(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(sdk_server, "_create_default_streamable_http_client", factory)
    # Deliberately present: anonymous requests must not copy ambient secrets.
    monkeypatch.setenv("PARALLEL_API_KEY", "must-not-be-sent")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return requests, state


class ScriptedSearchModel(Model):
    """A scripted model; Runner still performs real tool dispatch and consumption."""

    def __init__(self, *, error_expected=False, retry=False):
        self.turn = 0
        self.error_expected = error_expected
        self.retry = retry

    async def get_response(self, *args, **kwargs):
        inputs = args[1] if len(args) > 1 else kwargs["input"]
        tools = args[3] if len(args) > 3 else kwargs["tools"]
        assert {tool.name for tool in tools} == {"parallel_search", "parallel_fetch"}
        if self.turn:
            outputs = [x for x in inputs if x.get("type") == "function_call_output"]
            assert outputs
            output = outputs[-1]["output"]
            # The SDK renders Python function results as text, as it does for
            # existing DocETL tools. The scripted model reads that actual text.
            if self.error_expected:
                assert "error" in str(output).lower() or "timed out" in str(output)
            else:
                parsed = ast.literal_eval(output) if isinstance(output, str) else output
                assert parsed["results"][0]["url"] == "https://docetl.org/"
        self.turn += 1
        sequence = ["parallel_search", "parallel_fetch", "parallel_search"]
        if self.retry:
            sequence = ["parallel_search", "parallel_search"]
        if self.turn <= len(sequence):
            name = sequence[self.turn - 1]
            arguments = (
                {"objective": "Find DocETL", "search_queries": ["DocETL project"]}
                if name == "parallel_search"
                else {"urls": ["https://docetl.org/"], "objective": "Read DocETL"}
            )
            output = ResponseFunctionToolCall(
                type="function_call",
                name=name,
                arguments=json.dumps(arguments),
                call_id=f"call_{self.turn}",
            )
        else:
            output = ResponseOutputMessage(
                id="answer",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    ResponseOutputText(
                        type="output_text",
                        text=json.dumps({"answer": "https://docetl.org/"}),
                        annotations=[],
                    )
                ],
            )
        return ModelResponse(output=[output], usage=Usage(), response_id=None)

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError


async def run_agent(marker, *, scripted=None, **controls):
    config = docetl.Agent(
        tools=[marker],
        run_config={
            "model": scripted or ScriptedSearchModel(),
            "tracing_disabled": True,
        },
        **controls,
    )
    return await _run_openai_agent_async(
        runner=SimpleNamespace(total_token_usage=defaultdict(lambda: defaultdict(int))),
        model="test/scripted",
        op_type="map",
        prompt="Research DocETL and return its URL.",
        output_schema={"answer": "str"},
        agent=config,
        system_prompt="",
        scratchpad=None,
        litellm_completion_kwargs={},
    )


def tool_calls(requests):
    return [m["params"] for _, m in requests if m["method"] == "tools/call"]


def test_agent_dispatch_consumes_search_fetch_and_follow_up(mcp_transport):
    requests, _ = mcp_transport
    output, _ = asyncio.run(run_agent(docetl.tools.parallel_search()))
    assert output == {"answer": "https://docetl.org/"}
    calls = tool_calls(requests)
    assert [c["name"] for c in calls] == ["web_search", "web_fetch", "web_search"]
    assert len({c["arguments"]["session_id"] for c in calls}) == 1
    assert all(len(c["arguments"]["session_id"]) == 36 for c in calls)
    assert {m["method"] for _, m in requests} >= {
        "initialize",
        "tools/list",
        "tools/call",
    }
    for request, _ in requests:
        assert str(request.url) == PARALLEL_MCP_URL
        assert request.headers["user-agent"].startswith("DocETL/")
        assert "github.com/ucbepic/docetl" in request.headers["user-agent"]
        assert "authorization" not in request.headers


def test_reused_configuration_isolates_concurrent_and_new_runs(mcp_transport):
    requests, _ = mcp_transport
    marker = docetl.tools.parallel_search()

    async def run():
        await asyncio.gather(run_agent(marker), run_agent(marker))
        await run_agent(marker)

    asyncio.run(run())
    sessions = defaultdict(list)
    for call in tool_calls(requests):
        sessions[call["arguments"]["session_id"]].append(call["name"])
    assert len(sessions) == 3
    assert all(
        names == ["web_search", "web_fetch", "web_search"]
        for names in sessions.values()
    )


def test_quota_and_explicit_follow_up_retry_keep_session(mcp_transport):
    requests, state = mcp_transport
    state["error"] = True
    asyncio.run(
        run_agent(
            docetl.tools.parallel_search(),
            scripted=ScriptedSearchModel(error_expected=True, retry=True),
        )
    )
    calls = tool_calls(requests)
    assert len(calls) == 2
    assert calls[0]["arguments"]["session_id"] == calls[1]["arguments"]["session_id"]


@pytest.mark.parametrize("failure", ["error", "malformed"])
def test_agent_error_policy_propagates_failures(mcp_transport, failure):
    _, state = mcp_transport
    state[failure] = True
    with pytest.raises(Exception, match="quota reached|invalid JSON"):
        asyncio.run(
            run_agent(docetl.tools.parallel_search(), continue_on_tool_error=False)
        )


def test_tool_budget_is_preserved(mcp_transport):
    requests, _ = mcp_transport
    with pytest.raises(Exception, match="max_tool_calls=1"):
        asyncio.run(run_agent(docetl.tools.parallel_search(), max_tool_calls=1))
    assert len(tool_calls(requests)) == 1


def test_timeout_and_cancellation_close_transport(mcp_transport):
    requests, state = mcp_transport
    state["delay"] = 10

    async def run():
        client = ParallelSearchClient(session_id="timeout-test", timeout=0.1)
        with pytest.raises(TimeoutError):
            await client.search("test", ["test"])
        assert state["cancelled"]
        state["cancelled"] = False
        task = asyncio.create_task(
            ParallelSearchClient(session_id="cancel-test").search("test", ["test"])
        )
        while len(tool_calls(requests)) < 2:
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert state["cancelled"]

    asyncio.run(run())


@pytest.mark.parametrize(
    "url", ["file:///tmp/page", "javascript:alert(1)", "/relative", "https://"]
)
def test_fetch_rejects_non_http_urls_before_connecting(mcp_transport, url):
    requests, _ = mcp_transport
    with pytest.raises(ValueError, match="HTTP or HTTPS"):
        asyncio.run(ParallelSearchClient(session_id="test").fetch([url], "Read"))
    assert not requests


def test_search_operation_dispatch_and_browser_default(mcp_transport, monkeypatch):
    requests, _ = mcp_transport
    import docetl.operations.web_search as module

    operation = get_operation("web_search")
    runner = SimpleNamespace(config={})
    config = {
        "name": "research",
        "type": "web_search",
        "query_template": "{{ input.topic }}",
        "output_field": "sources",
        "max_results": 1,
    }
    browser_calls = []
    monkeypatch.setattr(
        module, "_search_brave", lambda q, n: browser_calls.append((q, n)) or []
    )
    rows = [{"topic": "DocETL"}, {"topic": ""}]
    output, _ = operation(runner, config, "unused", 1).execute(rows)
    assert browser_calls == [("DocETL", 1)]
    assert not requests
    assert output[0]["sources"] == []
    output, _ = operation(
        runner, {**config, "search_engine": "parallel"}, "unused", 1
    ).execute(rows)
    assert output[0]["sources"] == [
        {
            "title": "DocETL",
            "description": "Declarative data processing.",
            "url": "https://docetl.org/",
        }
    ]
    assert output[1]["sources"] == []
    assert "sources" not in rows[0]
    assert len(tool_calls(requests)) == 1


def test_empty_agent_tools_stay_disabled(mcp_transport):
    from docetl.operations.utils.openai_agents_runner import _build_sdk_tools

    requests, _ = mcp_transport
    assert (
        _build_sdk_tools(
            agent=docetl.Agent(),
            counter={"count": 0},
            model="unused",
            op_type="map",
            system_prompt="",
            litellm_completion_kwargs={},
            openai_agent_cls=None,
            model_settings_cls=None,
            litellm_model_cls=None,
            openai_responses_model_cls=None,
            openai_client=None,
        )
        == []
    )
    assert not requests


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout_is_rejected(timeout):
    with pytest.raises(ValueError, match="positive and finite"):
        docetl.tools.parallel_search(timeout=timeout)


def test_agent_dispatch_timeout_and_cancellation(mcp_transport):
    requests, state = mcp_transport
    state["delay"] = 10

    async def run():
        task = asyncio.create_task(run_agent(docetl.tools.parallel_search()))
        while not tool_calls(requests):
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert state["cancelled"]
        await run_agent(
            docetl.tools.parallel_search(timeout=0.05),
            scripted=ScriptedSearchModel(error_expected=True),
        )

    asyncio.run(run())


def test_pipeline_consumes_parallel_operation_results(mcp_transport, tmp_path):
    requests, _ = mcp_transport
    from docetl.runner import DSLRunner

    config = {
        "default_model": "unused",
        "datasets": {"companies": {"type": "memory", "path": [{"company": "DocETL"}]}},
        "operations": [
            {
                "name": "research",
                "type": "web_search",
                "search_engine": "parallel",
                "query_template": "{{ input.company }}",
                "output_field": "sources",
                "max_results": 1,
            }
        ],
        "pipeline": {
            "steps": [
                {
                    "name": "research_step",
                    "input": "companies",
                    "operations": ["research"],
                }
            ],
            "output": {"type": "file", "path": str(tmp_path / "results.json")},
        },
    }
    runner = DSLRunner(config, max_threads=1)
    runner.load_run_save()
    rows = json.loads((tmp_path / "results.json").read_text())
    assert rows[0]["sources"][0]["url"] == "https://docetl.org/"
    assert len(tool_calls(requests)) == 1


def test_frame_agent_uses_parallel_tools_and_consumes_output(mcp_transport):
    requests, _ = mcp_transport
    rows = (
        docetl.from_list([{"company": "DocETL"}])
        .map(
            prompt="Research {{ input.company }} and return its URL.",
            output={"schema": {"answer": "str"}},
            model="test/scripted",
            agent=docetl.Agent(
                tools=[docetl.tools.parallel_search()],
                run_config={"model": ScriptedSearchModel(), "tracing_disabled": True},
            ),
        )
        .collect(max_threads=1)
    )
    assert rows[0]["answer"] == "https://docetl.org/"
    assert len(tool_calls(requests)) == 3


@pytest.mark.parametrize("engine", ["brave", "bing", "google", "duckduckgo"])
def test_explicit_browser_choice_is_preserved(mcp_transport, monkeypatch, engine):
    import docetl.operations.web_search as module

    requests, _ = mcp_transport
    calls = []
    monkeypatch.setattr(module, "_search_" + engine, lambda q, n: calls.append(q) or [])
    config = {
        "name": "research",
        "type": "web_search",
        "search_engine": engine,
        "query_template": "DocETL",
        "output_field": "sources",
    }
    get_operation("web_search")(
        SimpleNamespace(config={}), config, "unused", 1
    ).execute([{}])
    assert calls == ["DocETL"]
    assert not requests


def test_parallel_failure_never_falls_back_to_browser(mcp_transport, monkeypatch):
    import docetl.operations.web_search as module

    requests, state = mcp_transport
    state["error"] = True

    def unexpected_browser(*args):
        raise AssertionError("Browser fallback must not run")

    monkeypatch.setattr(module, "_search_brave", unexpected_browser)
    config = {
        "name": "research",
        "type": "web_search",
        "search_engine": "parallel",
        "query_template": "DocETL",
        "output_field": "sources",
    }
    rows, _ = get_operation("web_search")(
        SimpleNamespace(config={}), config, "unused", 1
    ).execute([{}])
    assert rows[0]["sources"] == [
        {"title": "ERROR", "description": "quota reached", "url": ""}
    ]
    assert len(tool_calls(requests)) == 1
