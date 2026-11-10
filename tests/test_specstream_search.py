"""Exercise the SpecStream tools through the real Agents SDK with mocked HTTP."""

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
from docetl.operations.utils.specstream_search import (
    SPECSTREAM_API_URL,
    SpecStreamSearchClient,
)

SEARCH_REPLY = {
    "data": {
        "request_id": "req_1",
        "query": "DocETL project",
        "kind": "fact",
        "columns": [],
        "answer": None,
        "rows": [],
        "sources": [
            {
                "url": "https://docetl.org/",
                "title": "DocETL",
                "site": "docetl.org",
                "snippet": "Declarative data processing.",
                "cited": False,
                "published": "2024-10-01",
            },
            {
                "url": "https://github.com/ucbepic/docetl",
                "title": "ucbepic/docetl",
                "site": "github.com",
                "snippet": "A system for LLM-powered data processing.",
                "cited": False,
            },
        ],
        "handoff": None,
        "took_ms": 812,
        "price_usd": 0.005,
    }
}

EXTRACT_REPLY = {
    "data": {
        "request_id": "req_2",
        "results": [
            {
                "url": "https://docetl.org/",
                "status": "ok",
                "title": "DocETL",
                "text": "# DocETL\n\nDeclarative data processing.",
                "published": None,
            }
        ],
        "took_ms": 950,
        "price_usd": 0.0008,
    }
}


@pytest.fixture
def specstream_http(monkeypatch):
    """Keep the production client code, replacing only network I/O."""
    import docetl.operations.utils.specstream_search as module

    requests = []
    state = {"status": 200, "error": None, "malformed": False}

    async def respond(request):
        requests.append(request)
        if state["malformed"]:
            return httpx.Response(200, text="not JSON")
        if state["error"]:
            return httpx.Response(
                state["status"],
                json={"error": {"code": state["error"], "message": "Balance is empty"}},
            )
        if request.url.path == "/v1/web-search":
            return httpx.Response(200, json=SEARCH_REPLY)
        if request.url.path == "/v1/extract":
            return httpx.Response(200, json=EXTRACT_REPLY)
        raise AssertionError(str(request.url))

    def factory(**kwargs):
        return httpx.AsyncClient(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(module, "_create_http_client", factory)
    monkeypatch.setenv("SPECSTREAM_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return requests, state


class ScriptedSearchModel(Model):
    """A scripted model; Runner still performs real tool dispatch and consumption."""

    def __init__(self, *, error_expected=False):
        self.turn = 0
        self.error_expected = error_expected

    async def get_response(self, *args, **kwargs):
        inputs = args[1] if len(args) > 1 else kwargs["input"]
        tools = args[3] if len(args) > 3 else kwargs["tools"]
        assert {tool.name for tool in tools} == {
            "specstream_search",
            "specstream_fetch",
        }
        if self.turn:
            outputs = [x for x in inputs if x.get("type") == "function_call_output"]
            assert outputs
            output = outputs[-1]["output"]
            if self.error_expected:
                assert "error" in str(output).lower()
            else:
                parsed = ast.literal_eval(output) if isinstance(output, str) else output
                assert parsed["results"][0]["url"] == "https://docetl.org/"
        self.turn += 1
        sequence = ["specstream_search", "specstream_fetch"]
        if self.turn <= len(sequence):
            name = sequence[self.turn - 1]
            arguments = (
                {"query": "DocETL project", "num": 5}
                if name == "specstream_search"
                else {"urls": ["https://docetl.org/"]}
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


def test_agent_dispatch_consumes_search_and_fetch(specstream_http):
    requests, _ = specstream_http
    output, _ = asyncio.run(run_agent(docetl.tools.specstream_search()))
    assert output == {"answer": "https://docetl.org/"}
    assert [r.url.path for r in requests] == ["/v1/web-search", "/v1/extract"]
    search, extract = requests
    assert str(search.url).startswith(SPECSTREAM_API_URL + "/web-search?")
    assert search.method == "GET"
    assert dict(search.url.params) == {
        "q": "DocETL project",
        "num": "5",
        "speed": "fast",
    }
    assert extract.method == "POST"
    assert json.loads(extract.content) == {
        "urls": ["https://docetl.org/"],
        "format": "markdown",
    }
    for request in requests:
        assert request.headers["authorization"] == "Bearer test-key"
        assert request.headers["user-agent"].startswith("DocETL/")


def test_search_result_shape(specstream_http):
    payload = asyncio.run(SpecStreamSearchClient().search("DocETL project"))
    assert payload == {
        "results": [
            {
                "title": "DocETL",
                "url": "https://docetl.org/",
                "snippet": "Declarative data processing.",
                "published": "2024-10-01",
            },
            {
                "title": "ucbepic/docetl",
                "url": "https://github.com/ucbepic/docetl",
                "snippet": "A system for LLM-powered data processing.",
                "published": None,
            },
        ]
    }


def test_missing_key_fails_before_connecting(specstream_http, monkeypatch):
    requests, _ = specstream_http
    monkeypatch.delenv("SPECSTREAM_API_KEY")
    with pytest.raises(Exception, match="SPECSTREAM_API_KEY"):
        asyncio.run(
            run_agent(docetl.tools.specstream_search(), continue_on_tool_error=False)
        )
    assert not requests


def test_tool_errors_are_returned_to_the_model(specstream_http):
    requests, state = specstream_http
    state["status"], state["error"] = 402, "insufficient_balance"
    output, _ = asyncio.run(
        run_agent(
            docetl.tools.specstream_search(),
            scripted=ScriptedSearchModel(error_expected=True),
            continue_on_tool_error=True,
        )
    )
    assert output == {"answer": "https://docetl.org/"}
    assert len(requests) == 2


@pytest.mark.parametrize("failure", ["error", "malformed"])
def test_agent_error_policy_propagates_failures(specstream_http, failure):
    _, state = specstream_http
    if failure == "error":
        state["status"], state["error"] = 402, "insufficient_balance"
    else:
        state["malformed"] = True
    with pytest.raises(Exception, match="Balance is empty|invalid JSON"):
        asyncio.run(
            run_agent(docetl.tools.specstream_search(), continue_on_tool_error=False)
        )


@pytest.mark.parametrize(
    "url", ["file:///tmp/page", "javascript:alert(1)", "/relative", "https://"]
)
def test_fetch_rejects_non_http_urls_before_connecting(specstream_http, url):
    requests, _ = specstream_http
    with pytest.raises(ValueError, match="HTTP or HTTPS"):
        asyncio.run(SpecStreamSearchClient().fetch([url]))
    assert not requests


@pytest.mark.parametrize("num", [0, 101])
def test_search_rejects_out_of_range_num(specstream_http, num):
    requests, _ = specstream_http
    with pytest.raises(ValueError, match="between 1 and 100"):
        asyncio.run(SpecStreamSearchClient().search("DocETL", num=num))
    assert not requests


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout_is_rejected(timeout):
    with pytest.raises(ValueError, match="positive and finite"):
        docetl.tools.specstream_search(timeout=timeout)


def test_search_operation_dispatch(specstream_http, monkeypatch):
    requests, _ = specstream_http
    import docetl.operations.web_search as module

    def unexpected_browser(*args):
        raise AssertionError("Browser search must not run")

    monkeypatch.setattr(module, "_search_brave", unexpected_browser)
    config = {
        "name": "research",
        "type": "web_search",
        "search_engine": "specstream",
        "query_template": "{{ input.topic }}",
        "output_field": "sources",
        "max_results": 1,
    }
    rows = [{"topic": "DocETL"}, {"topic": ""}]
    output, _ = get_operation("web_search")(
        SimpleNamespace(config={}), config, "unused", 1
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
    assert len(requests) == 1
    assert requests[0].url.params["num"] == "1"


def test_specstream_failure_never_falls_back_to_browser(specstream_http, monkeypatch):
    import docetl.operations.web_search as module

    requests, state = specstream_http
    state["status"], state["error"] = 402, "insufficient_balance"

    def unexpected_browser(*args):
        raise AssertionError("Browser fallback must not run")

    monkeypatch.setattr(module, "_search_brave", unexpected_browser)
    config = {
        "name": "research",
        "type": "web_search",
        "search_engine": "specstream",
        "query_template": "DocETL",
        "output_field": "sources",
    }
    rows, _ = get_operation("web_search")(
        SimpleNamespace(config={}), config, "unused", 1
    ).execute([{}])
    assert rows[0]["sources"] == [
        {
            "title": "ERROR",
            "description": "SpecStream insufficient_balance: Balance is empty",
            "url": "",
        }
    ]
    assert len(requests) == 1
