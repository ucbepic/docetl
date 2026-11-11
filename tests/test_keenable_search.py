"""Offline tests for the optional Keenable web_search engine."""

import json
from types import SimpleNamespace

import httpx
import pytest

import docetl.operations.web_search as module
from docetl.operations import get_operation

RESULTS = {
    "query": "DocETL",
    "results": [
        {
            "title": "DocETL",
            "url": "https://docetl.org/",
            "snippet": "Declarative data processing.",
        },
        {"title": "Second", "url": "https://example.org/", "snippet": "More."},
    ],
}


@pytest.fixture
def keenable_transport(monkeypatch):
    requests = []
    mode = {"value": "json"}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append((dict(request.headers), payload))
        if payload["method"] == "initialize":
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": payload["id"], "result": {}},
                headers={"mcp-session-id": "s1"},
            )
        result = {
            "content": [{"type": "text", "text": json.dumps(RESULTS)}],
            "isError": mode["value"] == "tool_error",
        }
        message = {"jsonrpc": "2.0", "id": payload["id"], "result": result}
        if mode["value"] == "sse":
            return httpx.Response(
                200,
                text=f"event: message\ndata: {json.dumps(message)}\n\n",
                headers={"content-type": "text/event-stream"},
            )
        if mode["value"] == "rpc_error":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "error": {"code": -32000, "message": "Rate limited"},
                },
            )
        return httpx.Response(200, json=message)

    real_client = httpx.Client

    def client(*args, **kwargs):
        return real_client(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    return requests, mode


def run(rows, **config):
    operation = get_operation("web_search")
    base = {
        "name": "research",
        "type": "web_search",
        "search_engine": "keenable",
        "query_template": "{{ input.topic }}",
        "output_field": "sources",
        "max_results": 1,
    }
    output, _ = operation(
        SimpleNamespace(config={}), {**base, **config}, "unused", 1
    ).execute(rows)
    return output


@pytest.mark.parametrize("mode", ["json", "sse"])
def test_keenable_engine_maps_results(keenable_transport, mode):
    requests, state = keenable_transport
    state["value"] = mode
    output = run([{"topic": "DocETL"}, {"topic": ""}])
    assert output[0]["sources"] == [
        {
            "title": "DocETL",
            "description": "Declarative data processing.",
            "url": "https://docetl.org/",
        }
    ]
    assert output[1]["sources"] == []
    methods = [payload["method"] for _, payload in requests]
    assert methods == ["initialize", "tools/call"]
    headers, call = requests[1]
    assert headers["mcp-session-id"] == "s1"
    assert "x-api-key" not in headers
    assert call["params"] == {
        "name": "search_web_pages",
        "arguments": {"query": "DocETL", "max_results": 1, "output_format": "json"},
    }


def test_keenable_optional_api_key(keenable_transport, monkeypatch):
    requests, _ = keenable_transport
    monkeypatch.setenv("KEENABLE_API_KEY", "test-key")
    run([{"topic": "DocETL"}])
    assert all(headers["x-api-key"] == "test-key" for headers, _ in requests)


@pytest.mark.parametrize("mode", ["tool_error", "rpc_error"])
def test_keenable_errors_are_reported_per_row(keenable_transport, mode):
    _, state = keenable_transport
    state["value"] = mode
    output = run([{"topic": "DocETL"}])
    assert output[0]["sources"][0]["title"] == "ERROR"


def test_default_engine_does_not_call_keenable(keenable_transport, monkeypatch):
    requests, _ = keenable_transport
    monkeypatch.setattr(module, "_search_brave", lambda q, n: [])
    run([{"topic": "DocETL"}], search_engine="brave")
    assert not requests
