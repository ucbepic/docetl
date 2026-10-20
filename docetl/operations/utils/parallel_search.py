"""Parallel Search MCP client shared by agent tools and web-search operations."""

from __future__ import annotations

import asyncio
import json
from importlib.metadata import version
from typing import Any
from urllib.parse import urlsplit

from docetl.agents import Tool

PARALLEL_MCP_URL = "https://search.parallel.ai/mcp"


class ParallelSearchError(RuntimeError):
    """A failed MCP request or an invalid Parallel response."""


class ParallelSearchClient:
    """A research context, independent of any individual MCP connection.

    Owners supply an opaque session ID at their run boundary. The configuration
    object and transport never own or mutate that identity. Connections are
    scoped to a call so cancellation closes them in the task that opened them.
    """

    def __init__(
        self,
        *,
        session_id: str,
        timeout: float = 30.0,
        pending_requests: set[asyncio.Task[Any]] | None = None,
    ):
        self.session_id = session_id
        self.timeout = timeout
        self.pending_requests = (
            pending_requests if pending_requests is not None else set()
        )

    async def search(self, objective: str, search_queries: list[str]) -> dict[str, Any]:
        return await self._call(
            "web_search", {"objective": objective, "search_queries": search_queries}
        )

    async def fetch(self, urls: list[str], objective: str) -> dict[str, Any]:
        if not 1 <= len(urls) <= 20:
            raise ValueError("Parallel fetch requires between 1 and 20 URLs")
        for url in urls:
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("Parallel fetch requires absolute HTTP or HTTPS URLs")
        if len(objective) > 200:
            raise ValueError("Parallel fetch objective must be at most 200 characters")
        return await self._call("web_fetch", {"urls": urls, "objective": objective})

    async def _call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        request = asyncio.create_task(self._request(name, arguments))
        self.pending_requests.add(request)
        try:
            return await asyncio.wait_for(request, self.timeout)
        finally:
            self.pending_requests.discard(request)

    async def _request(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        from agents.mcp import MCPServerStreamableHttp

        # Deliberately anonymous: do not read ambient provider credentials.
        async with MCPServerStreamableHttp(
            name="DocETL Parallel Search",
            params={
                "url": PARALLEL_MCP_URL,
                "headers": {
                    "User-Agent": f"DocETL/{version('docetl')} "
                    "(+https://github.com/ucbepic/docetl)"
                },
                "timeout": self.timeout,
                "sse_read_timeout": self.timeout,
            },
            client_session_timeout_seconds=self.timeout,
            max_retry_attempts=0,
        ) as server:
            available = await server.list_tools()
            if name not in {tool.name for tool in available}:
                raise ParallelSearchError(f"Parallel MCP does not expose {name}")
            result = await server.call_tool(
                name, {**arguments, "session_id": self.session_id}
            )
            text = "\n".join(
                block.text for block in result.content if block.type == "text"
            )
            if result.isError:
                raise ParallelSearchError(text or f"Parallel {name} failed")
            payload = result.structuredContent
            if payload is None:
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ParallelSearchError("Parallel returned invalid JSON") from exc
            if not isinstance(payload, dict):
                raise ParallelSearchError("Parallel returned a non-object result")
            return payload

    def tools(self) -> list[Tool]:
        """Expose a small, strict subset of the MCP tool arguments to the model."""
        return [
            Tool(
                function=self.search,
                name="parallel_search",
                description="Search the web for current information and source URLs.",
                parameters={
                    "type": "object",
                    "properties": {
                        "objective": {"type": "string"},
                        "search_queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                        },
                    },
                    "required": ["objective", "search_queries"],
                    "additionalProperties": False,
                },
                timeout=self.timeout,
            ),
            Tool(
                function=self.fetch,
                name="parallel_fetch",
                description="Read HTTP or HTTPS source URLs and return page excerpts.",
                parameters={
                    "type": "object",
                    "properties": {
                        "urls": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 20,
                        },
                        "objective": {"type": "string", "maxLength": 200},
                    },
                    "required": ["urls", "objective"],
                    "additionalProperties": False,
                },
                timeout=self.timeout,
            ),
        ]
