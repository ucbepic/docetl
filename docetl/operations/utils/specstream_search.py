"""SpecStream Search API and Extract API client shared by agent tools and web-search operations."""

from __future__ import annotations

import os
from importlib.metadata import version
from typing import Any
from urllib.parse import urlsplit

import httpx

from docetl.agents import Tool

SPECSTREAM_API_URL = "https://api.specstream.ai/v1"


class SpecStreamSearchError(RuntimeError):
    """A failed SpecStream request or an invalid SpecStream response."""


def _create_http_client(**kwargs: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(**kwargs)


class SpecStreamSearchClient:
    """Search and page reading through the SpecStream HTTP API.

    The API key comes from ``api_key`` or the ``SPECSTREAM_API_KEY`` environment
    variable and is read when a request is made. Each request opens and closes
    its own HTTP client, so cancellation closes the connection it opened.
    """

    def __init__(self, *, api_key: str | None = None, timeout: float = 30.0):
        self._api_key = api_key
        self.timeout = timeout

    async def search(self, query: str, num: int = 10) -> dict[str, Any]:
        if not 1 <= num <= 100:
            raise ValueError("SpecStream search requires num between 1 and 100")
        payload = await self._request(
            "GET",
            "/web-search",
            params={"q": query, "num": num, "speed": "fast"},
        )
        sources = payload.get("sources")
        if not isinstance(sources, list):
            raise SpecStreamSearchError("SpecStream returned no sources list")
        return {
            "results": [
                {
                    "title": hit.get("title") or "",
                    "url": hit["url"],
                    "snippet": hit.get("snippet") or "",
                    "published": hit.get("published"),
                }
                for hit in sources
            ]
        }

    async def fetch(self, urls: list[str]) -> dict[str, Any]:
        if not 1 <= len(urls) <= 20:
            raise ValueError("SpecStream fetch requires between 1 and 20 URLs")
        for url in urls:
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError(
                    "SpecStream fetch requires absolute HTTP or HTTPS URLs"
                )
        payload = await self._request(
            "POST", "/extract", json={"urls": urls, "format": "markdown"}
        )
        pages = payload.get("results")
        if not isinstance(pages, list):
            raise SpecStreamSearchError("SpecStream returned no results list")
        return {
            "results": [
                {
                    "url": page.get("url"),
                    "status": page.get("status"),
                    "title": page.get("title"),
                    "text": page.get("text") or "",
                    "published": page.get("published"),
                }
                for page in pages
            ]
        }

    def _headers(self) -> dict[str, str]:
        api_key = self._api_key or os.environ.get("SPECSTREAM_API_KEY")
        if not api_key:
            raise SpecStreamSearchError(
                "SpecStream requires an API key; set SPECSTREAM_API_KEY"
            )
        return {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": f"DocETL/{version('docetl')} "
            "(+https://github.com/ucbepic/docetl)",
        }

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        headers = self._headers()
        async with _create_http_client(timeout=self.timeout) as client:
            response = await client.request(
                method, SPECSTREAM_API_URL + path, headers=headers, **kwargs
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise SpecStreamSearchError(
                f"SpecStream returned invalid JSON (HTTP {response.status_code})"
            ) from exc
        if response.status_code >= 400:
            error = body.get("error") if isinstance(body, dict) else None
            if isinstance(error, dict) and error.get("message"):
                raise SpecStreamSearchError(
                    f"SpecStream {error.get('code', response.status_code)}: "
                    f"{error['message']}"
                )
            raise SpecStreamSearchError(
                f"SpecStream request failed (HTTP {response.status_code})"
            )
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, dict):
            raise SpecStreamSearchError("SpecStream returned a non-object result")
        return data

    def tools(self) -> list[Tool]:
        """Expose search and page reading to the model with strict arguments."""
        return [
            Tool(
                function=self.search,
                name="specstream_search",
                description=(
                    "Search the web. Each result has a source URL, title, snippet "
                    "and publish date when known."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "minLength": 1},
                        "num": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                    "required": ["query", "num"],
                    "additionalProperties": False,
                },
                timeout=self.timeout,
            ),
            Tool(
                function=self.fetch,
                name="specstream_fetch",
                description="Read HTTP or HTTPS URLs and return each page's text.",
                parameters={
                    "type": "object",
                    "properties": {
                        "urls": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 20,
                        },
                    },
                    "required": ["urls"],
                    "additionalProperties": False,
                },
                timeout=self.timeout,
            ),
        ]
