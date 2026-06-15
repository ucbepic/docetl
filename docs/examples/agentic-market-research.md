# Agentic Map-Reduce: Market Signal Briefs

This tutorial shows how to use DocETL's Python API when an operation needs tools
over multiple turns. We will build a market-research pipeline:

1. A **map agent** researches each company, searches the web, fetches pages, and
   writes a compact evidence file.
2. A **reduce agent** groups companies by sector, reads the evidence files, and
   writes a sector brief.

This pattern is useful when the work cannot be done from the input row alone.
The model has to decide which sources to inspect, persist evidence, and then
synthesize across files.

!!! note

    Agentic operations are Python-only. YAML pipelines do not support callable
    tools or `docetl.Agent` configs.

## Install and configure

Install DocETL with the agent dependencies and set your LiteLLM-compatible model
credentials:

```bash
pip install docetl
export OPENAI_API_KEY="..."
# or Azure OpenAI:
export AZURE_API_BASE="https://<resource>.openai.azure.com/"
export AZURE_API_KEY="..."
export AZURE_API_VERSION="2024-12-01-preview"
```

The model stays on each operation. For Azure OpenAI, use the LiteLLM deployment
name, for example `model="azure/gpt-4o-mini"`.

## Full script

Save this as `agentic_market_research.py`.

```python
from __future__ import annotations

import hashlib
import json
import os
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote_plus, urlparse

import requests

import docetl


WORKSPACE = Path("agent_workspace")
EVIDENCE_DIR = WORKSPACE / "evidence"
REPORT_DIR = WORKSPACE / "reports"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.parts.append(text)

    def text(self) -> str:
        return " ".join(self.parts)


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower()).strip("-")
    return cleaned or hashlib.sha256(value.encode()).hexdigest()[:12]


@docetl.tool
def search_web(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search the web and return result titles, URLs, and snippets."""
    if os.environ.get("TAVILY_API_KEY"):
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": os.environ["TAVILY_API_KEY"],
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
            },
            timeout=20,
        )
        response.raise_for_status()
        return [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
            }
            for item in response.json().get("results", [])
        ]
    response = requests.get(
        "https://duckduckgo.com/html/",
        params={"q": query},
        headers={"User-Agent": "DocETL tutorial"},
        timeout=20,
    )
    response.raise_for_status()
    matches = re.findall(r'class="result__a" href="([^"]+)".*?>(.*?)</a>', response.text)
    return [
        {"title": re.sub("<.*?>", "", title), "url": url, "snippet": ""}
        for url, title in matches[:max_results]
    ]


@docetl.tool
def fetch_url(url: str, max_chars: int = 6000) -> dict[str, str]:
    """Fetch a URL and return readable text for source inspection."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return {"url": url, "text": "Unsupported URL scheme."}
    response = requests.get(url, headers={"User-Agent": "DocETL tutorial"}, timeout=20)
    response.raise_for_status()
    parser = TextExtractor()
    parser.feed(response.text)
    return {"url": url, "text": parser.text()[:max_chars]}


@docetl.tool
def write_json_file(filename: str, payload: dict) -> str:
    """Write JSON evidence under agent_workspace/evidence and return the path."""
    path = EVIDENCE_DIR / f"{safe_name(filename)}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return str(path)


@docetl.tool
def read_json_file(path: str) -> dict:
    """Read a JSON evidence file written by the map agent."""
    resolved = Path(path)
    if not resolved.is_relative_to(EVIDENCE_DIR):
        raise ValueError("Can only read files from agent_workspace/evidence.")
    return json.loads(resolved.read_text())


@docetl.tool
def write_markdown_file(filename: str, content: str) -> str:
    """Write a markdown report under agent_workspace/reports and return the path."""
    path = REPORT_DIR / f"{safe_name(filename)}.md"
    path.write_text(content)
    return str(path)


companies = [
    {
        "company": "NVIDIA",
        "sector": "AI infrastructure",
        "question": "What recent product, partnership, or demand signals matter for AI infrastructure buyers?",
    },
    {
        "company": "AMD",
        "sector": "AI infrastructure",
        "question": "What recent product, partnership, or demand signals matter for AI infrastructure buyers?",
    },
    {
        "company": "Stripe",
        "sector": "payments",
        "question": "What recent product, partnership, or regulatory signals matter for enterprise payments teams?",
    },
    {
        "company": "Adyen",
        "sector": "payments",
        "question": "What recent product, partnership, or regulatory signals matter for enterprise payments teams?",
    },
]

map_agent = docetl.Agent(
    tools=[search_web, fetch_url, write_json_file],
    max_turns=8,
    max_tool_calls=10,
    instructions=(
        "Prefer primary sources. Fetch at least two useful URLs before writing "
        "the evidence file. Include short quotes or paraphrases with URLs."
    ),
)

reduce_agent = docetl.Agent(
    tools=[read_json_file, write_markdown_file],
    max_turns=8,
    max_tool_calls=12,
    instructions=(
        "Read every evidence file in the group. Write a concise markdown brief "
        "with cited URLs and clear implications for a business reader."
    ),
)

rows = (
    docetl.from_list(companies)
    .map(
        name="research_company",
        prompt="""
        Research {{ input.company }} for this question:
        {{ input.question }}

        Use search_web to find recent sources, fetch_url to inspect the best
        sources, then write_json_file to save evidence.
        """,
        output={
            "schema": {
                "company": "str",
                "sector": "str",
                "evidence_file": "str",
                "source_urls": "list[str]",
                "signals": "list[str]",
            }
        },
        model="azure/gpt-4o-mini",
        agent=map_agent,
    )
    .reduce(
        name="write_sector_brief",
        reduce_key="sector",
        prompt="""
        You are writing a sector brief for {{ inputs[0].sector }}.

        Read each evidence_file with read_json_file:
        {% for item in inputs %}
        - {{ item.company }}: {{ item.evidence_file }}
        {% endfor %}

        Then write the markdown brief with write_markdown_file.
        """,
        output={
            "schema": {
                "sector_summary": "str",
                "top_signals": "list[str]",
                "brief_file": "str",
                "source_urls": "list[str]",
            }
        },
        model="azure/gpt-4o-mini",
        agent=reduce_agent,
    )
    .collect(max_threads=2)
)

print(json.dumps(rows, indent=2))
```

## What the agents do

The map agent can use tools in whatever order the model decides:

- `search_web(...)` finds candidate sources.
- `fetch_url(...)` inspects source text.
- `write_json_file(...)` persists evidence for downstream use.

The reduce agent receives grouped rows for one sector. It reads the map evidence
files with `read_json_file(...)`, synthesizes across companies, and writes a
markdown memo with `write_markdown_file(...)`.

## Why not a normal map and reduce?

A normal map prompt can only transform the row it receives. In this workflow,
each row needs exploratory actions:

- search queries may need refinement;
- source pages may or may not contain relevant details;
- evidence must be saved so the reduce step can audit exactly what was used;
- the reduce step needs to read multiple files before writing a sector-level
  deliverable.

That is where agents fit: the operation still returns DocETL's structured
schema, but the model can take tool-using turns before producing it.

## Tuning tool budgets

Use small budgets first:

```python
docetl.Agent(tools=[search_web, fetch_url], max_turns=4, max_tool_calls=6)
```

Increase `max_turns` when the agent needs more reasoning steps. Increase
`max_tool_calls` when each item legitimately requires more sources or file
operations. The limits are per operation call, so map limits apply per row and
reduce limits apply per group.

## Security notes

Tools execute as trusted Python in your process. Keep file tools scoped to a
workspace directory, validate URLs, and avoid exposing secrets through tool
outputs. If you pass OpenAI Agents SDK sandbox/native tools into `docetl.Agent`,
support depends on the selected SDK backend and LiteLLM model/provider.
