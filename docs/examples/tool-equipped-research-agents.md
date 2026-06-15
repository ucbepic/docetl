# Equipping DocETL Agents with Tools

This tutorial shows how to use DocETL's Python API when map, filter, or reduce
operations need tools over multiple turns. We will build a market-research
pipeline:

1. A **map agent** researches each company with the OpenAI Agents SDK
   `WebSearchTool` and writes a compact evidence file.
2. A **reduce agent** groups companies by sector, reads those evidence files,
   and writes a sector brief.

This pattern is useful when the work cannot be done from the input row alone.
The model has to search for current facts, persist evidence, and then synthesize
across files.

!!! note

    Tool-equipped `docetl.Agent` configs are Python-only. YAML pipelines do not
    support callable tools or OpenAI Agents SDK tool objects.

## Install and configure

Install DocETL and set your model credentials:

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

!!! tip

    `WebSearchTool` is an OpenAI Agents SDK hosted tool. Hosted SDK tools depend
    on the selected model/provider. For non-OpenAI LiteLLM providers, use tools
    that provider supports, MCP tools, or Python tools wrapped with
    `@docetl.tool`.

## Full script

Save this as `tool_equipped_research_agents.py`.

```python
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from agents import WebSearchTool

import docetl


WORKSPACE = Path("agent_workspace")
EVIDENCE_DIR = WORKSPACE / "evidence"
REPORT_DIR = WORKSPACE / "reports"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower()).strip("-")
    return cleaned or hashlib.sha256(value.encode()).hexdigest()[:12]


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
    tools=[WebSearchTool(), write_json_file],
    max_turns=8,
    max_tool_calls=10,
    instructions=(
        "Use web search for current sources. Prefer primary sources when "
        "available. Save compact evidence with write_json_file, including "
        "source URLs and the specific signals you found."
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

        Use web search to find recent sources, then write_json_file to save
        compact evidence. Return the path and the most important signals.
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
        model="gpt-4o-mini",
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
        model="gpt-4o-mini",
        agent=reduce_agent,
    )
    .collect(max_threads=2)
)

print(json.dumps(rows, indent=2))
```

## What the tools do

The map operation's agent has two tools:

- `WebSearchTool()` finds current sources through the OpenAI Agents SDK.
- `write_json_file(...)` persists the evidence that reduce will consume.

The reduce operation's agent has file tools:

- `read_json_file(...)` loads every evidence file in the sector group.
- `write_markdown_file(...)` writes a human-readable sector brief.

## Why use tools here?

A plain map prompt can only transform the row it receives. In this workflow,
each row needs external actions:

- search queries may need refinement;
- current sources are not in the input dataset;
- evidence must be saved so the reduce step can audit what was used;
- the reduce step needs to read multiple files before writing a sector-level
  deliverable.

DocETL still owns the dataflow and output schemas. The agent simply gets tool
turns before it returns each map or reduce result.

## Adding a filter agent

You can add a filter agent before the map step when only some inputs deserve
research:

```python
@docetl.tool
def has_public_company_signal(company: str) -> bool:
    """Return whether a company should be included in market research."""
    return company.strip().lower() not in {"stealth startup", "unknown"}

filter_agent = docetl.Agent(
    tools=[has_public_company_signal],
    max_turns=3,
    max_tool_calls=2,
)

frame = docetl.from_list(companies).filter(
    name="filter_researchable_companies",
    prompt="Use the tool to decide whether to research {{ input.company }}.",
    output={"schema": {"keep": "bool"}},
    model="gpt-4o-mini",
    agent=filter_agent,
)
```

DocETL removes the `keep` field from rows that pass the filter, so downstream
map/reduce steps see the original row shape.

## Tuning tool budgets

Use small budgets first:

```python
docetl.Agent(tools=[WebSearchTool(), write_json_file], max_turns=4, max_tool_calls=6)
```

Increase `max_turns` when the agent needs more reasoning steps. Increase
`max_tool_calls` when each item legitimately requires more sources or file
operations. The limits are per operation call, so map limits apply per row,
filter limits apply per row, and reduce limits apply per group.

## Security notes

Python tools execute as trusted Python in your process. Keep file tools scoped
to a workspace directory and avoid exposing secrets through tool outputs. Hosted
OpenAI Agents SDK tools, such as web search or sandbox/code tools, follow the
SDK and provider behavior for the selected model backend.
