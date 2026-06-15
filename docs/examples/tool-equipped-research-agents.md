# Equipping DocETL Agents with Tools

This tutorial shows how to use DocETL's Python API when map, filter, or reduce
agents need tools over multiple turns. We will build a market-research pipeline:

1. A **map agent** researches each company with the OpenAI Agents SDK
   `WebSearchTool`, uses a persistent hosted bash sandbox for scratch work, and
   calls a specialist evidence subagent.
2. A **reduce agent** groups companies by sector, uses the same kind of sandbox
   tool for tabulation, calls a specialist memo editor, and returns a structured
   sector brief.

DocETL owns the dataflow and schemas. Tools help each operation do work before
returning its structured result.

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

    `WebSearchTool` and hosted `ShellTool` are OpenAI Agents SDK tools. Hosted
    SDK tools depend on the selected model/provider. For non-OpenAI LiteLLM
    providers, use tools that provider supports, MCP tools, or Python tools
    wrapped with `@docetl.tool`.

## Full script

Save this as `tool_equipped_research_agents.py`.

```python
from __future__ import annotations

import json

from agents import WebSearchTool

import docetl


# Create one hosted container and bind every bash tool to that container. Agents
# can share this filesystem because sandbox.bash() uses container_reference.
# Durable map-to-reduce state still flows through DocETL schemas.
sandbox = docetl.tools.Sandbox.create(
    name="docetl-market-research",
    network="disabled",
    memory_limit="1g",
)
bash = sandbox.bash()


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
        "company": "Broadcom",
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
    {
        "company": "Block",
        "sector": "payments",
        "question": "What recent product, partnership, or regulatory signals matter for enterprise payments teams?",
    },
    {
        "company": "Datadog",
        "sector": "observability",
        "question": "What recent product, platform, or demand signals matter for engineering leaders buying observability tools?",
    },
    {
        "company": "New Relic",
        "sector": "observability",
        "question": "What recent product, platform, or demand signals matter for engineering leaders buying observability tools?",
    },
    {
        "company": "Grafana Labs",
        "sector": "observability",
        "question": "What recent product, platform, or demand signals matter for engineering leaders buying observability tools?",
    },
    {
        "company": "CrowdStrike",
        "sector": "cybersecurity",
        "question": "What recent product, threat, or platform signals matter for security leaders buying cybersecurity tools?",
    },
    {
        "company": "Palo Alto Networks",
        "sector": "cybersecurity",
        "question": "What recent product, threat, or platform signals matter for security leaders buying cybersecurity tools?",
    },
    {
        "company": "Cloudflare",
        "sector": "cybersecurity",
        "question": "What recent product, threat, or platform signals matter for security leaders buying cybersecurity tools?",
    },
]

evidence_specialist = docetl.Agent(
    tools=[bash],
    instructions=(
        "You are an evidence specialist. Use bash for scratch notes or small "
        "tables when helpful. Return compact evidence: claims, source URLs, "
        "risks, and a 0-100 signal score. Do not rely on sandbox files as the "
        "only output; the manager must receive the evidence text."
    ),
    max_turns=4,
)

brief_editor = docetl.Agent(
    tools=[bash],
    instructions=(
        "You are a memo editor. Use bash for scratch formatting or table checks "
        "when helpful. Tighten the brief into decision-ready prose with concise "
        "bullets, cited URLs, and explicit business implications."
    ),
    max_turns=4,
)

map_agent = docetl.Agent(
    tools=[
        WebSearchTool(),
        bash,
        evidence_specialist.as_tool(
            name="extract_evidence",
            description="Turn raw search findings into compact cited evidence.",
        ),
    ],
    max_turns=8,
    max_tool_calls=10,
    instructions=(
        "Use web search for current sources. Use bash for scratch notes or quick "
        "tables if useful. Call extract_evidence before final output. Return the "
        "DocETL schema fields directly; do not put durable state only in files."
    ),
)

reduce_agent = docetl.Agent(
    tools=[
        bash,
        brief_editor.as_tool(
            name="edit_sector_brief",
            description="Edit a sector brief for clarity and decision usefulness.",
        ),
    ],
    max_turns=8,
    max_tool_calls=10,
    instructions=(
        "Use bash to tabulate the grouped inputs if helpful. Draft the brief, "
        "call edit_sector_brief, then return the DocETL schema fields directly."
    ),
)

rows = (
    docetl.from_list(companies)
    .map(
        name="research_company",
        prompt="""
        Research {{ input.company }} for this question:
        {{ input.question }}

        Find current sources, condense the evidence, and return structured
        evidence for this company.
        """,
        output={
            "schema": {
                "company": "str",
                "sector": "str",
                "evidence_summary": "str",
                "signal_score": "int",
                "risks": "list[str]",
                "source_urls": "list[str]",
            }
        },
        model="gpt-4o-mini",
        agent=map_agent,
        validate=[
            "0 <= output['signal_score'] <= 100",
            "len(output['source_urls']) >= 1",
        ],
    )
    .reduce(
        name="write_sector_brief",
        reduce_key="sector",
        prompt="""
        You are writing a sector brief for {{ inputs[0].sector }}.

        Company evidence:
        {% for item in inputs %}
        - {{ item.company }} (score {{ item.signal_score }}):
          {{ item.evidence_summary }}
          risks={{ item.risks }}
          sources={{ item.source_urls }}
        {% endfor %}

        Rank the companies, synthesize the shared signals, and return a concise
        markdown brief.
        """,
        output={
            "schema": {
                "sector_summary": "str",
                "ranked_companies": "list[str]",
                "top_signals": "list[str]",
                "brief_markdown": "str",
                "source_urls": "list[str]",
            }
        },
        model="gpt-4o-mini",
        agent=reduce_agent,
        validate=[
            "len(output['ranked_companies']) >= 1",
            "len(output['source_urls']) >= 1",
        ],
    )
    .collect(max_threads=2)
)

print(json.dumps(rows, indent=2))
```

## What the tools do

The map operation's manager agent has three tools:

- `WebSearchTool()` finds current sources through the OpenAI Agents SDK.
- `bash` is a hosted shell tool bound to one persistent container created with
  `docetl.tools.Sandbox.create(...)`.
- `extract_evidence(...)` is a specialist subagent exposed with
  `evidence_specialist.as_tool(...)`.

The reduce operation's manager agent has two tools:

- `bash` lets the agent create scratch CSV/Markdown or run small checks.
- `edit_sector_brief(...)` is a specialist subagent exposed with
  `brief_editor.as_tool(...)`.

Every manager and specialist receives `bash`, which is bound to the same hosted
container id. They can read and write the same sandbox filesystem. Still pass
durable data between DocETL operations through declared output schemas, not only
through hidden files, so checkpointing, validation, and downstream operations
remain auditable.

## Why use tools here?

A plain map prompt can only transform the row it receives. In this workflow,
each row benefits from external actions:

- search current sources that are not in the input dataset;
- use bash for scratch extraction, sorting, or table checks;
- delegate evidence extraction to a specialist with narrower instructions;
- reduce structured evidence into a ranked sector brief.

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

## How schema enforcement works

For agent-backed operations, DocETL converts the operation's `output.schema` into
an OpenAI Agents SDK `output_type`. The SDK asks the agent for that typed final
output. DocETL then runs the normal operation validation path:

- missing output fields fail type validation;
- values must match the declared DocETL schema types;
- any `validate=[...]` expressions must pass;
- failed validation retries according to `num_retries_on_validate_failure`.

The sandbox can contain scratch files, but the accepted DocETL result is the
validated structured output.

## Tuning tool budgets

Use small budgets first:

```python
docetl.Agent(tools=[WebSearchTool(), bash], max_turns=4, max_tool_calls=6)
```

Increase `max_turns` when the agent needs more reasoning steps. Increase
`max_tool_calls` when each item legitimately requires more sources or sandbox
actions. The limits are per operation call, so map limits apply per row, filter
limits apply per row, and reduce limits apply per group.

## Security notes

Python tools execute as trusted Python in your process. Hosted OpenAI Agents SDK
tools, such as web search or hosted bash, follow the SDK and provider behavior
for the selected model backend. Keep sandbox network access disabled unless the
task requires it, and pass durable data between DocETL operations through output
schemas even when agents also share a persistent sandbox filesystem.
