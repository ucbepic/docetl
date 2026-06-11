# DocETL

[![GitHub](https://img.shields.io/github/stars/ucbepic/docetl?style=social)](https://github.com/ucbepic/docetl)
[![Website](https://img.shields.io/badge/Website-docetl.org-blue)](https://docetl.org)
[![Documentation](https://img.shields.io/badge/Documentation-docs-green)](https://ucbepic.github.io/docetl)
[![Discord](https://img.shields.io/discord/1285485891095236608?label=Discord&logo=discord)](https://discord.gg/fHp7B2X3xx)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2410.12189)

DocETL is a declarative query engine and optimizer for LLM-powered data
processing. Think of DocETL as an agentic map-reduce framework. DocETL exposes
high-level operations — e.g., map, reduce, filter, resolve, extract — that can
be authored in natural language and executed by agents, and an optimizer that
rewrites pipelines by searching over models, prompts, and operation
decompositions.

Use it when you have a task over a collection of documents or unstructured
records — e.g., extracting and aggregating themes across thousands of
transcripts — and you care about output quality, cost, or both.

## Getting Started

Pipelines can be written in Python or YAML — both are first class:

=== "Python"

    ```python
    import docetl

    docetl.default_model = "gpt-4o-mini"

    results = (
        docetl.read_json("input.json")
        .map(prompt="Classify: {{ input.text }}", output={"schema": {"category": "str"}})
        .reduce(reduce_key="category", prompt="Summarize: {{ inputs }}", output={"schema": {"summary": "str"}})
        .collect()
    )
    ```

    See the [Python API reference](api-reference/python.md) for the full reference.

=== "YAML"

    Define your pipeline declaratively, then run it from the CLI:

    ```yaml
    default_model: gpt-4o-mini
    datasets:
      docs:
        type: file
        path: input.json
    operations:
      - name: classify
        type: map
        prompt: "Classify: {{ input.text }}"
        output:
          schema:
            category: str
    pipeline:
      steps:
        - name: step1
          input: docs
          operations: [classify]
      output:
        type: file
        path: output.json
    ```

    ```bash
    docetl run pipeline.yaml
    ```

    See the [tutorial](tutorial.md) for a complete walkthrough.

### Pandas Integration

For quick exploration on existing DataFrames, use the `.semantic` accessor:

```python
df.semantic.map(prompt="...", output={"schema": {"field": "str"}})
```

See the [Pandas integration guide](pandas/index.md) for details.

!!! tip "Fastest Way: Claude Code"
    Clone this repo and run `claude` to use the built-in DocETL skill. Just describe your data processing task and Claude will create and run the pipeline for you. See [Quick Start (Claude Code)](quickstart-claude-code.md) for details.

## Project Origin

DocETL was created by members of the EPIC Data Lab and Data Systems and Foundations group at UC Berkeley. The EPIC (Effective Programming, Interaction, and Computation with Data) Lab focuses on developing low-code and no-code interfaces for data work, powered by next-generation predictive programming techniques. DocETL is one of the projects that emerged from our research efforts to streamline complex document processing tasks.

For more information about the labs and other projects, visit the [EPIC Lab webpage](https://epic.berkeley.edu/) and the [Data Systems and Foundations webpage](https://dsf.berkeley.edu/).
