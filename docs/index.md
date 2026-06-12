# DocETL: Declarative & Agentic Map-Reduce { .center-title }

<p align="center">
<a href="https://github.com/ucbepic/docetl"><img src="https://img.shields.io/github/stars/ucbepic/docetl?style=social" alt="GitHub stars"></a>
<a href="https://docetl.org"><img src="https://img.shields.io/badge/Website-docetl.org-blue" alt="Website"></a>
<a href="https://ucbepic.github.io/docetl"><img src="https://img.shields.io/badge/Documentation-docs-green" alt="Documentation"></a>
<a href="https://discord.gg/fHp7B2X3xx"><img src="https://img.shields.io/discord/1285485891095236608?label=Discord&logo=discord" alt="Discord"></a>
<a href="https://arxiv.org/abs/2410.12189"><img src="https://img.shields.io/badge/Paper-arXiv-red" alt="Paper"></a>
</p>

<p align="center"><img src="assets/docetl-overview.svg" alt="DocETL pipeline overview" width="680"></p>

DocETL helps you process large collections of data (structured and
unstructured) with LLMs. You write each operation in natural language, e.g.,
"pull out every complaint in this ticket," and DocETL

- provides the operators you need (map, reduce, filter, and more) and
  orchestrates them, parallelizing work across your data,
- optimizes your pipeline automatically, swapping models, rewriting prompts,
  decomposing operations, and replacing subtasks with code wherever possible,
  to raise accuracy and cut cost, and
- returns tables, easy to query in your favorite database.

Without DocETL, you write each LLM call yourself, wire them together, and
tune the result for accuracy, cost, and latency by hand.


## Getting Started

Write pipelines in Python, or in YAML if you prefer low-code / no-code. (There is also a [pandas accessor](pandas/index.md) for quick one-off operations on DataFrames.)

=== "Python"

    ```python
    import docetl

    docetl.default_model = "gpt-4o-mini"

    # What themes come up across customer interviews?
    interviews = docetl.read_json("interviews.json")

    # Extract themes from each interview
    interviews = interviews.map(
        prompt="List the themes discussed in this interview: {{ input.transcript }}",
        output={"schema": {"themes": "list[str]"}},
    )

    # Synthesize the top themes across all interviews
    themes = interviews.reduce(
        reduce_key="_all",
        prompt="""Identify the top recurring themes across these interviews:
    {% for item in inputs %}- {{ item.themes }}
    {% endfor %}""",
        output={"schema": {"top_themes": "list[{name: str, description: str, num_mentions: int}]"}},
    )

    results = themes.collect()  # nothing runs until this line
    ```

    `collect()` returns the result rows as a list of dicts (use
    `themes.to_pandas()` for a DataFrame); `results[0]["top_themes"]` looks like:

    ```python
    [
        {"name": "onboarding friction", "description": "confusion during first-week setup", "num_mentions": 12},
        {"name": "pricing transparency", "description": "unclear billing and plan limits", "num_mentions": 8},
        ...
    ]
    ```

    See the [User Guide](concepts/pipelines.md) for details.

=== "YAML"

    Define the same pipeline declaratively, then run it from the CLI:

    ```yaml
    default_model: gpt-4o-mini
    datasets:
      interviews:
        type: file
        path: interviews.json
    operations:
      - name: extract_themes
        type: map
        prompt: "List the themes discussed in this interview: {{ input.transcript }}"
        output:
          schema:
            themes: list[str]
      - name: synthesize_themes
        type: reduce
        reduce_key: _all
        prompt: |
          Identify the top recurring themes across these interviews:
          {% for item in inputs %}- {{ item.themes }}
          {% endfor %}
        output:
          schema:
            top_themes: "list[{name: str, description: str, num_mentions: int}]"
    pipeline:
      steps:
        - name: analyze
          input: interviews
          operations: [extract_themes, synthesize_themes]
      output:
        type: file
        path: themes.json
    ```

    ```bash
    docetl run pipeline.yaml
    ```

    See the [tutorial](tutorial.md) for a complete walkthrough.

## Project Origin

DocETL is built at UC Berkeley by the [EPIC Data Lab](https://epic.berkeley.edu/) and the [Data Systems and Foundations](https://dsf.berkeley.edu/) group.
