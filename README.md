# DocETL

[![Website](https://img.shields.io/badge/Website-docetl.org-blue)](https://docetl.org)
[![Documentation](https://img.shields.io/badge/Docs-ucbepic.github.io/docetl-green)](https://ucbepic.github.io/docetl)
[![Discord](https://img.shields.io/discord/1285485891095236608?label=Discord&logo=discord)](https://discord.gg/fHp7B2X3xx)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2410.12189)

LLM-powered data processing pipelines for complex document tasks. Define operations declaratively, and DocETL handles chunking, validation, retries, and cost-accuracy optimization.

```python
import docetl

docetl.default_model = "gpt-4o-mini"

df = (
    docetl.read_json("tickets.json")
    .map(
        prompt="Classify this support ticket: {{ input.text }}",
        output={"schema": {"category": "str", "priority": "str"}},
    )
    .reduce(
        reduce_key="category",
        prompt="Summarize these tickets: {% for t in inputs %}{{ t.text }}{% endfor %}",
        output={"schema": {"summary": "str"}},
    )
    .collect()
)
```

![DocETL TUI](docs/assets/progress-view/tui-real-complete.png)

## Install

```bash
pip install docetl
```

Set your API key (or the key for whichever LLM provider you use):

```bash
export OPENAI_API_KEY=your_key_here
```

## Why DocETL

Use DocETL when you need to **maximize correctness** for complex tasks over unstructured data:

- **Map-reduce over documents** — classify, extract, summarize, then aggregate by group
- **Long documents** — automatic splitting with context-preserving gather operations
- **Entity resolution** — fuzzy deduplication across LLM-extracted fields
- **Validation and retries** — define rules, and operations automatically retry on failure
- **Cost-accuracy optimization** — [MOAR](https://ucbepic.github.io/docetl/optimization/python-api/) explores model choices and prompt rewrites to find the Pareto frontier

## Three ways to build pipelines

| | Best for | How it works |
|---|---|---|
| **[Python API](https://ucbepic.github.io/docetl/python/)** (recommended) | Production code, notebooks, scripting | Chain operations in Python — `read_json().map().reduce().collect()` |
| **[YAML](https://ucbepic.github.io/docetl/tutorial/)** (low-code) | Config-driven workflows, no Python needed | Declare your pipeline in YAML, run with `docetl run pipeline.yaml` |
| **[DocWrangler UI](https://ucbepic.github.io/docetl/playground/)** | Prompt development, exploration | Visual playground — edit prompts, see results in real time |

<details>
<summary>Python API example</summary>

```python
import docetl

docetl.default_model = "gpt-4o-mini"

df = (
    docetl.read_json("data.json")
    .map(prompt="Classify: {{ input.text }}", output={"schema": {"category": "str"}})
    .reduce(reduce_key="category", prompt="Summarize: {% for t in inputs %}{{ t.text }}{% endfor %}", output={"schema": {"summary": "str"}})
    .collect()
)
```

</details>

<details>
<summary>YAML example</summary>

```yaml
datasets:
  tickets:
    type: file
    path: tickets.json

default_model: gpt-4o-mini

operations:
  - name: classify
    type: map
    prompt: "Classify this support ticket and assign a priority."
    output:
      schema:
        category: str
        priority: str

pipeline:
  steps:
    - name: triage
      input: tickets
      operations: [classify]
  output:
    type: file
    path: output.json
```

```bash
docetl run pipeline.yaml
```

</details>

<details>
<summary>DocWrangler UI</summary>

Interactive playground at [docetl.org/playground](https://docetl.org/playground):

![DocWrangler](docs/assets/tutorial/one-operation.png)

</details>

## Documentation

| Resource | Description |
|----------|-------------|
| [Python API Guide](https://ucbepic.github.io/docetl/python/) | Frame API reference — operations, config, optimization |
| [YAML Tutorial](https://ucbepic.github.io/docetl/tutorial/) | Step-by-step walkthrough of YAML pipelines |
| [Operators](https://ucbepic.github.io/docetl/operators/map/) | Map, filter, reduce, resolve, split, gather, extract, and more |
| [Optimization (MOAR)](https://ucbepic.github.io/docetl/optimization/python-api/) | Automatic cost-accuracy optimization |
| [DocWrangler Setup](https://ucbepic.github.io/docetl/playground/) | Run the interactive UI locally or via Docker |
| [Claude Code Quick Start](https://ucbepic.github.io/docetl/quickstart-claude-code/) | Describe your task and let Claude build the pipeline |

## Community

- [Discord](https://discord.gg/fHp7B2X3xx) — ask questions, share pipelines
- [Conversation Generator](https://github.com/PassionFruits-net/docetl-conversation) — community project
- [Text-to-speech](https://github.com/PassionFruits-net/docetl-speaker) — community project
- [YouTube Transcript Topics](https://github.com/rajib76/docetl_examples) — community project

## Development

```bash
git clone https://github.com/ucbepic/docetl.git
cd docetl
make install
make tests-basic  # < $0.01 with OpenAI
```

See the [DocWrangler Setup Guide](https://ucbepic.github.io/docetl/playground/) for running the UI locally.

## Papers

DocETL was created at the [EPIC Data Lab](https://epic.berkeley.edu/) and [Data Systems and Foundations](https://dsf.berkeley.edu/) group at UC Berkeley.

**DocETL** — VLDB 2025 ([paper](https://arxiv.org/abs/2410.12189))

```bibtex
@article{shankar2025docetl,
  title={DocETL: Agentic Query Rewriting and Evaluation for Complex Document Processing},
  author={Shankar, Shreya and Chambers, Tristan and Shah, Tarak and Parameswaran, Aditya G and Wu, Eugene},
  journal={Proceedings of the VLDB Endowment},
  volume={18},
  number={9},
  pages={3035--3048},
  year={2025}
}
```

**DocWrangler** — UIST 2025, Best Paper Honorable Mention ([paper](https://arxiv.org/abs/2504.14764))

```bibtex
@inproceedings{shankar2025docwrangler,
  title={Steering Semantic Data Processing With DocWrangler},
  author={Shankar, Shreya and Chopra, Bhavya and Hasan, Mawil and Lee, Stephen and Hartmann, Bj{\"o}rn and Hellerstein, Joseph M and Parameswaran, Aditya G and Wu, Eugene},
  booktitle={Proceedings of the ACM Symposium on User Interface Software and Technology (UIST)},
  year={2025}
}
```

**MOAR** — VLDB 2026 ([paper](https://arxiv.org/abs/2512.02289))

```bibtex
@article{wei2026moar,
  title={Multi-Objective Agentic Rewrites for Unstructured Data Processing},
  author={Wei, Lindsey Linxi and Shankar, Shreya and Zeighami, Sepanta and Chung, Yeounoh and Ozcan, Fatma and Parameswaran, Aditya G},
  journal={Proceedings of the VLDB Endowment},
  year={2026}
}
```
