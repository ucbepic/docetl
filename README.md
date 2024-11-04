# DocETL: Powering Complex Document Processing Pipelines

[![Website](https://img.shields.io/badge/Website-docetl.org-blue)](https://docetl.org)
[![Documentation](https://img.shields.io/badge/Documentation-docs-green)](https://ucbepic.github.io/docetl)
[![Discord](https://img.shields.io/discord/1285485891095236608?label=Discord&logo=discord)](https://discord.gg/fHp7B2X3xx)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2410.12189)

![DocETL Figure](docs/assets/readmefig.png)

DocETL is a tool for creating and executing data processing pipelines, especially suited for complex document processing tasks. It offers a low-code, declarative YAML interface to define LLM-powered operations on complex data.

## When to Use DocETL

DocETL is the ideal choice when you're looking to maximize correctness and output quality for complex tasks over a collection of documents or unstructured datasets. You should consider using DocETL if:

- You want to perform semantic processing on a collection of data
- You have complex tasks that you want to represent via map-reduce (e.g., map over your documents, then group by the result of your map call & reduce)
- You're unsure how to best express your task to maximize LLM accuracy
- You're working with long documents that don't fit into a single prompt or are too lengthy for effective LLM reasoning
- You have validation criteria and want tasks to automatically retry when the validation fails

## Cool Things People Are Doing with DocETL

- [Conversation Generator](https://github.com/PassionFruits-net/docetl-conversation)
- [Text-to-speech](https://github.com/PassionFruits-net/docetl-speaker)
- [YouTube Transcript Topic Extraction](https://github.com/rajib76/docetl_examples)

## (Educational) Threads

- [UI/UX Thoughts](https://x.com/sh_reya/status/1846235904664273201)
- [Using Gleaning to Improve Output Quality](https://x.com/sh_reya/status/1843354256335876262)
- [Deep Dive on Resolve Operator](https://x.com/sh_reya/status/1840796824636121288)

## Installation

You can install DocETL using either PyPI or from source. We recommend installing from source for the latest features and bug fixes.

### Prerequisites

Before installing DocETL, ensure you have Python 3.10 or later installed on your system. You can check your Python version by running:

```bash
python --version
```

### Install from PyPI

```bash
pip install docetl
```

### Install from Source

1. Clone the DocETL repository (or your fork):

```bash
git clone https://github.com/ucbepic/docetl.git
cd docetl
```

2. Install Poetry (if not already installed):

```bash
pip install poetry
```

3. Install the project dependencies:

```bash
poetry install
```

4. Set up your OpenAI API key and other environment variables:

Copy the .env.sample file under the root directory to .env and modify the environment variables inside as needed.

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=

BACKEND_ALLOW_ORIGINS=
BACKEND_HOST=localhost
BACKEND_PORT=8000
BACKEND_RELOAD=True

FRONTEND_HOST=0.0.0.0
FRONTEND_PORT=3000
```

Alternatively, you can set the OPENAI_API_KEY environment variable and others in your shell.

5. Run the basic test suite to ensure everything is working (this costs less than $0.01 with OpenAI):

```bash
make tests-basic
```

That's it! You've successfully installed DocETL and are ready to start processing documents.

For more detailed information on usage and configuration, please refer to our [documentation](https://ucbepic.github.io/docetl).
