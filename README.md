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
- You have complex tasks that you want to represent via map-reduce
- You're unsure how to best express your task to maximize LLM accuracy
- You're working with long documents that don't fit into a single prompt
- You have validation criteria and want tasks to automatically retry when validation fails

## Community Projects

- [Conversation Generator](https://github.com/PassionFruits-net/docetl-conversation)
- [Text-to-speech](https://github.com/PassionFruits-net/docetl-speaker)
- [YouTube Transcript Topic Extraction](https://github.com/rajib76/docetl_examples)

## Educational Resources

- [UI/UX Thoughts](https://x.com/sh_reya/status/1846235904664273201)
- [Using Gleaning to Improve Output Quality](https://x.com/sh_reya/status/1843354256335876262)
- [Deep Dive on Resolve Operator](https://x.com/sh_reya/status/1840796824636121288)

## Installation

There are three ways to run DocETL:

### 1. Using Docker (Recommended for Quick Start)

The easiest way to get started is using Docker:

1. Create the required environment files:

Create `.env` in the root directory:
```bash
OPENAI_API_KEY=your_api_key_here
BACKEND_ALLOW_ORIGINS=
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
BACKEND_RELOAD=True
FRONTEND_HOST=0.0.0.0
FRONTEND_PORT=3000
```

Create `.env.local` in the `website` directory:
```bash
OPENAI_API_KEY=sk-xxx
OPENAI_API_BASE=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini

NEXT_PUBLIC_BACKEND_HOST=localhost
NEXT_PUBLIC_BACKEND_PORT=8000
```

2. Run Docker:
```bash
make docker
```

This will:
- Create a Docker volume for persistent data
- Build the DocETL image
- Run the container with the UI accessible at http://localhost:3000 and API at http://localhost:8000

To clean up Docker resources (note that this will delete the Docker volume):
```bash
make docker-clean
```

### 2. Using pip (Basic Installation)

If you just want to use DocETL as a Python package:

#### Prerequisites
- Python 3.10 or later
- OpenAI API key

```bash
pip install docetl
```

To see examples of how to use DocETL, check out the [tutorial](https://ucbepic.github.io/docetl/tutorial/).

### 3. Running the UI Locally (Development Setup)

For development or if you want to run the UI locally:

1. Clone the repository:
```bash
git clone https://github.com/ucbepic/docetl.git
cd docetl
```

2. Set up environment variables in `.env` in the root/top-level directory:
```bash
OPENAI_API_KEY=your_api_key_here
BACKEND_ALLOW_ORIGINS=
BACKEND_HOST=localhost
BACKEND_PORT=8000
BACKEND_RELOAD=True
FRONTEND_HOST=0.0.0.0
FRONTEND_PORT=3000
```

And create an .env.local file in the `website` directory with the following:
```bash
OPENAI_API_KEY=sk-xxx
OPENAI_API_BASE=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini

NEXT_PUBLIC_BACKEND_HOST=localhost
NEXT_PUBLIC_BACKEND_PORT=8000
```

3. Install dependencies:
```bash
make install      # Install Python package
make install-ui   # Install UI dependencies
```

Note that the OpenAI API key, base, and model name are for the UI assistant only; not the DocETL pipeline execution engine.

4. Start the development server:
```bash
make run-ui-dev
```

5. Visit http://localhost:3000/playground

### Development Setup

If you're planning to contribute or modify DocETL, you can verify your setup by running the test suite:

```bash
make tests-basic  # Runs basic test suite (costs < $0.01 with OpenAI)
```

For detailed documentation and tutorials, visit our [documentation](https://ucbepic.github.io/docetl).
