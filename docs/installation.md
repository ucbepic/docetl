# Installation

DocETL can be easily installed using pip, Python's package installer, or from source. Follow these steps to get DocETL up and running on your system:

## 🛠️ Prerequisites

Before installing DocETL, ensure you have Python 3.10 or later installed on your system. You can check your Python version by running:

## 📦 Installation via pip

1. Install DocETL using pip:

```bash
pip install docetl
```

If you want to use the parsing tools, you need to install the `parsing` extra:

```bash
pip install docetl[parsing]
```

This command will install DocETL along with its dependencies as specified in the pyproject.toml file. To verify that DocETL has been installed correctly, you can run the following command in your terminal:

```bash
docetl version
```

## 🔧 Installation from Source

To install DocETL from source, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/ucbepic/docetl.git
cd docetl
```

2. Install Poetry (if not already installed):

```bash
pip install poetry
```

3. Install the project dependencies and DocETL:

```bash
poetry install
```

If you want to use the parsing tools, you need to install the `parsing` extra:

```bash
poetry install --extras "parsing"
```

This will create a virtual environment and install all the required dependencies.

4. Set up your OpenAI API key:

Create a .env file in the project root and add your OpenAI API key:

```bash
OPENAI_API_KEY=your_api_key_here
```

Alternatively, you can set the OPENAI_API_KEY environment variable in your shell.

5. Run the basic test suite to ensure everything is working (this costs less than $0.01 with OpenAI):

```bash
make tests-basic
```

## 🚨 Troubleshooting

If you encounter any issues during installation, please ensure that:

- Your Python version is 3.10 or later
- You have the latest version of pip installed
- Your system meets all the requirements specified in the pyproject.toml file

For further assistance, please refer to the project's GitHub repository or reach out on the [Discord server](https://discord.gg/fHp7B2X3xx).
