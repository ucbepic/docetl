# Installation

DocETL can be easily installed using pip, Python's package installer, or from source. Follow these steps to get DocETL up and running on your system:

## Prerequisites

Before installing DocETL, ensure you have Python 3.10 or later installed on your system. You can check your Python version by running:

```bash
python --version
```

## Installation via pip

1. Install DocETL using pip:

```bash
pip install docetl
```

This command will install DocETL along with its dependencies as specified in the pyproject.toml file.

## Installation from Source

To install DocETL from source, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/shreyashankar/docetl.git
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

This will create a virtual environment and install all the required dependencies.

## Verifying the Installation

To verify that DocETL has been installed correctly, you can run the following command in your terminal:

```bash
docetl version
```

If the installation was successful, this command will display the version of DocETL installed on your system.

## Troubleshooting

If you encounter any issues during installation, please ensure that:

- Your Python version is 3.10 or later
- You have the latest version of pip installed
- Your system meets all the requirements specified in the pyproject.toml file

For further assistance, please refer to the project's GitHub repository or reach out to the community for support.
