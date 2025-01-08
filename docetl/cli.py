from pathlib import Path
from typing import Optional

import os
import typer

from docetl.operations.utils import clear_cache as cc
from docetl.runner import DSLRunner

from dotenv import load_dotenv

app = typer.Typer()

@app.command()
def build(
    yaml_file: Path = typer.Argument(
        ..., help="Path to the YAML file containing the pipeline configuration"
    ),
    max_threads: Optional[int] = typer.Option(
        None, help="Maximum number of threads to use for running operations"
    ),
    model: str = typer.Option("gpt-4o", help="Model to use for optimization"),
    resume: bool = typer.Option(
        False, help="Resume optimization from a previous build that may have failed"
    ),
    timeout: int = typer.Option(
        60, help="Timeout for optimization operations in seconds"
    ),
):
    """
    Build and optimize the configuration specified in the YAML file.

    Args:
        yaml_file (Path): Path to the YAML file containing the pipeline configuration.
        max_threads (Optional[int]): Maximum number of threads to use for running operations.
        model (str): Model to use for optimization. Defaults to "gpt-4o".
        resume (bool): Whether to resume optimization from a previous run. Defaults to False.
        timeout (int): Timeout for optimization operations in seconds. Defaults to 60.
    """
    # Get the current working directory (where the user called the command)
    cwd = os.getcwd()

    # Load .env file from the current working directory
    env_file = os.path.join(cwd, ".env")
    if os.path.exists(env_file):
        load_dotenv(env_file)

    runner = DSLRunner.from_yaml(str(yaml_file), max_threads=max_threads)
    runner.optimize(
        save=True, return_pipeline=False, model=model, resume=resume, timeout=timeout
    )


@app.command()
def run(
    yaml_file: Path = typer.Argument(
        ..., help="Path to the YAML file containing the pipeline configuration"
    ),
    max_threads: Optional[int] = typer.Option(
        None, help="Maximum number of threads to use for running operations"
    ),
):
    """
    Run the configuration specified in the YAML file.

    Args:
        yaml_file (Path): Path to the YAML file containing the pipeline configuration.
        max_threads (Optional[int]): Maximum number of threads to use for running operations.
    """
    # Get the current working directory (where the user called the command)
    cwd = os.getcwd()

    # Load .env file from the current working directory
    env_file = os.path.join(cwd, ".env")
    if os.path.exists(env_file):
        load_dotenv(env_file)

    runner = DSLRunner.from_yaml(str(yaml_file), max_threads=max_threads)
    runner.load_run_save()


@app.command()
def clear_cache():
    """
    Clear the LLM cache stored on disk.
    """
    cc()


@app.command()
def version():
    """
    Display the current version of DocETL.
    """
    import docetl

    typer.echo(f"DocETL version: {docetl.__version__}")


if __name__ == "__main__":
    app()
