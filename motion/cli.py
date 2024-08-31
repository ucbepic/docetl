from pathlib import Path
from typing import Optional

import typer

from motion.builder import Optimizer
from motion.runner import DSLRunner
from motion.operations.utils import clear_cache as cc

app = typer.Typer()


@app.command()
def build(
    yaml_file: Path = typer.Argument(
        ..., help="Path to the YAML file containing the pipeline configuration"
    ),
    max_threads: Optional[int] = typer.Option(
        None, help="Maximum number of threads to use for parallel operations"
    ),
    model: str = typer.Option("gpt-4o", help="Model to use for optimization"),
    timeout: int = typer.Option(
        60, help="Timeout for optimization operations in seconds"
    ),
):
    """
    Build and optimize the configuration specified in the YAML file.

    Args:
        yaml_file (Path): Path to the YAML file containing the pipeline configuration.
        max_threads (Optional[int]): Maximum number of threads to use for parallel operations.
        model (str): Model to use for optimization. Defaults to "gpt-4o".
        timeout (int): Timeout for optimization operations in seconds. Defaults to 60.
    """
    optimizer = Optimizer(
        str(yaml_file),
        max_threads=max_threads,
        model=model,
        timeout=timeout,
    )
    optimizer.optimize()
    typer.echo("Optimization complete. Check the optimized configuration.")


@app.command()
def run(
    yaml_file: Path = typer.Argument(
        ..., help="Path to the YAML file containing the pipeline configuration"
    ),
    max_threads: Optional[int] = typer.Option(
        None, help="Maximum number of threads to use for parallel operations"
    ),
):
    """
    Run the configuration specified in the YAML file.

    Args:
        yaml_file (Path): Path to the YAML file containing the pipeline configuration.
        max_threads (Optional[int]): Maximum number of threads to use for parallel operations.
    """
    runner = DSLRunner(str(yaml_file), max_threads=max_threads)
    runner.run()


@app.command()
def clear_cache():
    """
    Clear the LLM cache stored on disk.
    """
    try:
        cc()
        typer.echo("Cache cleared successfully.")
    except Exception as e:
        typer.echo(f"An error occurred while clearing the cache: {str(e)}")


if __name__ == "__main__":
    app()
