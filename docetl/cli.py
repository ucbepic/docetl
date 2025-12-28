import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from docetl.operations.utils import clear_cache as cc
from docetl.runner import DSLRunner

console = Console(stderr=True)

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def build(
    yaml_file: Path = typer.Argument(
        ..., help="Path to the YAML file containing the pipeline configuration"
    ),
    optimizer: str = typer.Option(
        "moar",
        "--optimizer",
        "-o",
        help="Optimizer to use: 'moar' (default) or 'v1' (deprecated)",
    ),
    max_threads: int | None = typer.Option(
        None, help="Maximum number of threads to use for running operations"
    ),
    resume: bool = typer.Option(
        False, help="Resume optimization from a previous build that may have failed"
    ),
    save_path: Path = typer.Option(
        None, help="Path to save the optimized pipeline configuration"
    ),
):
    """
    Build and optimize the configuration specified in the YAML file.
    Any arguments passed here will override the values in the YAML file.

    Args:
        yaml_file (Path): Path to the YAML file containing the pipeline configuration.
        optimizer (str): Optimizer to use - 'moar' or 'v1' (required).
        max_threads (int | None): Maximum number of threads to use for running operations.
        resume (bool): Whether to resume optimization from a previous run. Defaults to False.
        save_path (Path): Path to save the optimized pipeline configuration.
    """
    # Get the current working directory (where the user called the command)
    cwd = os.getcwd()

    # Load .env file from the current working directory
    env_file = os.path.join(cwd, ".env")
    if os.path.exists(env_file):
        load_dotenv(env_file)

    # Validate optimizer choice
    if optimizer not in ["moar", "v1"]:
        typer.echo(
            f"Error: optimizer must be 'moar' or 'v1', got '{optimizer}'", err=True
        )
        raise typer.Exit(1)

    # Load YAML to check for optimizer_config
    import yaml as yaml_lib

    with open(yaml_file, "r") as f:
        config = yaml_lib.safe_load(f)

    if optimizer == "moar":
        optimizer_config = config.get("optimizer_config", {})
        if not optimizer_config:
            example_yaml = """optimizer_config:
  type: moar
  save_dir: ./moar_results
  available_models:
    - gpt-5
    - gpt-4o
  evaluation_file: workloads/medical/evaluate_medications.py
  metric_key: medication_extraction_score
  max_iterations: 40
  model: gpt-5"""

            error_panel = Panel(
                f"[bold red]Error:[/bold red] optimizer_config section is required in YAML for MOAR optimizer.\n\n"
                f"[bold]Example:[/bold]\n"
                f"[dim]{example_yaml}[/dim]\n\n"
                f"[yellow]Note:[/yellow] dataset_name is inferred from the 'datasets' section. "
                f"dataset_path can optionally be specified in optimizer_config, otherwise it's inferred from the 'datasets' section.",
                title="[bold red]Missing optimizer_config[/bold red]",
                border_style="red",
            )
            console.print(error_panel)
            raise typer.Exit(1)

        if optimizer_config.get("type") != "moar":
            error_panel = Panel(
                f"[bold red]Error:[/bold red] optimizer_config.type must be 'moar', got '[yellow]{optimizer_config.get('type')}[/yellow]'",
                title="[bold red]Invalid optimizer type[/bold red]",
                border_style="red",
            )
            console.print(error_panel)
            raise typer.Exit(1)

        # Validate required fields in optimizer_config
        required_fields = {
            "save_dir": "Output directory for MOAR results",
            "available_models": "List of model names to use",
            "evaluation_file": "Path to evaluation function file",
            "metric_key": "Key to extract from evaluation results",
            "max_iterations": "Number of MOARSearch iterations",
            "model": "LLM model name for directive instantiation",
        }

        missing_fields = [
            field for field in required_fields if not optimizer_config.get(field)
        ]
        if missing_fields:
            # Create a table for required fields
            fields_table = Table(
                show_header=True, header_style="bold cyan", box=None, padding=(0, 2)
            )
            fields_table.add_column("Field", style="yellow")
            fields_table.add_column("Description", style="dim")

            for field, desc in required_fields.items():
                style = "bold red" if field in missing_fields else "dim"
                fields_table.add_row(f"[{style}]{field}[/{style}]", desc)

            # Create example YAML
            example_yaml = """optimizer_config:
  type: moar
  save_dir: ./moar_results
  available_models:
    - gpt-5
    - gpt-4o
  evaluation_file: workloads/medical/evaluate_medications.py
  metric_key: medication_extraction_score
  max_iterations: 40
  model: gpt-5"""

            missing_list = ", ".join(
                [f"[bold red]{f}[/bold red]" for f in missing_fields]
            )

            # Build error content with table rendered separately
            from rich.console import Group

            error_group = Group(
                f"[bold red]Missing required fields:[/bold red] {missing_list}\n",
                "[bold]Required fields:[/bold]",
                fields_table,
                f"\n[bold]Example:[/bold]\n[dim]{example_yaml}[/dim]\n",
                "[yellow]Note:[/yellow] dataset_name is inferred from the 'datasets' section. "
                "dataset_path can optionally be specified in optimizer_config, otherwise it's inferred from the 'datasets' section.",
            )

            error_panel = Panel(
                error_group,
                title="[bold red]Missing Required Fields[/bold red]",
                border_style="red",
            )
            console.print(error_panel)
            raise typer.Exit(1)

        # Run MOAR optimization
        from docetl.moar.cli_helpers import run_moar_optimization

        try:
            results = run_moar_optimization(
                yaml_path=str(yaml_file),
                optimizer_config=optimizer_config,
            )
            typer.echo("\n✅ MOAR optimization completed successfully!")
            typer.echo(f"   Results saved to: {optimizer_config.get('save_dir')}")
            if results.get("evaluation_file"):
                typer.echo(f"   Evaluation: {results['evaluation_file']}")
        except Exception as e:
            typer.echo(f"Error running MOAR optimization: {e}", err=True)
            raise typer.Exit(1)

    else:  # v1 optimizer (deprecated)
        console.print(
            Panel(
                "[bold yellow]Warning:[/bold yellow] The V1 optimizer is deprecated. "
                "Please use MOAR optimizer instead: [bold]docetl build pipeline.yaml --optimizer moar[/bold]",
                title="[bold yellow]Deprecated Optimizer[/bold yellow]",
                border_style="yellow",
            )
        )
        runner = DSLRunner.from_yaml(str(yaml_file), max_threads=max_threads)
        runner.optimize(
            save=True,
            return_pipeline=False,
            resume=resume,
            save_path=save_path,
        )


@app.command()
def run(
    yaml_file: Path = typer.Argument(
        ..., help="Path to the YAML file containing the pipeline configuration"
    ),
    max_threads: int | None = typer.Option(
        None, help="Maximum number of threads to use for running operations"
    ),
):
    """
    Run the configuration specified in the YAML file.

    Args:
        yaml_file (Path): Path to the YAML file containing the pipeline configuration.
        max_threads (int | None): Maximum number of threads to use for running operations.
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


@app.command("install-skill")
def install_skill(
    uninstall: bool = typer.Option(
        False, "--uninstall", "-u", help="Remove the installed skill instead"
    ),
):
    """
    Install the DocETL Claude Code skill to your personal skills directory.

    This makes the DocETL skill available in Claude Code for any project.
    The skill helps you build and run DocETL pipelines.
    """
    import shutil

    # Find the skill source - try multiple locations
    # 1. Installed package location (via importlib.resources)
    # 2. Development location (relative to this file)
    skill_source = None

    # Try to find via package resources first
    try:
        import importlib.resources as pkg_resources

        # For Python 3.9+, use files()
        try:
            package_root = Path(pkg_resources.files("docetl")).parent
            potential_source = package_root / ".claude" / "skills" / "docetl"
            if potential_source.exists():
                skill_source = potential_source
        except (TypeError, AttributeError):
            pass
    except ImportError:
        pass

    # Fallback: try relative to this file (development mode)
    if skill_source is None:
        dev_source = Path(__file__).parent.parent / ".claude" / "skills" / "docetl"
        if dev_source.exists():
            skill_source = dev_source

    if skill_source is None or not skill_source.exists():
        console.print(
            Panel(
                "[bold red]Error:[/bold red] Could not find the DocETL skill files.\n\n"
                "This may happen if the package was not installed correctly.\n"
                "Try reinstalling: [bold]pip install --force-reinstall docetl[/bold]",
                title="[bold red]Skill Not Found[/bold red]",
                border_style="red",
            )
        )
        raise typer.Exit(1)

    # Target directory
    skill_target = Path.home() / ".claude" / "skills" / "docetl"

    if uninstall:
        if skill_target.exists():
            shutil.rmtree(skill_target)
            console.print(
                Panel(
                    f"[bold green]Success![/bold green] DocETL skill removed from:\n"
                    f"[dim]{skill_target}[/dim]",
                    title="[bold green]Skill Uninstalled[/bold green]",
                    border_style="green",
                )
            )
        else:
            console.print(
                Panel(
                    "[yellow]The DocETL skill is not currently installed.[/yellow]",
                    title="[yellow]Nothing to Uninstall[/yellow]",
                    border_style="yellow",
                )
            )
        return

    # Create parent directories if needed
    skill_target.parent.mkdir(parents=True, exist_ok=True)

    # Copy the skill
    if skill_target.exists():
        shutil.rmtree(skill_target)

    shutil.copytree(skill_source, skill_target)

    console.print(
        Panel(
            f"[bold green]Success![/bold green] DocETL skill installed to:\n"
            f"[dim]{skill_target}[/dim]\n\n"
            "[bold]Next steps:[/bold]\n"
            "1. Restart Claude Code if it's running\n"
            "2. The skill will automatically activate when you work on DocETL tasks\n\n"
            "[dim]To uninstall: docetl install-skill --uninstall[/dim]",
            title="[bold green]Skill Installed[/bold green]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    app()
