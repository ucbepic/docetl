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
    max_threads: int | None = typer.Option(
        None, help="Maximum number of threads to use for running operations"
    ),
):
    """
    Optimize a pipeline using MOAR (Multi-Objective Agentic Rewrites).

    Requires an optimizer_config section in the YAML with either:
      - evaluation_file: path to a Python file with a @docetl.register_eval function
        and metric_key: key to extract from evaluation results, or
      - judge_model: an LLM judge that rates and ranks plan outputs
        (optionally judge_criteria; auto-generated from the pipeline if omitted)

    Models are auto-detected from API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY,
    GEMINI_API_KEY, AZURE_API_KEY) unless available_models is set explicitly.

    Args:
        yaml_file (Path): Path to the YAML pipeline file.
        max_threads (int | None): Maximum number of threads for operations.
    """
    cwd = os.getcwd()

    env_file = os.path.join(cwd, ".env")
    if os.path.exists(env_file):
        load_dotenv(env_file)

    import yaml as yaml_lib

    with open(yaml_file, "r") as f:
        config = yaml_lib.safe_load(f)

    optimizer_config = config.get("optimizer_config", {})
    if not optimizer_config:
        example_yaml = """optimizer_config:
  evaluation_file: evaluate.py    # or use judge_model instead (see below)
  metric_key: score
  save_dir: ./moar_results       # optional, defaults to temp dir
  max_iterations: 20              # optional, defaults to 20
  # judge_model: gpt-4.1          # LLM judge instead of evaluation_file
  # judge_criteria: "..."         # optional, auto-generated if omitted
  # available_models:             # optional, auto-detected from API keys
  #   - gpt-4.1
  #   - anthropic/claude-sonnet-4-6"""

        error_panel = Panel(
            f"[bold red]Error:[/bold red] optimizer_config section is required in YAML.\n\n"
            f"[bold]Example:[/bold]\n"
            f"[dim]{example_yaml}[/dim]",
            title="[bold red]Missing optimizer_config[/bold red]",
            border_style="red",
        )
        console.print(error_panel)
        raise typer.Exit(1)

    judge_cfg = optimizer_config.get("judge") or {}
    judge_model = optimizer_config.get("judge_model") or judge_cfg.get("model")
    judge_criteria = optimizer_config.get("judge_criteria") or judge_cfg.get(
        "criteria"
    )

    if judge_model and optimizer_config.get("evaluation_file"):
        console.print(
            Panel(
                "[bold red]Error:[/bold red] optimizer_config has both "
                "judge_model and evaluation_file — pick one way to score plans.",
                title="[bold red]Conflicting optimizer_config[/bold red]",
                border_style="red",
            )
        )
        raise typer.Exit(1)

    required_fields = {
        "evaluation_file": "Path to evaluation function file",
        "metric_key": "Key to extract from evaluation results",
    }

    missing_fields = (
        []
        if judge_model
        else [field for field in required_fields if not optimizer_config.get(field)]
    )
    if missing_fields:
        fields_table = Table(
            show_header=True, header_style="bold cyan", box=None, padding=(0, 2)
        )
        fields_table.add_column("Field", style="yellow")
        fields_table.add_column("Description", style="dim")

        for field, desc in required_fields.items():
            style = "bold red" if field in missing_fields else "dim"
            fields_table.add_row(f"[{style}]{field}[/{style}]", desc)

        missing_list = ", ".join([f"[bold red]{f}[/bold red]" for f in missing_fields])

        from rich.console import Group

        error_group = Group(
            f"[bold red]Missing required fields:[/bold red] {missing_list}\n",
            "[bold]Required fields:[/bold]",
            fields_table,
        )

        error_panel = Panel(
            error_group,
            title="[bold red]Missing Required Fields[/bold red]",
            border_style="red",
        )
        console.print(error_panel)
        raise typer.Exit(1)

    from docetl.moar.optimizer import MOAROptimizer
    from docetl.utils_evaluation import load_custom_evaluate_func

    try:
        # Resolve dataset path for wrapping the eval function
        dataset_path = optimizer_config.get("dataset_path")
        if not dataset_path:
            ds = config.get("datasets", {})
            if ds:
                _, ds_cfg = next(iter(ds.items()))
                dataset_path = ds_cfg.get("path", "")

        eval_fn = None
        if not judge_model:
            eval_fn = load_custom_evaluate_func(
                optimizer_config["evaluation_file"],
                dataset_path or "",
            )

        opt = MOAROptimizer(
            pipeline=str(yaml_file),
            eval_fn=eval_fn,
            metric_key=optimizer_config.get("metric_key"),
            judge_model=judge_model,
            judge_criteria=judge_criteria,
            models=optimizer_config.get("available_models"),
            agent_model=optimizer_config.get("rewrite_agent_model")
            or optimizer_config.get("model"),
            max_iterations=optimizer_config.get("max_iterations", 20),
            save_dir=optimizer_config.get("save_dir"),
            exploration_weight=optimizer_config.get("exploration_weight", 1.414),
            dataset_path=optimizer_config.get("dataset_path"),
        )
        result = opt.optimize()

        typer.echo("\n✅ MOAR optimization completed successfully!")
        typer.echo(f"   Frontier: {len(result.frontier)} pipelines")
        best = result.best()
        if best:
            typer.echo(
                f"   Best accuracy: {best.accuracy:.4f} (cost: ${best.cost:.4f})"
            )
        cheapest = result.cheapest()
        if cheapest and cheapest is not best:
            typer.echo(
                f"   Cheapest: ${cheapest.cost:.4f} (accuracy: {cheapest.accuracy:.4f})"
            )
        if result.save_dir:
            typer.echo(f"\n   Optimized pipelines saved to: {result.save_dir}/")
            for p in result.frontier:
                tag = ""
                if best and p is best:
                    tag = " (best accuracy)"
                elif cheapest and p is cheapest:
                    tag = " (cheapest)"
                typer.echo(f"     - {Path(p.yaml_path).name}{tag}")
    except Exception as e:
        typer.echo(f"Error running MOAR optimization: {e}", err=True)
        raise typer.Exit(1)


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
            package_dir = Path(pkg_resources.files("docetl"))
            for potential_source in (
                package_dir / "skills" / "docetl",  # installed wheel
                package_dir.parent / ".claude" / "skills" / "docetl",  # sdist/dev
            ):
                if potential_source.exists():
                    skill_source = potential_source
                    break
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
