"""
Helper functions for running MOAR optimizer from CLI.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

from docetl.console import DOCETL_CONSOLE
from docetl.moar import MOARSearch
from docetl.reasoning_optimizer.directives import ALL_DIRECTIVES
from docetl.utils_dataset import get_dataset_stats
from docetl.utils_evaluation import load_custom_evaluate_func


def infer_dataset_info(yaml_path: str, config: dict) -> tuple[str, str]:
    """
    Infer dataset path and name from YAML config.

    Args:
        yaml_path: Path to YAML file
        config: Full YAML config dictionary

    Returns:
        tuple: (dataset_path, dataset_name)

    Raises:
        ValueError: If datasets section is missing or empty
    """
    datasets = config.get("datasets", {})
    if not datasets:
        raise ValueError("YAML config must contain a 'datasets' section")

    # Get the first dataset (assuming single dataset per config)
    dataset_name, dataset_config = next(iter(datasets.items()))
    dataset_path = dataset_config.get("path")

    if not dataset_path:
        raise ValueError(f"Dataset '{dataset_name}' in config must have a 'path' field")

    # Resolve relative paths - try as-is first, then relative to YAML file location
    if Path(dataset_path).is_absolute():
        # Already absolute, use as-is
        pass
    elif Path(dataset_path).exists():
        # Path exists as-is (relative to current working directory)
        dataset_path = str(Path(dataset_path).resolve())
    else:
        # Try resolving relative to YAML file location
        yaml_dir = Path(yaml_path).parent
        resolved_path = yaml_dir / dataset_path
        if resolved_path.exists():
            dataset_path = str(resolved_path.resolve())
        else:
            # Use the resolved path anyway (might be created later)
            dataset_path = str(resolved_path.resolve())

    return dataset_path, dataset_name


def load_evaluation_function(config: dict, dataset_file_path: str):
    """
    Load evaluation function from optimizer_config.

    Args:
        config: optimizer_config dictionary from YAML
        dataset_file_path: Path to the dataset file

    Returns:
        callable: Evaluation function

    Raises:
        ValueError: If required parameters are missing
    """
    evaluation_file = config.get("evaluation_file")
    if not evaluation_file:
        raise ValueError(
            "optimizer_config must contain 'evaluation_file' (path to Python file with @docetl.register_eval decorated function)"
        )

    DOCETL_CONSOLE.log(
        f"[bold blue]📊 Loading evaluation function from: {evaluation_file}[/bold blue]"
    )
    evaluate_func = load_custom_evaluate_func(evaluation_file, dataset_file_path)
    DOCETL_CONSOLE.log("[green]✅ Evaluation function loaded[/green]")
    return evaluate_func


def run_moar_optimization(
    yaml_path: str,
    optimizer_config: dict,
) -> Dict[str, Any]:
    """
    Run MOAR optimization from CLI.

    Args:
        yaml_path: Path to the YAML pipeline file
        optimizer_config: optimizer_config dictionary from YAML (must contain save_dir)

    Returns:
        dict: Experiment summary
    """
    # Load full config to infer dataset info
    with open(yaml_path, "r") as f:
        full_config = yaml.safe_load(f)

    # Use dataset_path from optimizer_config if provided, otherwise infer from datasets section
    if optimizer_config.get("dataset_path"):
        dataset_path = optimizer_config.get("dataset_path")
        # Resolve relative paths
        if Path(dataset_path).is_absolute():
            dataset_path = str(Path(dataset_path).resolve())
        elif Path(dataset_path).exists():
            dataset_path = str(Path(dataset_path).resolve())
        else:
            yaml_dir = Path(yaml_path).parent
            dataset_path = str((yaml_dir / dataset_path).resolve())
        # Infer dataset name from datasets section
        _, dataset_name = infer_dataset_info(yaml_path, full_config)
    else:
        # Infer both dataset path and name from config
        dataset_path, dataset_name = infer_dataset_info(yaml_path, full_config)

    # Extract MOAR parameters from optimizer_config (all required, no defaults)
    save_dir = optimizer_config.get("save_dir")
    if not save_dir:
        raise ValueError("optimizer_config must contain 'save_dir' for MOAR optimizer")

    available_models = optimizer_config.get("available_models")
    if not available_models:
        raise ValueError(
            "optimizer_config must contain 'available_models' (list of model names) for MOAR optimizer"
        )

    evaluation_file = optimizer_config.get("evaluation_file")
    if not evaluation_file:
        raise ValueError(
            "optimizer_config must contain 'evaluation_file' (path to Python file with @docetl.register_eval decorated function) for MOAR optimizer"
        )

    metric_key = optimizer_config.get("metric_key")
    if not metric_key:
        raise ValueError(
            "optimizer_config must contain 'metric_key' (key to extract from evaluation results) for MOAR optimizer"
        )

    max_iterations = optimizer_config.get("max_iterations")
    if max_iterations is None:
        raise ValueError(
            "optimizer_config must contain 'max_iterations' (number of MOARSearch iterations) for MOAR optimizer"
        )

    model = optimizer_config.get("model")
    if not model:
        raise ValueError(
            "optimizer_config must contain 'model' (LLM model name for directive instantiation) for MOAR optimizer"
        )

    # Optional parameters
    exploration_weight = optimizer_config.get("exploration_weight", 1.414)
    build_first_layer = optimizer_config.get("build_first_layer", False)

    # Resolve save directory (handle relative paths)
    save_dir = Path(save_dir)
    if not save_dir.is_absolute():
        # Resolve relative to current working directory
        save_dir = Path.cwd() / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    DOCETL_CONSOLE.log("[bold blue]🌳 Running MOARSearch[/bold blue]")
    DOCETL_CONSOLE.log(f"[dim]Input Pipeline:[/dim] {yaml_path}")
    DOCETL_CONSOLE.log(f"[dim]Save Directory:[/dim] {save_dir}")
    DOCETL_CONSOLE.log(f"[dim]Max Iterations:[/dim] {max_iterations}")
    DOCETL_CONSOLE.log(f"[dim]Exploration Weight (c):[/dim] {exploration_weight}")
    DOCETL_CONSOLE.log(f"[dim]Model:[/dim] {model}")
    DOCETL_CONSOLE.log(f"[dim]Dataset:[/dim] {dataset_name}")
    DOCETL_CONSOLE.log()

    # Load sample input data
    DOCETL_CONSOLE.log("[bold blue]🚀 Initializing MOARSearch...[/bold blue]")
    with open(dataset_path, "r") as f:
        dataset_data = json.load(f)

    # Take only the first 5 documents for sample input
    if isinstance(dataset_data, list):
        sample_input_data = dataset_data[:5]
    else:
        sample_input_data = dataset_data

    # Use all registered rewrite directives
    available_actions = set(ALL_DIRECTIVES)

    # Get dataset statistics
    dataset_stats = get_dataset_stats(yaml_path, dataset_name)

    # Load evaluation function (pass dataset_path so it can be provided to eval function)
    evaluate_func = load_evaluation_function(optimizer_config, dataset_path)

    # Initialize MOARSearch
    moar = MOARSearch(
        root_yaml_path=yaml_path,
        available_actions=available_actions,
        sample_input=sample_input_data,
        dataset_stats=dataset_stats,
        dataset_name=dataset_name,
        available_models=available_models,
        evaluate_func=evaluate_func,
        exploration_constant=exploration_weight,
        max_iterations=max_iterations,
        model=model,
        output_dir=str(save_dir),
        build_first_layer=build_first_layer,
        custom_metric_key=metric_key,
        sample_dataset_path=dataset_path,  # Use the dataset_path (which may be from optimizer_config)
    )

    DOCETL_CONSOLE.log(
        f"[green]✅ MOARSearch initialized with root node: {yaml_path}[/green]"
    )

    # Run MOARSearch optimization
    DOCETL_CONSOLE.log(
        f"[bold blue]\n🔍 Running MOARSearch optimization for {max_iterations} iterations...[/bold blue]"
    )
    start_time = datetime.now()
    best_nodes = moar.search()
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    DOCETL_CONSOLE.log(
        f"[green]✅ MOARSearch optimization completed in {duration:.2f} seconds[/green]"
    )

    # Run evaluation
    DOCETL_CONSOLE.log("[bold blue]📊 Running evaluation...[/bold blue]")

    # Prepare nodes for evaluation
    nodes_for_evaluation = []
    for n in moar.pareto_frontier.plans:
        n.moar_accuracy = moar.pareto_frontier.plans_accuracy.get(n)
        n.on_frontier = n in moar.pareto_frontier.frontier_plans
        nodes_for_evaluation.append(n)

    from docetl.utils_evaluation import run_evaluation

    eval_results = run_evaluation(
        nodes_or_files=nodes_for_evaluation,
        evaluate_func=evaluate_func,
        metric_key=metric_key,
        output_path=save_dir,
        dataset_name=dataset_name,
    )

    # Save experiment summary
    results = {
        "optimizer": "moar",
        "input_pipeline": yaml_path,
        "model": model,
        "max_iterations": max_iterations,
        "exploration_weight": exploration_weight,
        "save_dir": str(save_dir),
        "dataset": dataset_name,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration,
        "num_best_nodes": len(best_nodes) if best_nodes else 0,
        "total_nodes_explored": (
            len(moar.all_nodes) if hasattr(moar, "all_nodes") else 0
        ),
        "total_search_cost": (
            moar.total_search_cost if hasattr(moar, "total_search_cost") else 0
        ),
    }

    if eval_results:
        results["evaluation_file"] = str(save_dir / "evaluation_metrics.json")

    # Save Pareto frontier if available
    if hasattr(moar, "pareto_frontier") and moar.pareto_frontier.frontier_plans:
        pareto_file = save_dir / "pareto_frontier.json"
        pareto_data = []
        for node in moar.pareto_frontier.frontier_plans:
            pareto_data.append(
                {
                    "node_id": node.get_id(),
                    "yaml_path": node.yaml_file_path,
                    "cost": node.cost,
                    "accuracy": moar.pareto_frontier.plans_accuracy.get(node),
                }
            )
        with open(pareto_file, "w") as f:
            json.dump(pareto_data, f, indent=2)
        results["pareto_frontier_file"] = str(pareto_file)

    # Save experiment summary
    summary_file = save_dir / "experiment_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    DOCETL_CONSOLE.log(f"[green]✅ Experiment summary saved to: {summary_file}[/green]")

    return results
