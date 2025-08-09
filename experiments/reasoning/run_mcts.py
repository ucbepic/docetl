#!/usr/bin/env python3
"""
MCTS Experiment Runner

This script runs the MCTS-based optimization for DocETL pipelines.
It's extracted from the MCTS folder to provide a clean experiment interface.
"""

import os
import json
import argparse
import glob
from pathlib import Path
from datetime import datetime

from docetl.mcts import MCTS, Node, ParetoFrontier, AccuracyComparator
from docetl.reasoning_optimizer.directives import (
    DEFAULT_MODEL, DEFAULT_OUTPUT_DIR, ALL_DIRECTIVES
)
from experiments.reasoning.evaluation.utils import run_dataset_evaluation, get_evaluate_func, get_dataset_stats
# Modal integration
import modal
import yaml

app = modal.App("docetl-mcts")

# Build image with project deps and local sources for experiments (docetl installed from pyproject)
image = (
    modal.Image.debian_slim(python_version="3.10")
    # .pip_install("poetry")
    # .poetry_install_from_file("pyproject.toml", ignore_lockfile=True)
    .add_local_file("pyproject.toml", "/pyproject.toml", copy=True)
    .add_local_file("poetry.lock", "/poetry.lock", copy=True)
    .add_local_file("README.md", "/README.md", copy=True)
    # .add_local_python_source("docetl", "server", copy=True)
    .add_local_dir("docetl", remote_path="/docetl", copy=True)
    .pip_install("poetry")
    .run_commands(["poetry config virtualenvs.create false", "poetry install --all-extras --no-root && poetry install --all-extras"])
    .pip_install("matplotlib", "Levenshtein", "nltk")
    .add_local_python_source("experiments", ignore=["**/.venv/*"])
    .add_local_python_source("docetl", ignore=["**/.venv/*"])
)

# Named volume for datasets and outputs
VOLUME_NAME = "docetl-ro-experiments"
VOLUME_MOUNT_PATH = "/mnt/docetl-ro-experiments"
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _resolve_in_volume(path: str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((Path(VOLUME_MOUNT_PATH) / p).resolve())


def _rewrite_pipeline_yaml_for_modal(orig_yaml_path: str, experiment_name: str) -> str:
    with open(orig_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    base_mount = Path(VOLUME_MOUNT_PATH)

    def make_abs(p: str) -> str:
        pp = Path(p)
        if pp.is_absolute():
            return str(pp)
        return str(base_mount / p)

    # Rewrite pipeline output path(s)
    pipeline_cfg = cfg.get("pipeline", {})
    output_root = base_mount / "outputs" / experiment_name
    if isinstance(pipeline_cfg, dict):
        out = pipeline_cfg.get("output")
        if isinstance(out, dict):
            if isinstance(out.get("path"), str):
                original_name = Path(out["path"]).name
                output_root.mkdir(parents=True, exist_ok=True)
                out["path"] = str(output_root / original_name)
            if isinstance(out.get("intermediate_dir"), str):
                out["intermediate_dir"] = str(output_root / "intermediates")

    # Save rewritten YAML into volume tmp
    tmp_dir = base_mount / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    new_yaml_path = tmp_dir / f"{Path(orig_yaml_path).stem}_modal.yaml"
    with open(new_yaml_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return str(new_yaml_path)


@app.function(image=image, secrets=[modal.Secret.from_dotenv()], volumes={VOLUME_MOUNT_PATH: volume}, timeout=60 * 60 * 12)
def run_mcts_remote(
    yaml_path: str,
    dataset_path: str,
    data_dir: str | None = None,
    output_dir: str | None = None,
    experiment_name: str = "mcts_experiment",
    max_iterations: int = 100,
    exploration_weight: float = 1.414,
    model: str = DEFAULT_MODEL,
    dataset: str = "cuad",
    ground_truth_path: str | None = None,
):
    os.environ["EXPERIMENT_OUTPUT_DIR"] = str(Path(VOLUME_MOUNT_PATH) / "outputs")
    resolved_output_dir = _resolve_in_volume(output_dir) if output_dir else None
    # resolved_yaml_path = _resolve_in_volume(yaml_path)
    # resolved_dataset_path = _resolve_in_volume(dataset_path)
    # resolved_data_dir = _resolve_in_volume(data_dir) if data_dir else None
    # resolved_ground_truth = _resolve_in_volume(ground_truth_path) if ground_truth_path else None

    # Write a temporary YAML with dataset/output paths rewritten into the mounted volume
    modal_yaml_path = _rewrite_pipeline_yaml_for_modal(yaml_path, experiment_name)

    results = run_mcts_experiment(
        yaml_path=modal_yaml_path,
        dataset_path=dataset_path,
        data_dir=data_dir,
        output_dir=resolved_output_dir,
        experiment_name=experiment_name,
        max_iterations=max_iterations,
        exploration_weight=exploration_weight,
        model=model,
        dataset=dataset,
        ground_truth_path=ground_truth_path,
    )
    volume.commit()
    return results


@app.local_entrypoint()
def modal_main(
    yaml_path: str,
    dataset_path: str,
    experiment_name: str,
    data_dir: str | None = None,
    output_dir: str | None = None,
    max_iterations: int = 100,
    exploration_weight: float = 1.414,
    model: str = DEFAULT_MODEL,
    dataset: str = "cuad",
    ground_truth: str | None = None,
):
    run_mcts_remote.remote(
        yaml_path=yaml_path,
        dataset_path=dataset_path,
        data_dir=data_dir,
        output_dir=output_dir,
        experiment_name=experiment_name,
        max_iterations=max_iterations,
        exploration_weight=exploration_weight,
        model=model,
        dataset=dataset,
        ground_truth_path=ground_truth,
    )


def run_mcts_experiment(
    yaml_path: str,
    dataset_path: str,
    data_dir: str = None,
    output_dir: str = None, 
    experiment_name: str = "mcts_experiment",
    max_iterations: int = 100,
    exploration_weight: float = 1.414,
    model: str = DEFAULT_MODEL,
    dataset: str = "cuad",
    ground_truth_path: str | None = None,
):
    """
    Run MCTS optimization experiment with specified parameters.
    
    Args:
        yaml_path: Path to the input YAML pipeline file
        dataset_path: Path to the dataset file for sample input data
        data_dir: Directory containing input data files
        output_dir: Directory to save experiment outputs
        experiment_name: Name for this experiment run
        max_iterations: Maximum MCTS iterations
        exploration_weight: UCB exploration parameter (c)
        model: LLM model to use for directive instantiation
    """
    # Set up environment
    if data_dir:
        os.environ['EXPERIMENT_DATA_DIR'] = data_dir
    # Determine output directory (env var, parameter, or default)
    if output_dir is None:
        output_dir = os.environ.get('EXPERIMENT_OUTPUT_DIR', DEFAULT_OUTPUT_DIR)
    # Create output directory
    output_path = Path(output_dir) / experiment_name
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"🌳 Running MCTS Optimization Experiment")
    print(f"=" * 50)
    print(f"Input Pipeline: {yaml_path}")
    print(f"Data Directory: {data_dir or 'default'}")
    print(f"Output Directory: {output_path}")
    print(f"Max Iterations: {max_iterations}")
    print(f"Exploration Weight (c): {exploration_weight}")
    print(f"Model: {model}")
    print(f"Experiment: {experiment_name}")
    print(f"Dataset for evaluation: {dataset}")
    print()
    # Initialize MCTS
    print("🚀 Initializing MCTS...")
    
    # Load sample input data for accuracy comparator from dataset_path
    with open(dataset_path, 'r') as f:
        dataset_data = json.load(f)
    
    # Take only the first 5 documents
    if isinstance(dataset_data, list):
        sample_input_data = dataset_data[:5]
    else:
        sample_input_data = dataset_data
    
    # Initialize accuracy comparator
    accuracy_comparator = AccuracyComparator(input_data=sample_input_data, model=model)
    
    # Use all registered rewrite directives from the central registry
    available_actions = set(ALL_DIRECTIVES)
    
    # Get dataset statistics
    dataset_stats = get_dataset_stats(dataset, yaml_path)
    
    # Initialize MCTS
    mcts = MCTS(
        root_yaml_path=yaml_path,
        accuracy_comparator=accuracy_comparator,
        available_actions=available_actions,
        sample_input=sample_input_data,
        dataset_stats=dataset_stats,
        dataset_name=dataset,
        exploration_constant=exploration_weight,
        max_iterations=max_iterations,
        model=model,
        output_dir=str(output_path),
    )
    print(f"✅ MCTS initialized with root node: {yaml_path}")
    # Run MCTS optimization
    print(f"\n🔍 Running MCTS optimization for {max_iterations} iterations...")
    start_time = datetime.now()
    best_nodes = mcts.search()
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"✅ MCTS optimization completed in {duration:.2f} seconds")
    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    # Prepare nodes with MCTS-specific attributes for evaluation
    nodes_for_evaluation = []
    for n in mcts.pareto_frontier.plans:
        # Add MCTS-specific attributes to the node
        n.mcts_accuracy = mcts.pareto_frontier.plans_accuracy.get(n)
        n.on_frontier = n in mcts.pareto_frontier.frontier_plans
        nodes_for_evaluation.append(n)
    eval_results, pareto_auc = run_dataset_evaluation(
        dataset=dataset,
        nodes_or_files=nodes_for_evaluation,
        output_path=output_path,
        ground_truth_path=ground_truth_path,
        method_name="docetl_mcts",
        root_cost=mcts.root.cost
    )
    # Save results
    results = {
        "experiment_name": experiment_name,
        "input_pipeline": yaml_path,
        "model": model,
        "max_iterations": max_iterations,
        "exploration_weight": exploration_weight,
        "data_dir": data_dir,
        "output_dir": str(output_path),
        "dataset": dataset,
        "ground_truth": ground_truth_path,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration,
        "num_best_nodes": len(best_nodes) if best_nodes else 0,
        "total_nodes_explored": len(mcts.all_nodes) if hasattr(mcts, 'all_nodes') else 0,
    }
    # Add Pareto AUC if it was computed
    if pareto_auc is not None:
        results["pareto_auc"] = pareto_auc
    if eval_results:
        results["evaluation_file"] = str(output_path / "evaluation_metrics.json")
    # Save Pareto frontier if available
    if hasattr(mcts, 'pareto_frontier') and mcts.pareto_frontier.frontier_plans:
        pareto_file = output_path / "pareto_frontier.json"
        pareto_data = []
        
        for solution in mcts.pareto_frontier.frontier_plans:
            pareto_data.append({
                "accuracy": mcts.pareto_frontier.plans_accuracy.get(solution, None),
                "cost": getattr(solution, 'cost', None),
                "value": getattr(solution, 'value', 0),
                "config_path": getattr(solution, 'yaml_file_path', None)
            })
        
        with open(pareto_file, 'w') as f:
            json.dump(pareto_data, f, indent=2)
        
        results["pareto_frontier_file"] = str(pareto_file)
        print(f"📈 Pareto frontier saved to: {pareto_file}")
    # Save experiment summary
    summary_file = output_path / "experiment_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📋 Experiment Summary:")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Best Configs Found: {results['num_best_nodes']}")
    print(f"   Summary saved to: {summary_file}")
    print(f"   All outputs in: {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run MCTS reasoning optimization experiment", 
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic MCTS run
  python run_mcts.py --yaml_path ./pipeline.yaml --experiment_name mcts_test
  
  # With custom parameters
  python run_mcts.py --yaml_path ./pipeline.yaml --data_dir ./data --output_dir ./results --max_iterations 200 --experiment_name mcts_deep
  
  # High exploration
  python run_mcts.py --yaml_path ./pipeline.yaml --exploration_weight 2.0 --experiment_name mcts_explore
        """
    )
    
    parser.add_argument("--yaml_path", type=str, required=True,
                       help="Path to the input YAML pipeline file")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to the dataset file for sample input data")
    parser.add_argument("--data_dir", type=str,
                       help="Directory containing input data files (sets EXPERIMENT_DATA_DIR)")
    parser.add_argument("--output_dir", type=str,
                       help=f"Directory to save experiment outputs (default: EXPERIMENT_OUTPUT_DIR env var or {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--experiment_name", type=str, required=True,
                       help="Name for this experiment run")
    parser.add_argument("--max_iterations", type=int, default=100,
                       help="Maximum MCTS iterations (default: 100)")
    parser.add_argument("--exploration_weight", type=float, default=1.414,
                       help="UCB exploration parameter c (default: 1.414)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, 
                       help=f"LLM model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--dataset", type=str, default="cuad", help="Dataset name for evaluation (default: cuad)")
    parser.add_argument("--ground_truth", type=str, help="Path to ground-truth file (if not default)")
    
    args = parser.parse_args()
    
    result = run_mcts_experiment(
        yaml_path=args.yaml_path,
        dataset_path=args.dataset_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        max_iterations=args.max_iterations,
        exploration_weight=args.exploration_weight,
        model=args.model,
        dataset=args.dataset,
        ground_truth_path=args.ground_truth,
    )
    
    print("\n🎉 MCTS experiment completed successfully!")


if __name__ == "__main__":
    main()