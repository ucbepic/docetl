"""
MOARSearch Experiment Runner

This script runs the MOARSearch-based optimization for DocETL pipelines.
"""

import os
import json
import argparse
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

from docetl.moar import MOARSearch, Node, ParetoFrontier, AVAILABLE_MODELS
from docetl.reasoning_optimizer.directives import (
    DEFAULT_MODEL, DEFAULT_OUTPUT_DIR, ALL_DIRECTIVES
)
from experiments.reasoning.evaluation.utils import run_dataset_evaluation, get_dataset_stats, load_custom_evaluate_func
import modal
import yaml
from experiments.reasoning.utils import app, volume, VOLUME_MOUNT_PATH, image



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
def run_moar_remote(
    yaml_path: str,
    dataset_path: str,
    data_dir: str | None = None,
    output_dir: str | None = None,
    experiment_name: str = "moar_experiment",
    max_iterations: int = 40,
    exploration_weight: float = 1.414,
    model: str = DEFAULT_MODEL,
    dataset: str = "cuad",
    ground_truth_path: str | None = None,
    original_query_result: Dict[str, Any] | None = None,
    build_first_layer: Optional[bool] = False,
    available_models: List[str] | None = None,
    accuracy_function: str | None = None,
    accuracy_metric_key: str | None = None,
):
    os.environ["EXPERIMENT_OUTPUT_DIR"] = str(Path(VOLUME_MOUNT_PATH) / "outputs")
    resolved_output_dir = _resolve_in_volume(output_dir) if output_dir else None
    
    # Write a temporary YAML with dataset/output paths rewritten into the mounted volume
    modal_yaml_path = _rewrite_pipeline_yaml_for_modal(yaml_path, experiment_name)

    results = run_moar_experiment(
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
        original_query_result=original_query_result,
        build_first_layer=build_first_layer,
        available_models=available_models,
        accuracy_function=accuracy_function,
        accuracy_metric_key=accuracy_metric_key,
    )
    volume.commit()
    return results


@app.local_entrypoint()
def modal_main_moar(
    yaml_path: str,
    dataset_path: str,
    experiment_name: str,
    data_dir: str | None = None,
    output_dir: str | None = None,
    max_iterations: int = 40,
    exploration_weight: float = 1.414,
    model: str = DEFAULT_MODEL,
    dataset: str = "cuad",
    ground_truth: str | None = None,
    original_query_result: Dict[str, Any] | None = None,
    available_models: List[str] | None = None,
    accuracy_function: str | None = None,
    accuracy_metric_key: str | None = None,
):
    run_moar_remote.remote(
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
        original_query_result=original_query_result,
        available_models=available_models,
        accuracy_function=accuracy_function,
        accuracy_metric_key=accuracy_metric_key,
    )


def run_moar_experiment(
    yaml_path: str,
    dataset_path: str,
    data_dir: str = None,
    output_dir: str = None, 
    experiment_name: str = "moar_experiment",
    max_iterations: int = 40,
    exploration_weight: float = 1.414,
    model: str = DEFAULT_MODEL,
    dataset: str = "cuad",
    ground_truth_path: str | None = None,
    original_query_result: Dict[str, Any] | None = None,
    build_first_layer: Optional[bool] = False,
    available_models: List[str] | None = None,
    accuracy_function: str | None = None,
    accuracy_metric_key: str | None = None,
):
    """
    Run MOARSearch-based optimization experiment with specified parameters.
    
    Args:
        yaml_path: Path to the input YAML pipeline file
        dataset_path: Path to the dataset file for sample input data
        data_dir: Directory containing input data files
        output_dir: Directory to save experiment outputs
        experiment_name: Name for this experiment run
        max_iterations: Maximum optimization search iterations
        exploration_weight: UCB exploration parameter (c)
        model: LLM model to use for directive instantiation
        dataset: Dataset name for evaluation
        ground_truth_path: Path to ground-truth file (if not default)
        original_query_result: Original query result (if not default)
        build_first_layer: Whether to build the first layer of the tree 
        available_models: List of available models for operators
        accuracy_function: Path to Python file containing custom evaluate_results function
        accuracy_metric_key: Key to extract from evaluation results dict for accuracy metric
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
    print(f"🌳 Running MOARSearch Optimization")
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
    # Initialize MOARSearch
    print("🚀 Initializing MOARSearch...")
    
    # Load sample input data for accuracy comparator from dataset_path
    with open(dataset_path, 'r') as f:
        dataset_data = json.load(f)
    
    # Take only the first 5 documents
    if isinstance(dataset_data, list):
        sample_input_data = dataset_data[:5]
    else:
        sample_input_data = dataset_data
    
    
    # Use all registered rewrite directives from the central registry
    available_actions = set(ALL_DIRECTIVES)
    
    # Get dataset statistics
    dataset_stats = get_dataset_stats(dataset, yaml_path)

    # Use provided available_models or default list
    if available_models is None:
        available_models = AVAILABLE_MODELS
    
    # Load evaluation function
    if accuracy_function:
        if not accuracy_metric_key:
            raise ValueError("--accuracy_metric_key must be provided when using --accuracy_function")
        print(f"📊 Loading custom accuracy function from: {accuracy_function}")
        evaluate_func = load_custom_evaluate_func(accuracy_function)
        print(f"✅ Custom accuracy function loaded. Metric key: {accuracy_metric_key}")
    else:
        # Use default evaluation function from experiments
        from experiments.reasoning.evaluation.utils import get_evaluate_func, dataset_accuracy_metrics
        evaluate_func = get_evaluate_func(dataset, mode="train")
        # Get the default metric key for this dataset
        accuracy_metric_key = dataset_accuracy_metrics.get(dataset.lower(), "accuracy")
    
    # Initialize MOARSearch
    moar = MOARSearch(
        root_yaml_path=yaml_path,
        available_actions=available_actions,
        sample_input=sample_input_data,
        dataset_stats=dataset_stats,
        dataset_name=dataset,
        available_models=available_models,
        evaluate_func=evaluate_func,
        exploration_constant=exploration_weight,
        max_iterations=max_iterations,
        model=model,
        output_dir=str(output_path),
        build_first_layer=build_first_layer,
        custom_metric_key=accuracy_metric_key,
    )
    print(f"✅ MOARSearch initialized with root node: {yaml_path}")
    # Run MOARSearch optimization
    print(f"\n🔍 Running MOARSearch optimization for {max_iterations} iterations...")
    start_time = datetime.now()
    best_nodes = moar.search()
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"✅ MOARSearch optimization completed in {duration:.2f} seconds")
    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    # Prepare nodes with MOARSearch-specific attributes for evaluation
    nodes_for_evaluation = []
    for n in moar.pareto_frontier.plans:
        # Add MOARSearch-specific attributes to the node
        n.moar_accuracy = moar.pareto_frontier.plans_accuracy.get(n)
        n.on_frontier = n in moar.pareto_frontier.frontier_plans
        nodes_for_evaluation.append(n)
    # Use original query result cost if available, otherwise use MOARSearch root cost
    root_cost = original_query_result["cost"] if original_query_result and original_query_result["success"] else moar.root.cost
    
    eval_results, pareto_auc = run_dataset_evaluation(
        dataset=dataset,
        nodes_or_files=nodes_for_evaluation,
        output_path=output_path,
        ground_truth_path=ground_truth_path,
        method_name="docetl_moar",
        root_cost=root_cost,
        custom_evaluate_func=evaluate_func,
        custom_metric_key=accuracy_metric_key,
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
        "total_nodes_explored": len(moar.all_nodes) if hasattr(moar, 'all_nodes') else 0,
        "original_query_cost": original_query_result["cost"] if original_query_result and original_query_result["success"] else None,
        "original_query_success": original_query_result["success"] if original_query_result else None,
    }
    # Add Pareto AUC if it was computed
    if pareto_auc is not None:
        results["pareto_auc"] = pareto_auc
    if eval_results:
        results["evaluation_file"] = str(output_path / "evaluation_metrics.json")
    # Save Pareto frontier if available
    if hasattr(moar, 'pareto_frontier') and moar.pareto_frontier.frontier_plans:
        pareto_file = output_path / "pareto_frontier.json"
        pareto_data = []
        
        for solution in moar.pareto_frontier.frontier_plans:
            pareto_data.append({
                "accuracy": moar.pareto_frontier.plans_accuracy.get(solution, None),
                "cost": getattr(solution, 'cost', None),
                "value": getattr(solution, 'value', 0),
                "config_path": str(getattr(solution, 'yaml_file_path', None)) if getattr(solution, 'yaml_file_path', None) is not None else None
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
        description="Run MOARSearch reasoning optimization experiment", 
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic MOARSearch run
  python run_moar.py --yaml_path ./pipeline.yaml --experiment_name moar_test
  
  # With custom parameters
  python run_moar.py --yaml_path ./pipeline.yaml --data_dir ./data --output_dir ./results --max_iterations 200 --experiment_name moar_deep
  
  # High exploration
  python run_moar.py --yaml_path ./pipeline.yaml --exploration_weight 2.0 --experiment_name moar_explore
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
    parser.add_argument("--max_iterations", type=int, default=40,
                       help="Maximum MOARSearch iterations (default: 40)")
    parser.add_argument("--exploration_weight", type=float, default=1.414,
                       help="UCB exploration parameter c (default: 1.414)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, 
                       help=f"LLM model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--dataset", type=str, default="cuad", help="Dataset name for evaluation (default: cuad)")
    parser.add_argument("--ground_truth", type=str, help="Path to ground-truth file (if not default)")
    parser.add_argument("--available_models", type=str, nargs="+", 
                       help="List of available models for first layer (default: all models). Example: --available_models gpt-5 gpt-5-mini gpt-4o")
    parser.add_argument("--accuracy_function", type=str,
                       help="Path to Python file containing custom evaluate_results function for user datasets")
    parser.add_argument("--accuracy_metric_key", type=str,
                       help="Key to extract from evaluation results dict for accuracy metric (required with --accuracy_function)")
    
    args = parser.parse_args()
    
    # Validate: if no custom accuracy function, dataset must be a supported one
    if not args.accuracy_function:
        supported_datasets = {"cuad", "blackvault", "medec", "biodex", "sustainability", "game_reviews", "facility"}
        if args.dataset.lower() not in supported_datasets:
            parser.error(f"Dataset '{args.dataset}' is not supported. Use --accuracy_function for custom datasets, or choose from: {', '.join(sorted(supported_datasets))}")
    
    result = run_moar_experiment(
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
        available_models=args.available_models,
        accuracy_function=args.accuracy_function,
        accuracy_metric_key=args.accuracy_metric_key,
    )
    
    print("\n🎉 MOARSearch experiment completed successfully!")


if __name__ == "__main__":
    main()