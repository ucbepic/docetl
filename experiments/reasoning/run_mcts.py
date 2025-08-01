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
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from docetl.mcts import MCTS, Node, ParetoFrontier, AccuracyComparator
from docetl.reasoning_optimizer.directives import (
    DEFAULT_MODEL, DEFAULT_OUTPUT_DIR, ALL_DIRECTIVES
)
from experiments.reasoning.evaluation.utils import run_dataset_evaluation, get_evaluate_func, get_dataset_stats

def run_mcts_experiment(
    yaml_path: str,
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
    
    # Load sample input data for accuracy comparator
    sample_input_data = {"document": "Sample contract document for comparison purposes..."}
    
    # Initialize accuracy comparator
    accuracy_comparator = AccuracyComparator(input_data=sample_input_data, model=model)
    
    # Use all registered rewrite directives from the central registry
    available_actions = set(ALL_DIRECTIVES)
    
    # Get dataset-specific evaluation function and statistics
    evaluate_func = get_evaluate_func(dataset)
    dataset_stats = get_dataset_stats(dataset, yaml_path)
    
    # Initialize MCTS
    mcts = MCTS(
        root_yaml_path=yaml_path,
        accuracy_comparator=accuracy_comparator,
        available_actions=available_actions,
        sample_input=sample_input_data,
        evaluate_func=evaluate_func,
        dataset_stats=dataset_stats,
        exploration_constant=exploration_weight,
        max_iterations=max_iterations,
        model=model,
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
        method_name="docetl_mcts"
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