#!/usr/bin/env python3
"""
Bootstrap evaluation script for computing 95% confidence intervals on accuracy metrics.

This script evaluates all JSON files in a given directory by:
1. Loading each JSON file containing prediction results
2. For each file, performing bootstrap sampling (1000 iterations)
3. Computing accuracy metrics for each bootstrap sample
4. Calculating 95% confidence intervals (2.5th and 97.5th percentiles)
5. Outputting results with confidence intervals

Usage:
    python bootstrap_evaluation.py <directory_path> [--dataset <dataset_name>] [--iterations <n>]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from collections import defaultdict


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Import evaluation functions from the existing codebase
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from experiments.reasoning.evaluation.utils import get_evaluate_func


def bootstrap_sample(data: List[Any], n_samples: int = None) -> List[Any]:
    """
    Create a bootstrap sample by sampling with replacement.
    
    Args:
        data: Original data list
        n_samples: Number of samples to draw (default: len(data))
    
    Returns:
        Bootstrap sample of the same size as original data
    """
    if n_samples is None:
        n_samples = len(data)
    
    indices = np.random.choice(len(data), size=n_samples, replace=True)
    return [data[i] for i in indices]


def save_bootstrap_sample_to_file(bootstrap_data: List[Any], temp_file_path: str):
    """Save bootstrap sample to a temporary JSON file."""
    with open(temp_file_path, 'w') as f:
        json.dump(bootstrap_data, f, indent=2, cls=NumpyEncoder)


def compute_bootstrap_confidence_interval(
    original_file_path: str,
    dataset_name: str,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Compute bootstrap confidence intervals for evaluation metrics.
    Only computes CI for the primary metric of each dataset.
    """
    
    # Dataset-to-primary-metric mapping
    dataset_metrics = {
        "cuad": "avg_f1",
        "blackvault": "avg_distinct_locations", 
        "game_reviews": "weighted_score",
        "medec": "combined_score",
        "sustainability": "combined_score",
    }
    
    primary_metric = dataset_metrics.get(dataset_name.lower())
    if not primary_metric:
        print(f"Unknown dataset '{dataset_name}' - no primary metric defined")
        return None
    # Load original data
    try:
        with open(original_file_path, 'r') as f:
            original_data = json.load(f)
    except Exception as e:
        print(f"Error loading {original_file_path}: {e}")
        return None
    
    if not original_data:
        print(f"Empty data in {original_file_path}")
        return None
    
    # Get evaluation function
    try:
        evaluate_func = get_evaluate_func(dataset_name)
    except ValueError as e:
        print(f"Unknown dataset '{dataset_name}': {e}")
        return None
    
    # Compute original metrics
    try:
        original_metrics = evaluate_func("original", original_file_path)
        if primary_metric not in original_metrics:
            print(f"Primary metric '{primary_metric}' not found in evaluation results")
            print(f"Available metrics: {list(original_metrics.keys())}")
            return None
    except Exception as e:
        print(f"Error evaluating original file {original_file_path}: {e}")
        return None
    
    # Bootstrap sampling
    bootstrap_metrics = []
    temp_file_path = f"{original_file_path}.bootstrap_temp.json"
    
    print(f"Performing {n_bootstrap} bootstrap iterations for {os.path.basename(original_file_path)}...")
    
    for i in range(n_bootstrap):
        if (i + 1) % 100 == 0:
            print(f"  Bootstrap iteration {i + 1}/{n_bootstrap}")
        
        # Create bootstrap sample
        bootstrap_data = bootstrap_sample(original_data)
        
        # Save to temporary file
        save_bootstrap_sample_to_file(bootstrap_data, temp_file_path)
        
        # Evaluate bootstrap sample
        try:
            bootstrap_result = evaluate_func("bootstrap", temp_file_path)
            bootstrap_metrics.append(bootstrap_result)
        except Exception as e:
            print(f"  Warning: Bootstrap iteration {i + 1} failed: {e}")
            continue
    
    # Clean up temporary file
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)
    
    if not bootstrap_metrics:
        print(f"No successful bootstrap iterations for {original_file_path}")
        return None
    
    # Compute confidence intervals (only for primary metric)
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    # Extract primary metric values from all bootstrap samples
    metric_values = []
    for bootstrap_result in bootstrap_metrics:
        if primary_metric in bootstrap_result:
            value = bootstrap_result[primary_metric]
            # Handle different value types (including numpy types)
            if isinstance(value, (int, float, np.integer, np.floating)):
                metric_values.append(float(value))  # Convert to regular Python float
            elif isinstance(value, dict) and 'value' in value:
                val = value['value']
                if isinstance(val, (int, float, np.integer, np.floating)):
                    metric_values.append(float(val))
                else:
                    metric_values.append(val)
    
    if not metric_values:
        print(f"No bootstrap samples produced valid values for primary metric '{primary_metric}'")
        return None
    
    # Calculate confidence interval for primary metric only
    lower_bound = float(np.percentile(metric_values, lower_percentile))
    upper_bound = float(np.percentile(metric_values, upper_percentile))
    mean_bootstrap = float(np.mean(metric_values))
    std_bootstrap = float(np.std(metric_values))
    
    # Ensure original value is also a regular Python type
    original_val = original_metrics.get(primary_metric, 'N/A')
    if isinstance(original_val, (np.integer, np.floating)):
        original_val = float(original_val)
    
    confidence_intervals = {
        primary_metric: {
            'original': original_val,
            'bootstrap_mean': mean_bootstrap,
            'bootstrap_std': std_bootstrap,
            'confidence_interval': [lower_bound, upper_bound],
            'confidence_level': confidence_level
        }
    }
    
    return {
        'file_path': original_file_path,
        'dataset': dataset_name,
        'primary_metric': primary_metric,
        'n_bootstrap_samples': len(bootstrap_metrics),
        'n_original_samples': len(original_data),
        'metrics': confidence_intervals
    }


def evaluate_directory(directory_path: str, dataset_name: str, n_bootstrap: int = 1000) -> Dict[str, Any]:
    """
    Evaluate all JSON files in a directory with bootstrap confidence intervals.
    
    Args:
        directory_path: Path to directory containing JSON files
        dataset_name: Dataset name for evaluation
        n_bootstrap: Number of bootstrap iterations
    
    Returns:
        Dictionary containing results for all files
    """
    directory = Path(directory_path)
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")
    
    # Find all JSON files
    json_files = list(directory.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {directory_path}")
        return {}
    
    print(f"Found {len(json_files)} JSON files in {directory_path}")
    
    results = {}
    
    for json_file in sorted(json_files):
        print(f"\nEvaluating {json_file.name}...")
        if json_file.name == "evaluation_metrics.json" or json_file.name == "experiment_summary.json" or json_file.name == "pareto_frontier.json": continue
        
        
        result = compute_bootstrap_confidence_interval(
            str(json_file),
            dataset_name,
            n_bootstrap
        )
        
        if result:
            results[json_file.name] = result
            print(f"  Completed bootstrap evaluation for {json_file.name}")
        else:
            print(f"  Failed to evaluate {json_file.name}")
    
    return results


def format_results_summary(results: Dict[str, Any]) -> str:
    """Format results into a readable summary."""
    if not results:
        return "No results to display."
    
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("BOOTSTRAP EVALUATION RESULTS SUMMARY")
    summary_lines.append("=" * 80)
    
    for filename, file_result in results.items():
        summary_lines.append(f"\nFile: {filename}")
        summary_lines.append(f"Dataset: {file_result['dataset']}")
        summary_lines.append(f"Primary metric: {file_result['primary_metric']}")
        summary_lines.append(f"Original samples: {file_result['n_original_samples']}")
        summary_lines.append(f"Bootstrap iterations: {file_result['n_bootstrap_samples']}")
        summary_lines.append("-" * 40)
        
        # Only one metric (primary metric) will be in the results
        for metric_name, metric_data in file_result['metrics'].items():
            original_val = metric_data['original']
            bootstrap_mean = metric_data['bootstrap_mean']
            ci_lower, ci_upper = metric_data['confidence_interval']
            confidence_level = metric_data['confidence_level']
            
            summary_lines.append(f"  {metric_name}:")
            summary_lines.append(f"    Original: {original_val}")
            summary_lines.append(f"    Bootstrap mean: {bootstrap_mean:.4f}")
            summary_lines.append(f"    {int(confidence_level*100)}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    summary_lines.append("\n" + "=" * 80)
    return "\n".join(summary_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap evaluation of JSON result files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bootstrap_evaluation.py ./results --dataset cuad
  python bootstrap_evaluation.py ./output --dataset blackvault --iterations 2000
  python bootstrap_evaluation.py ./experiments/results --dataset game_reviews
        """
    )
    
    parser.add_argument(
        "directory",
        help="Directory containing JSON result files to evaluate"
    )
    
    parser.add_argument(
        "--dataset",
        default="cuad",
        help="Dataset name for evaluation (default: cuad). Options: cuad, blackvault, game_reviews, medec, sustainability"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of bootstrap iterations (default: 1000)"
    )
    
    parser.add_argument(
        "--output",
        help="Output file path for detailed results (JSON format). If not specified, prints to stdout."
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for intervals (default: 0.95 for 95%% CI)"
    )
    
    args = parser.parse_args()
    
    # Validate confidence level
    if not 0 < args.confidence < 1:
        print("Error: Confidence level must be between 0 and 1")
        sys.exit(1)
    
    try:
        # Run bootstrap evaluation
        results = evaluate_directory(
            args.directory,
            args.dataset,
            args.iterations
        )
        
        if not results:
            print("No results generated.")
            sys.exit(1)
        
        # Save detailed results to file if specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)
            print(f"\nDetailed results saved to: {args.output}")
        
        # Print summary
        summary = format_results_summary(results)
        print(summary)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()