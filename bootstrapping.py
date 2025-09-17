#!/usr/bin/env python3
"""
Bootstrap sampling script for evaluation metrics visualization.

This script:
1. Loads evaluation_metrics.json files from modal paths
2. For each entry, finds the corresponding JSON file
3. Performs bootstrap sampling 10 times to get 10 different accuracy and cost estimates
4. Estimates per-document cost based on word count
5. Visualizes results with different colors/shapes for each entry

Usage:
    python bootstrapping.py <dataset_name> [--modal_path <path>] [--bootstrap_iterations <n>]
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import random
from collections import defaultdict
from tqdm import tqdm
# Add the project root to the path to import evaluation functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from experiments.reasoning.evaluation.utils import get_evaluate_func
from experiments.reasoning.utils import app, volume, VOLUME_MOUNT_PATH, image


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


def count_words_in_document(doc: Any) -> int:
    """
    Count words in a document (handles various data types).
    
    Args:
        doc: Document data (dict, list, string, etc.)
    
    Returns:
        Word count
    """
    if isinstance(doc, dict):
        # Convert dict to string representation
        text = json.dumps(doc, separators=(',', ':'))
    elif isinstance(doc, list):
        # Convert list to string representation
        text = json.dumps(doc, separators=(',', ':'))
    elif isinstance(doc, str):
        text = doc
    else:
        text = str(doc)
    
    # Remove JSON syntax characters and count words
    text = text.replace('{', '').replace('}', '').replace('[', '').replace(']', '')
    text = text.replace('"', '').replace(',', ' ').replace(':', ' ')
    words = text.split()
    return len([word for word in words if word.strip()])


def estimate_per_doc_cost(data: List[Any], original_cost: float, original_data: List[Any]) -> float:
    """
    Estimate per-document cost based on word count, using original cost as reference.
    
    Args:
        data: List of documents (bootstrap sample)
        original_cost: Original cost from evaluation metrics
        original_data: Original data to calculate cost per word
    
    Returns:
        Estimated cost per document for the bootstrap sample
    """
    if not data or not original_data:
        return 0.0
    
    # Calculate total words in original data
    original_total_words = sum(count_words_in_document(doc) for doc in original_data)
    if original_total_words == 0:
        return 0.0
    
    # Calculate cost per word from original data
    cost_per_word = original_cost / original_total_words
    
    # Calculate total words in bootstrap sample
    bootstrap_total_words = sum(count_words_in_document(doc) for doc in data)
    
    # Estimate cost for bootstrap sample
    return bootstrap_total_words * cost_per_word


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


def load_json_file(file_path: str) -> Optional[List[Any]]:
    """
    Load JSON file and return its contents.
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        JSON data or None if loading fails
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def find_json_file_in_modal_paths(entry: Dict[str, Any], dataset: str) -> Optional[str]:
    """
    Find the corresponding JSON file for an evaluation entry.
    
    Args:
        entry: Evaluation metrics entry
        dataset: Dataset name
    
    Returns:
        Path to the JSON file or None if not found
    """
    file_name = entry.get("file")
    if not file_name:
        return None
    
    # Common modal output paths to search
    search_paths = [
        Path(VOLUME_MOUNT_PATH) / f"outputs/{dataset}_mcts",
        Path(VOLUME_MOUNT_PATH) / f"outputs/{dataset}",  
    ]
    
    # Search in each potential path
    for base_path in search_paths:
        full_path = base_path / file_name
        if full_path.exists():
            return str(full_path)
    
    # Also try relative to current directory
    if os.path.exists(file_name):
        return file_name
    
    return None


def bootstrap_evaluation_entry(
    entry: Dict[str, Any], 
    dataset: str, 
    n_bootstrap: int = 1000
) -> Optional[Dict[str, Any]]:
    """
    Perform bootstrap sampling for a single evaluation entry.
    
    Args:
        entry: Evaluation metrics entry
        dataset: Dataset name
        n_bootstrap: Number of bootstrap iterations
    
    Returns:
        Dictionary with bootstrap results or None if failed
    """
    # Find the corresponding JSON file
    json_file_path = find_json_file_in_modal_paths(entry, dataset)
    if not json_file_path:
        print(f"Could not find JSON file for entry: {entry.get('file')}")
        return None
    
    # Load the original data
    original_data = load_json_file(json_file_path)
    if not original_data:
        return None
    
    # Get evaluation function
    try:
        evaluate_func = get_evaluate_func(dataset)
    except ValueError as e:
        print(f"Unknown dataset '{dataset}': {e}")
        return None
    
    # Get the accuracy metric for this dataset
    dataset_metrics = {
        "cuad": "avg_f1",
        "blackvault": "avg_distinct_locations", 
        "game_reviews": "weighted_score",
        "medec": "combined_score",
        "sustainability": "combined_score",
        "biodex": "avg_rp_at_5"
    }
    
    accuracy_metric = dataset_metrics.get(dataset.lower())
    if not accuracy_metric:
        print(f"Unknown dataset '{dataset}' - no accuracy metric defined")
        return None
    
    # Perform bootstrap sampling
    bootstrap_results = []
    temp_file_path = f"{json_file_path}.bootstrap_temp.json"
    
    print(f"Performing {n_bootstrap} bootstrap iterations for {entry.get('file')}...")
    
    for i in range(n_bootstrap):
        # Create bootstrap sample
        bootstrap_data = bootstrap_sample(original_data)
        
        # Save to temporary file
        with open(temp_file_path, 'w') as f:
            json.dump(bootstrap_data, f, indent=2, cls=NumpyEncoder)
        
        # Evaluate bootstrap sample
        try:
            bootstrap_result = evaluate_func("bootstrap", temp_file_path)
            
            # Extract accuracy metric
            accuracy_value = bootstrap_result.get(accuracy_metric, 0.0)
            if isinstance(accuracy_value, dict) and 'value' in accuracy_value:
                accuracy_value = accuracy_value['value']
            
            # Estimate cost based on word count using original cost as reference
            original_cost = entry.get('cost', 0.0)
            estimated_cost = estimate_per_doc_cost(bootstrap_data, original_cost, original_data)
            
            bootstrap_results.append({
                'accuracy': float(accuracy_value),
                'cost': float(estimated_cost),
                'iteration': i
            })
            
        except Exception as e:
            print(f"  Warning: Bootstrap iteration {i + 1} failed: {e}")
            continue
    
    # Clean up temporary file
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)
    
    if not bootstrap_results:
        print(f"No successful bootstrap iterations for {entry.get('file')}")
        return None
    
    return {
        'entry': entry,
        'original_file': json_file_path,
        'bootstrap_results': bootstrap_results,
        'accuracy_metric': accuracy_metric
    }


def save_bootstrap_data_to_json(
    bootstrap_data: List[Dict[str, Any]], 
    dataset: str,
    output_path: str = "bootstrap_data.json",
    confidence_level: float = 0.68
) -> None:
    """
    Save bootstrap results to a well-formatted JSON file.
    
    Args:
        bootstrap_data: List of bootstrap results for each entry
        dataset: Dataset name
        output_path: Path to save the JSON file
    """
    # Create a structured output format
    output_data = {
        "dataset": dataset,
        "timestamp": str(np.datetime64('now')),
        "summary": {
            "total_entries": len(bootstrap_data),
            "bootstrap_iterations_per_entry": len(bootstrap_data[0]['bootstrap_results']) if bootstrap_data else 0,
            "confidence_level": confidence_level
        },
        "entries": []
    }
    
    for i, entry_data in enumerate(bootstrap_data):
        entry = entry_data['entry']
        bootstrap_results = entry_data['bootstrap_results']
        
        # Calculate statistics for this entry
        accuracies = [result['accuracy'] for result in bootstrap_results]
        costs = [result['cost'] for result in bootstrap_results]
        
        # Calculate confidence intervals based on confidence_level
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        accuracy_ci = [float(np.percentile(accuracies, lower_percentile)), float(np.percentile(accuracies, upper_percentile))]
        cost_ci = [float(np.percentile(costs, lower_percentile)), float(np.percentile(costs, upper_percentile))]
        
        entry_summary = {
            "entry_id": i + 1,
            "file_name": entry.get('file', 'Unknown'),
            "node_id": entry.get('node_id', 'N/A'),
            "original_metrics": {
                "accuracy": entry.get(entry_data['accuracy_metric'], 0.0),
                "cost": entry.get('cost', 0.0),
                "accuracy_metric_name": entry_data['accuracy_metric']
            },
            "bootstrap_statistics": {
                "accuracy": {
                    "mean": float(np.mean(accuracies)),
                    "std": float(np.std(accuracies)),
                    "min": float(np.min(accuracies)),
                    "max": float(np.max(accuracies)),
                    "median": float(np.median(accuracies)),
                    "ci": accuracy_ci
                },
                "cost": {
                    "mean": float(np.mean(costs)),
                    "std": float(np.std(costs)),
                    "min": float(np.min(costs)),
                    "max": float(np.max(costs)),
                    "median": float(np.median(costs)),
                    "ci": cost_ci
                }
            },
            "bootstrap_samples": bootstrap_results
        }
        
        output_data["entries"].append(entry_summary)
    
    # Save to JSON file with pretty formatting
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, cls=NumpyEncoder)
    
    print(f"Bootstrap data saved to: {output_path}")


def create_bootstrap_visualization(
    bootstrap_data: List[Dict[str, Any]], 
    dataset: str,
    output_path: str = "bootstrap_visualization.png",
    confidence_level: float = 0.68
) -> None:
    """
    Create visualization of bootstrap results with confidence intervals.
    
    Args:
        bootstrap_data: List of bootstrap results for each entry
        dataset: Dataset name
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Define colors for different entries
    colors = plt.cm.tab10(np.linspace(0, 1, len(bootstrap_data)))
    
    for i, entry_data in enumerate(bootstrap_data):
        entry = entry_data['entry']
        bootstrap_results = entry_data['bootstrap_results']
        
        # Extract accuracy and cost values
        accuracies = [result['accuracy'] for result in bootstrap_results]
        costs = [result['cost'] for result in bootstrap_results]
        
        # Calculate confidence intervals based on confidence_level
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        accuracy_lower = np.percentile(accuracies, lower_percentile)
        accuracy_upper = np.percentile(accuracies, upper_percentile)
        cost_lower = np.percentile(costs, lower_percentile)
        cost_upper = np.percentile(costs, upper_percentile)
        
        # Calculate rectangle dimensions
        width = cost_upper - cost_lower
        height = accuracy_upper - accuracy_lower
        
        # Plot confidence interval rectangle
        color = colors[i % len(colors)]
        rectangle = plt.Rectangle(
            (cost_lower, accuracy_lower), 
            width, 
            height,
            facecolor=color,
            alpha=0.3,
            edgecolor=color,
            linewidth=2,
            label=f"{entry.get('file', f'Entry {i+1}')} (node_id: {entry.get('node_id', 'N/A')})"
        )
        plt.gca().add_patch(rectangle)
        
        # Plot original point
        original_cost = entry.get('cost', 0.0)
        original_accuracy = entry.get(entry_data['accuracy_metric'], 0.0)
        
        plt.scatter(
            original_cost, 
            original_accuracy, 
            c=[color], 
            marker='o', 
            s=100, 
            edgecolors='black',
            linewidth=2,
            alpha=1.0,
            zorder=5  # Ensure points are on top of rectangles
        )
        
        # Add text annotation for confidence intervals
        ci_percentage = int(confidence_level * 100)
        plt.annotate(
            f'{ci_percentage}% CI: [{accuracy_lower:.3f}, {accuracy_upper:.3f}]',
            xy=(original_cost, original_accuracy),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=8,
            alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )
    
    plt.xlabel('Cost (estimated per-document)')
    plt.ylabel(f'Accuracy ({entry_data["accuracy_metric"]})')
    ci_percentage = int(confidence_level * 100)
    plt.title(f'Bootstrap {ci_percentage}% Confidence Intervals for {dataset.upper()} Dataset')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Show the plot
    plt.show()


def process_evaluation_metrics(
    metrics_file_path: str, 
    dataset: str, 
    n_bootstrap: int = 1000
) -> List[Dict[str, Any]]:
    """
    Process evaluation metrics file and perform bootstrap sampling.
    
    Args:
        metrics_file_path: Path to evaluation_metrics.json file
        dataset: Dataset name
        n_bootstrap: Number of bootstrap iterations
    
    Returns:
        List of bootstrap results for each entry
    """
    # Load evaluation metrics
    metrics_data = load_json_file(metrics_file_path)
    if not metrics_data:
        print(f"Could not load evaluation metrics from {metrics_file_path}")
        return []
    
    print(f"Loaded {len(metrics_data)} evaluation entries from {metrics_file_path}")
    
    bootstrap_results = []
    
    for entry in tqdm(metrics_data):
        
        result = bootstrap_evaluation_entry(
            entry, 
            dataset, 
            n_bootstrap
        )
        
        if result:
            bootstrap_results.append(result)
            print(f"  Successfully completed bootstrap sampling")
        else:
            print(f"  Failed to process entry")
    
    return bootstrap_results


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap sampling for evaluation metrics visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bootstrapping.py medec --modal_path outputs/medec_mcts
  python bootstrapping.py cuad --bootstrap_iterations 2000
  python bootstrapping.py game_reviews --json_output my_bootstrap_data.json
  python bootstrapping.py medec --confidence_level 0.95
        """
    )
    
    parser.add_argument(
        "dataset",
        help="Dataset name (cuad, blackvault, game_reviews, medec, sustainability, biodex)"
    )
    
    parser.add_argument(
        "--modal_path",
        help="Specific modal path to search for evaluation_metrics.json (optional)"
    )
    
    parser.add_argument(
        "--bootstrap_iterations",
        type=int,
        default=1000,
        help="Number of bootstrap iterations (default: 1000)"
    )
    
    parser.add_argument(
        "--output",
        default="bootstrap_visualization.png",
        help="Output file path for visualization (default: bootstrap_visualization.png)"
    )
    
    parser.add_argument(
        "--json_output",
        help="Output file path for bootstrap data JSON (default: bootstrap_data.json)"
    )
    
    parser.add_argument(
        "--confidence_level",
        type=float,
        default=0.68,
        help="Confidence level for intervals (default: 0.68 for 68%% CI)"
    )
    
    args = parser.parse_args()
    
    # Find evaluation_metrics.json file
    metrics_file_path = None
    
    if args.modal_path:
        # Use specific path if provided
        potential_path = Path(VOLUME_MOUNT_PATH) / args.modal_path / "evaluation_metrics.json"
        if potential_path.exists():
            metrics_file_path = str(potential_path)
        else:
            print(f"Evaluation metrics file not found at {potential_path}")
            sys.exit(1)
    else:
        # Search for evaluation_metrics.json in common locations
        search_paths = [
            Path(VOLUME_MOUNT_PATH) / f"outputs/{args.dataset}_mcts/evaluation_metrics.json",
        ]
        
        for path in search_paths:
            if path.exists():
                metrics_file_path = str(path)
                break
        
        if not metrics_file_path:
            print(f"Could not find evaluation_metrics.json for dataset '{args.dataset}'")
            print("Searched paths:")
            for path in search_paths:
                print(f"  - {path}")
            sys.exit(1)
    print(f"Using evaluation metrics file: {metrics_file_path}")
    
    try:
        # Process evaluation metrics and perform bootstrap sampling
        bootstrap_results = process_evaluation_metrics(
            metrics_file_path,
            args.dataset,
            args.bootstrap_iterations
        )
        
        if not bootstrap_results:
            print("No bootstrap results generated.")
            sys.exit(1)
        
        # Save bootstrap data to JSON in dataset directory
        dataset_output_dir = Path(f"outputs/{args.dataset}_mcts")
        json_output_path = dataset_output_dir / (args.json_output if args.json_output else "bootstrap_data.json")
        save_bootstrap_data_to_json(
            bootstrap_results,
            args.dataset,
            str(json_output_path),
            args.confidence_level
        )
        
        # Create visualization in dataset directory
        viz_output_path = dataset_output_dir / args.output
        create_bootstrap_visualization(
            bootstrap_results,
            args.dataset,
            str(viz_output_path),
            args.confidence_level
        )
        
        # Print summary
        print(f"\n{'='*60}")
        print("BOOTSTRAP SAMPLING SUMMARY")
        print(f"{'='*60}")
        print(f"Dataset: {args.dataset}")
        print(f"Bootstrap iterations: {args.bootstrap_iterations}")
        print(f"Processed entries: {len(bootstrap_results)}")
        
        for i, result in enumerate(bootstrap_results):
            entry = result['entry']
            bootstrap_data = result['bootstrap_results']
            
            print(f"\nEntry {i+1}: {entry.get('file', 'Unknown')}")
            print(f"  Node ID: {entry.get('node_id', 'N/A')}")
            print(f"  Original accuracy: {entry.get(result['accuracy_metric'], 'N/A')}")
            print(f"  Original cost: {entry.get('cost', 'N/A')}")
            print(f"  Bootstrap samples: {len(bootstrap_data)}")
            
            if bootstrap_data:
                accuracies = [r['accuracy'] for r in bootstrap_data]
                costs = [r['cost'] for r in bootstrap_data]
                print(f"  Bootstrap accuracy range: [{min(accuracies):.4f}, {max(accuracies):.4f}]")
                print(f"  Bootstrap cost range: [{min(costs):.4f}, {max(costs):.4f}]")
        
        print(f"\nVisualization saved to: {args.output}")
        print(f"Bootstrap data saved to: {json_output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@app.function(
    image=image,
    volumes={VOLUME_MOUNT_PATH: volume},
    timeout=3600
)
def run_bootstrap_analysis(dataset: str, modal_path: str = None, bootstrap_iterations: int = 10, 
                          output: str = "bootstrap_visualization.png", 
                          json_output: str = "bootstrap_data.json",
                          confidence_level: float = 0.68):
    """
    Modal function to run bootstrap analysis.
    
    Args:
        dataset: Dataset name
        modal_path: Specific modal path to search for evaluation_metrics.json
        bootstrap_iterations: Number of bootstrap iterations
        output: Output file path for visualization
        json_output: Output file path for bootstrap data JSON
        confidence_level: Confidence level for intervals (e.g., 0.68 for 68% CI)
    """
    # Find evaluation_metrics.json file
    metrics_file_path = None
    
    if modal_path:
        # Use specific path if provided
        potential_path = Path(VOLUME_MOUNT_PATH) / modal_path / "evaluation_metrics.json"
        if potential_path.exists():
            metrics_file_path = str(potential_path)
        else:
            print(f"Evaluation metrics file not found at {potential_path}")
            return {"error": f"Evaluation metrics file not found at {potential_path}"}
    else:
        # Search for evaluation_metrics.json in common locations
        search_paths = [
            Path(VOLUME_MOUNT_PATH) / f"outputs/{dataset}_mcts/evaluation_metrics.json", 
        ]
        
        for path in search_paths:
            if path.exists():
                metrics_file_path = str(path)
                break
        
        if not metrics_file_path:
            print(f"Could not find evaluation_metrics.json for dataset '{dataset}'")
            print("Searched paths:")
            for path in search_paths:
                print(f"  - {path}")
            return {"error": f"Could not find evaluation_metrics.json for dataset '{dataset}'"}
    
    print(f"Using evaluation metrics file: {metrics_file_path}")
    
    try:
        # Process evaluation metrics and perform bootstrap sampling
        bootstrap_results = process_evaluation_metrics(
            metrics_file_path,
            dataset,
            bootstrap_iterations
        )
        
        if not bootstrap_results:
            print("No bootstrap results generated.")
            return {"error": "No bootstrap results generated"}
        
        # Save bootstrap data to JSON in dataset directory
        dataset_output_dir = Path(VOLUME_MOUNT_PATH) / f"outputs/{dataset}_mcts"
        json_output_path = dataset_output_dir / json_output
        save_bootstrap_data_to_json(
            bootstrap_results,
            dataset,
            str(json_output_path),
            confidence_level
        )
        
        # Create visualization in dataset directory
        viz_output_path = dataset_output_dir / output
        create_bootstrap_visualization(
            bootstrap_results,
            dataset,
            str(viz_output_path),
            confidence_level
        )
        
        # Print summary
        print(f"\n{'='*60}")
        print("BOOTSTRAP SAMPLING SUMMARY")
        print(f"{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"Bootstrap iterations: {bootstrap_iterations}")
        print(f"Processed entries: {len(bootstrap_results)}")
        
        for i, result in enumerate(bootstrap_results):
            entry = result['entry']
            bootstrap_data = result['bootstrap_results']
            
            print(f"\nEntry {i+1}: {entry.get('file', 'Unknown')}")
            print(f"  Node ID: {entry.get('node_id', 'N/A')}")
            print(f"  Original accuracy: {entry.get(result['accuracy_metric'], 'N/A')}")
            print(f"  Original cost: {entry.get('cost', 'N/A')}")
            print(f"  Bootstrap samples: {len(bootstrap_data)}")
            
            if bootstrap_data:
                accuracies = [r['accuracy'] for r in bootstrap_data]
                costs = [r['cost'] for r in bootstrap_data]
                print(f"  Bootstrap accuracy range: [{min(accuracies):.4f}, {max(accuracies):.4f}]")
                print(f"  Bootstrap cost range: [{min(costs):.4f}, {max(costs):.4f}]")
        
        print(f"\nVisualization saved to: {viz_output_path}")
        print(f"Bootstrap data saved to: {json_output_path}")
        
        return {
            "success": True,
            "dataset": dataset,
            "processed_entries": len(bootstrap_results),
            "visualization_path": str(viz_output_path),
            "json_data_path": str(json_output_path)
        }
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


@app.local_entrypoint()
def main_modal(dataset: str, modal_path: str = None, bootstrap_iterations: int = 10, 
                output: str = "bootstrap_visualization.png", 
                json_output: str = "bootstrap_data.json",
                confidence_level: float = 0.68):
    """
    Modal entrypoint for bootstrap analysis.
    """
    result = run_bootstrap_analysis.remote(
        dataset=dataset,
        modal_path=modal_path,
        bootstrap_iterations=bootstrap_iterations,
        output=output,
        json_output=json_output,
        confidence_level=confidence_level
    )
    print("Bootstrap analysis completed!")
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
