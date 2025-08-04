import json
import matplotlib.pyplot as plt
from pathlib import Path
from .cuad import evaluate_results as cuad_evaluate
from .blackvault import evaluate_results as blackvault_evaluate
from .game_reviews import evaluate_results as game_reviews_evaluate
from .medec import evaluate_results as medec_evaluate

def get_evaluate_func(dataset):
    """
    Get the appropriate evaluation function for a dataset.
    
    Args:
        dataset (str): Dataset name ('cuad' or 'blackvault')
        
    Returns:
        callable: Evaluation function that takes (method_name, results_file_path)
    """
    if dataset.lower() == "cuad":
        def cuad_eval_func(method_name, results_file_path):
            ground_truth_path = "experiments/reasoning/data/CUAD-master_clauses.csv"
            return cuad_evaluate(method_name, results_file_path, ground_truth_path)
        return cuad_eval_func
    
    elif dataset.lower() == "blackvault":
        def blackvault_eval_func(method_name, results_file_path):
            return blackvault_evaluate(method_name, results_file_path)
        return blackvault_eval_func
    
    elif dataset.lower() == "game_reviews":
        def game_reviews_eval_func(method_name, results_file_path):
            return game_reviews_evaluate(method_name, results_file_path)
        return game_reviews_eval_func
    
    elif dataset.lower() == "medec":
        def medec_eval_func(method_name, results_file_path):
            return medec_evaluate(method_name, results_file_path)
        return medec_eval_func
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def compute_dataset_stats(data, dataset_name="data"):
    """
    Compute statistics for a dataset by analyzing the actual data.
    
    Args:
        data (list): List of data records
        dataset_name (str): Name of the dataset
        
    Returns:
        str: Formatted dataset statistics
    """
    if not data:
        return f"Dataset: {dataset_name}\nType: file\nRecords loaded: 0\nNo data available"
    
    num_records = len(data)
    total_tokens = 0
    field_stats = {}
    
    # Analyze each record
    for record in data:
        if isinstance(record, dict):
            for key, value in record.items():
                if key not in field_stats:
                    field_stats[key] = {'total_chars': 0, 'count': 0, 'type': type(value).__name__}
                
                if isinstance(value, str):
                    char_count = len(value)
                    field_stats[key]['total_chars'] += char_count
                    field_stats[key]['count'] += 1
                    total_tokens += char_count / 4  # 4 characters per token approximation
                elif isinstance(value, (int, float)):
                    # Numbers are typically short, estimate as ~5 characters
                    field_stats[key]['total_chars'] += 5
                    field_stats[key]['count'] += 1
                    total_tokens += 1.25
                elif isinstance(value, list):
                    # For lists, estimate based on string representation
                    str_repr = str(value)
                    char_count = len(str_repr)
                    field_stats[key]['total_chars'] += char_count
                    field_stats[key]['count'] += 1
                    total_tokens += char_count / 4
    
    # Format the output
    stats_lines = [
        f"Dataset: {dataset_name}",
        f"Type: file", 
        f"Records loaded: {num_records}",
        f"Input schema:"
    ]
    
    for field, stats in field_stats.items():
        if stats['count'] > 0:
            avg_tokens = (stats['total_chars'] / stats['count']) / 4
            field_type = "string" if stats['type'] in ['str'] else stats['type']
            stats_lines.append(f"    {field}: {field_type} (avg: {avg_tokens:.1f} tokens)")
    
    stats_lines.append(f"Total tokens: {int(total_tokens):,}")
    
    return "\n        ".join(stats_lines)

def get_dataset_stats(dataset, yaml_path):
    """
    Get dataset statistics by loading and analyzing the actual data.
    
    Args:
        dataset (str): Dataset name ('cuad' or 'blackvault')
        yaml_path (str): Path to the YAML configuration file
        
    Returns:
        str: Formatted dataset statistics
    """
    import yaml
    
    # Load the YAML config to get the data path
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract dataset info from config
    datasets = config.get('datasets', {})
    if not datasets:
        return f"Dataset: {dataset}\nType: file\nRecords loaded: 0\nNo datasets found in config"
    
    # Get the first dataset (assuming single dataset per config)
    dataset_name, dataset_config = next(iter(datasets.items()))
    data_path = dataset_config.get('path')
    
    if not data_path:
        return f"Dataset: {dataset_name}\nType: file\nRecords loaded: 0\nNo data path found"
    
    # Load the data
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        return compute_dataset_stats(data, dataset_name)
        
    except Exception as e:
        return f"Dataset: {dataset_name}\nType: file\nRecords loaded: 0\nError loading data: {e}"

def run_dataset_evaluation(dataset, nodes_or_files, output_path, ground_truth_path=None, method_name="docetl"):
    """
    Run evaluation for a specific dataset on a set of nodes or files.
    
    Args:
        dataset (str): Dataset name ('cuad' or 'blackvault')
        nodes_or_files (list): List of nodes (with result_path) or file paths
        output_path (Path): Path to save evaluation results
        ground_truth_path (str, optional): Path to ground truth file
        method_name (str): Method name for evaluation
        
    Returns:
        tuple: (eval_results, pareto_auc) where eval_results is list of evaluation metrics
    """
    eval_results = []
    pareto_auc = None
    
    if dataset.lower() == "cuad":
        if ground_truth_path is None:
            default_gt = Path("experiments/reasoning/data/CUAD-master_clauses.csv")
            ground_truth_path = str(default_gt)

        print(f"\n🧪 Evaluating extraction JSONs against CUAD ground truth ...")

        for item in nodes_or_files:
            # Handle both node objects and file paths
            if hasattr(item, 'result_path'):
                # This is a node object
                jf = item.result_path
                node_data = {
                    "node_id": item.get_id(),
                    "cost": item.cost,
                    "visits": getattr(item, 'visits', 0),
                    "value": getattr(item, 'value', 0),
                }
            else:
                # This is a file path with associated data
                jf = item["file_path"]
                node_data = {
                    "node_id": item.get("node_id", "unknown"),
                    "cost": item.get("cost", 0.0),
                    "visits": item.get("visits", 0),
                    "value": item.get("value", 0),
                }
            
            if jf is None or not Path(jf).exists():
                continue
            
            try:
                metrics = cuad_evaluate(method_name, jf, ground_truth_path)
                jp = Path(jf).resolve()
                op_root = output_path.resolve()
                if hasattr(jp, "is_relative_to") and jp.is_relative_to(op_root):
                    display_path = str(jp.relative_to(op_root))
                else:
                    display_path = jp.name

                result = {
                    "file": display_path,
                    "precision": metrics["avg_precision"],
                    "recall": metrics["avg_recall"],
                    "f1": metrics["avg_f1"],
                    **node_data
                }
                
                # Add frontier information if available
                if hasattr(item, 'result_path'):
                    result.update({
                        "mcts_accuracy": getattr(item, 'mcts_accuracy', None),
                        "on_frontier": getattr(item, 'on_frontier', False),
                    })
                
                eval_results.append(result)
            except Exception as e:
                print(f"   ⚠️  Evaluation failed for {jf}: {e}")

        if eval_results:
            # Create plots and compute AUC
            pareto_auc = _create_cuad_plots_and_auc(eval_results, output_path)
            
    elif dataset.lower() == "blackvault":
        print(f"\n🧪 Evaluating extraction JSONs for BlackVault dataset ...")

        for item in nodes_or_files:
            # Handle both node objects and file paths
            if hasattr(item, 'result_path'):
                # This is a node object
                jf = item.result_path
                node_data = {
                    "node_id": item.get_id(),
                    "cost": item.cost,
                    "visits": getattr(item, 'visits', 0),
                    "value": getattr(item, 'value', 0),
                }
            else:
                # This is a file path with associated data
                jf = item["file_path"]
                node_data = {
                    "node_id": item.get("node_id", "unknown"),
                    "cost": item.get("cost", 0.0),
                    "visits": item.get("visits", 0),
                    "value": item.get("value", 0),
                }
            
            if jf is None or not Path(jf).exists():
                continue
            
            try:
                metrics = blackvault_evaluate(method_name, jf)
                jp = Path(jf).resolve()
                op_root = output_path.resolve()
                if hasattr(jp, "is_relative_to") and jp.is_relative_to(op_root):
                    display_path = str(jp.relative_to(op_root))
                else:
                    display_path = jp.name

                result = {
                    "file": display_path,
                    "avg_distinct_locations": metrics["avg_distinct_locations"],
                    "total_documents": metrics["total_documents"],
                    "total_distinct_locations": metrics["total_distinct_locations"],
                    **node_data
                }
                
                # Add frontier information if available
                if hasattr(item, 'result_path'):
                    result.update({
                        "mcts_accuracy": getattr(item, 'mcts_accuracy', None),
                        "on_frontier": getattr(item, 'on_frontier', False),
                    })
                
                eval_results.append(result)
            except Exception as e:
                print(f"   ⚠️  Evaluation failed for {jf}: {e}")

        if eval_results:
            # Create plots and compute AUC
            pareto_auc = _create_blackvault_plots_and_auc(eval_results, output_path)
    
    elif dataset.lower() == "game_reviews":
        print(f"\n🧪 Evaluating game reviews analysis results ...")

        for item in nodes_or_files:
            # Handle both node objects and file paths
            if hasattr(item, 'result_path'):
                # This is a node object
                jf = item.result_path
                node_data = {
                    "node_id": item.get_id(),
                    "cost": item.cost,
                    "visits": getattr(item, 'visits', 0),
                    "value": getattr(item, 'value', 0),
                }
            else:
                # This is a file path with associated data
                jf = item["file_path"]
                node_data = {
                    "node_id": item.get("node_id", "unknown"),
                    "cost": item.get("cost", 0.0),
                    "visits": item.get("visits", 0),
                    "value": item.get("value", 0),
                }
            
            if jf is None or not Path(jf).exists():
                continue
            
            try:
                metrics = game_reviews_evaluate(method_name, jf)
                jp = Path(jf).resolve()
                op_root = output_path.resolve()
                if hasattr(jp, "is_relative_to") and jp.is_relative_to(op_root):
                    display_path = str(jp.relative_to(op_root))
                else:
                    display_path = jp.name

                result = {
                    "file": display_path,
                    "combined_accuracy_score": metrics["weighted_score"],  # Use weighted score (50-50 Kendall's tau + sentiment)
                    "sentiment_accuracy": metrics["sentiment_accuracy"],
                    "kendall_tau_score": metrics["kendall_tau_score"],
                    "weighted_score": metrics["weighted_score"],
                    **node_data
                }
                
                # Add frontier information if available
                if hasattr(item, 'result_path'):
                    result.update({
                        "mcts_accuracy": getattr(item, 'mcts_accuracy', None),
                        "on_frontier": getattr(item, 'on_frontier', False),
                    })
                
                eval_results.append(result)
            except Exception as e:
                print(f"   ⚠️  Evaluation failed for {jf}: {e}")

        if eval_results:
            # Create plots and compute AUC based on weighted score (50-50 Kendall's tau + sentiment)
            pareto_auc = _create_game_reviews_plots_and_auc(eval_results, output_path)
    
    elif dataset.lower() == "medec":
        print(f"\n🧪 Evaluating medical error detection results ...")

        for item in nodes_or_files:
            # Handle both node objects and file paths
            if hasattr(item, 'result_path'):
                # This is a node object
                jf = item.result_path
                node_data = {
                    "node_id": item.get_id(),
                    "cost": item.cost,
                    "visits": getattr(item, 'visits', 0),
                    "value": getattr(item, 'value', 0),
                }
            else:
                # This is a file path with associated data
                jf = item["file_path"]
                node_data = {
                    "node_id": item.get("node_id", "unknown"),
                    "cost": item.get("cost", 0.0),
                    "visits": item.get("visits", 0),
                    "value": item.get("value", 0),
                }
            
            if jf is None or not Path(jf).exists():
                continue
            
            try:
                metrics = medec_evaluate(method_name, jf)
                jp = Path(jf).resolve()
                op_root = output_path.resolve()
                if hasattr(jp, "is_relative_to") and jp.is_relative_to(op_root):
                    display_path = str(jp.relative_to(op_root))
                else:
                    display_path = jp.name

                result = {
                    "file": display_path,
                    "combined_score": metrics["combined_score"],
                    "error_flag_accuracy": metrics["error_flag_accuracy"],
                    "avg_error_sentence_jaccard": metrics["avg_error_sentence_jaccard"],
                    "avg_corrected_sentence_jaccard": metrics["avg_corrected_sentence_jaccard"],
                    "total_cases": metrics["total_cases"],
                    "num_error_cases": metrics["num_error_cases"],
                    "num_corrected_cases": metrics["num_corrected_cases"],
                    **node_data
                }
                
                # Add frontier information if available
                if hasattr(item, 'result_path'):
                    result.update({
                        "mcts_accuracy": getattr(item, 'mcts_accuracy', None),
                        "on_frontier": getattr(item, 'on_frontier', False),
                    })
                
                eval_results.append(result)
            except Exception as e:
                print(f"   ⚠️  Evaluation failed for {jf}: {e}")

        if eval_results:
            # Create plots and compute AUC
            pareto_auc = _create_medec_plots_and_auc(eval_results, output_path)
    
    # Save evaluation results
    if eval_results:
        eval_out_file = output_path / "evaluation_metrics.json"
        with open(eval_out_file, "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"📊 Evaluation results written to {eval_out_file}")
    
    return eval_results, pareto_auc

def _create_cuad_plots_and_auc(eval_results, output_path):
    """Create plots and compute AUC for CUAD dataset"""
    pareto_auc = None
    
    # Plot F1 vs Cost scatter
    try:
        costs = [row["cost"] for row in eval_results]
        f1s = [row["f1"] for row in eval_results]
        colors = ["blue" if row.get("on_frontier", False) else "grey" for row in eval_results]

        plt.figure(figsize=(8,6))
        plt.scatter(costs, f1s, c=colors)
        for row in eval_results:
            mcts_accuracy = row.get("mcts_accuracy", 0)
            if mcts_accuracy is not None:
                label = f"{row['node_id']} ({mcts_accuracy:.2f})"
            else:
                label = row["file"]
            plt.annotate(label, (row["cost"], row["f1"]), textcoords="offset points", xytext=(4,4), fontsize=8)

        plt.xlabel("Cost ($)")
        plt.ylabel("F1 Score")
        plt.title("Cost vs F1 for all plans")
        plt.grid(True, linestyle="--", alpha=0.5)
        plot_path = output_path / "cost_vs_f1.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📈 Scatter plot saved to: {plot_path}")
    except Exception as e:
        print(f"⚠️  Failed to create scatter plot: {e}")
    
    # Compute Area Under the Pareto Frontier (Cost vs F1)
    try:
        frontier_points = [row for row in eval_results if row.get("on_frontier", False)]
        if len(frontier_points) >= 2:
            # Sort frontier points by cost (x-axis)
            frontier_points.sort(key=lambda r: r["cost"])

            pareto_auc = 0.0
            prev_point = frontier_points[0]
            for curr_point in frontier_points[1:]:
                width = curr_point["cost"] - prev_point["cost"]
                if width > 0:  # Ignore duplicate cost values
                    pareto_auc += 0.5 * width * (prev_point["f1"] + curr_point["f1"])
                prev_point = curr_point
        elif frontier_points:
            pareto_auc = 0.0  # Single point frontier → zero area

        if pareto_auc is not None:
            print(f"📐 Area under Pareto frontier (Cost vs F1): {pareto_auc:.4f}")
    except Exception as e:
        print(f"⚠️  Failed to compute Pareto AUC: {e}")
    
    return pareto_auc

def _create_game_reviews_plots_and_auc(eval_results, output_path):
    """Create plots and compute AUC for game reviews dataset"""
    pareto_auc = None
    
    # Plot Combined Accuracy Score vs Cost scatter
    try:
        costs = [row["cost"] for row in eval_results]
        combined_scores = [row["combined_accuracy_score"] for row in eval_results]
        colors = ["blue" if row.get("on_frontier", False) else "grey" for row in eval_results]

        plt.figure(figsize=(8,6))
        plt.scatter(costs, combined_scores, c=colors)
        for row in eval_results:
            mcts_accuracy = row.get("mcts_accuracy", 0)
            if mcts_accuracy is not None:
                label = f"{row['node_id']} ({mcts_accuracy:.2f})"
            else:
                label = row["file"]
            plt.annotate(label, (row["cost"], row["combined_accuracy_score"]), textcoords="offset points", xytext=(4,4), fontsize=8)

        plt.xlabel("Cost ($)")
        plt.ylabel("Weighted Score (Kendall's τ + Sentiment)")
        plt.title("Cost vs Weighted Score (50-50 Kendall's τ + Sentiment) for all plans")
        plt.grid(True, linestyle="--", alpha=0.5)
        plot_path = output_path / "cost_vs_weighted_score.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📈 Scatter plot saved to: {plot_path}")
    except Exception as e:
        print(f"⚠️  Failed to create scatter plot: {e}")
    
    # Compute Area Under the Pareto Frontier (Cost vs Weighted Score)
    try:
        frontier_points = [row for row in eval_results if row.get("on_frontier", False)]
        if len(frontier_points) >= 2:
            # Sort frontier points by cost (x-axis)
            frontier_points.sort(key=lambda r: r["cost"])

            pareto_auc = 0.0
            prev_point = frontier_points[0]
            for curr_point in frontier_points[1:]:
                width = curr_point["cost"] - prev_point["cost"]
                if width > 0:  # Ignore duplicate cost values
                    pareto_auc += 0.5 * width * (prev_point["combined_accuracy_score"] + curr_point["combined_accuracy_score"])
                prev_point = curr_point
        elif frontier_points:
            pareto_auc = 0.0  # Single point frontier → zero area

        if pareto_auc is not None:
            print(f"📐 Area under Pareto frontier (Cost vs Weighted Score): {pareto_auc:.4f}")
    except Exception as e:
        print(f"⚠️  Failed to compute Pareto AUC: {e}")
    
    return pareto_auc

def _create_blackvault_plots_and_auc(eval_results, output_path):
    """Create plots and compute AUC for BlackVault dataset"""
    pareto_auc = None
    
    # Plot Avg Distinct Locations vs Cost scatter
    try:
        costs = [row["cost"] for row in eval_results]
        avg_locations = [row["avg_distinct_locations"] for row in eval_results]
        colors = ["blue" if row.get("on_frontier", False) else "grey" for row in eval_results]

        plt.figure(figsize=(8,6))
        plt.scatter(costs, avg_locations, c=colors)
        for row in eval_results:
            mcts_accuracy = row.get("mcts_accuracy", 0)
            if mcts_accuracy is not None:
                label = f"{row['node_id']} ({mcts_accuracy:.2f})"
            else:
                label = row["file"]
            plt.annotate(label, (row["cost"], row["avg_distinct_locations"]), textcoords="offset points", xytext=(4,4), fontsize=8)

        plt.xlabel("Cost ($)")
        plt.ylabel("Avg Distinct Locations")
        plt.title("Cost vs Avg Distinct Locations for all plans")
        plt.grid(True, linestyle="--", alpha=0.5)
        plot_path = output_path / "cost_vs_avg_locations.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📈 Scatter plot saved to: {plot_path}")
    except Exception as e:
        print(f"⚠️  Failed to create scatter plot: {e}")
    
    # Compute Area Under the Pareto Frontier (Cost vs Avg Distinct Locations)
    try:
        frontier_points = [row for row in eval_results if row.get("on_frontier", False)]
        if len(frontier_points) >= 2:
            # Sort frontier points by cost (x-axis)
            frontier_points.sort(key=lambda r: r["cost"])

            pareto_auc = 0.0
            prev_point = frontier_points[0]
            for curr_point in frontier_points[1:]:
                width = curr_point["cost"] - prev_point["cost"]
                if width > 0:  # Ignore duplicate cost values
                    pareto_auc += 0.5 * width * (prev_point["avg_distinct_locations"] + curr_point["avg_distinct_locations"])
                prev_point = curr_point
        elif frontier_points:
            pareto_auc = 0.0  # Single point frontier → zero area

        if pareto_auc is not None:
            print(f"📐 Area under Pareto frontier (Cost vs Avg Distinct Locations): {pareto_auc:.4f}")
    except Exception as e:
        print(f"⚠️  Failed to compute Pareto AUC: {e}")
    
    return pareto_auc

def _create_medec_plots_and_auc(eval_results, output_path):
    """Create plots and compute AUC for MEDEC dataset"""
    pareto_auc = None
    
    # Plot Combined Score vs Cost scatter
    try:
        costs = [row["cost"] for row in eval_results]
        scores = [row["combined_score"] for row in eval_results]
        colors = ["blue" if row.get("on_frontier", False) else "grey" for row in eval_results]

        plt.figure(figsize=(8,6))
        plt.scatter(costs, scores, c=colors)
        for row in eval_results:
            mcts_accuracy = row.get("mcts_accuracy", 0)
            if mcts_accuracy is not None:
                label = f"{row['node_id']} ({mcts_accuracy:.2f})"
            else:
                label = row["file"]
            plt.annotate(label, (row["cost"], row["combined_score"]), textcoords="offset points", xytext=(4,4), fontsize=8)

        plt.xlabel("Cost ($)")
        plt.ylabel("Combined Score")
        plt.title("Cost vs Combined Score (50% Error Flag + 25% Error Sim + 25% Corrected Sim)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plot_path = output_path / "cost_vs_combined_score.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📈 Scatter plot saved to: {plot_path}")
    except Exception as e:
        print(f"⚠️  Failed to create scatter plot: {e}")
    
    # Compute Area Under the Pareto Frontier (Cost vs Combined Score)
    try:
        frontier_points = [row for row in eval_results if row.get("on_frontier", False)]
        if len(frontier_points) >= 2:
            # Sort frontier points by cost (x-axis)
            frontier_points.sort(key=lambda r: r["cost"])

            pareto_auc = 0.0
            prev_point = frontier_points[0]
            for curr_point in frontier_points[1:]:
                width = curr_point["cost"] - prev_point["cost"]
                if width > 0:  # Ignore duplicate cost values
                    pareto_auc += 0.5 * width * (prev_point["combined_score"] + curr_point["combined_score"])
                prev_point = curr_point
        elif frontier_points:
            pareto_auc = 0.0  # Single point frontier → zero area

        if pareto_auc is not None:
            print(f"📐 Area under Pareto frontier (Cost vs Combined Score): {pareto_auc:.4f}")
    except Exception as e:
        print(f"⚠️  Failed to compute Pareto AUC: {e}")
    
    return pareto_auc

