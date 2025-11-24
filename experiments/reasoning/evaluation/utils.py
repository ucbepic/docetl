import json
import matplotlib.pyplot as plt
from pathlib import Path
from .cuad import evaluate_results as cuad_evaluate
from .blackvault import evaluate_results as blackvault_evaluate
from .game_reviews import evaluate_results as game_reviews_evaluate
from .medec import evaluate_results as medec_evaluate
from .sustainability import evaluate_results as sustainability_evaluate
from .biodex import evaluate_results as biodex_evaluate
from .facility import evaluate_results as facility_evaluate
import importlib.util
from pathlib import Path

dataset_accuracy_metrics = {
    "cuad": "avg_f1",
    "blackvault": "avg_distinct_locations", 
    "game_reviews": "weighted_score",
    "medec": "combined_score",
    "sustainability": "economic_activity_accuracy",
    "biodex": "avg_rp_at_5"
}

def _extract_node_data(item):
    """Extract node data from either a node object or a dict/file path."""
    if hasattr(item, 'result_path'):
        jf = item.result_path
        node_data = {
            "node_id": item.get_id(),
            "cost": item.cost,
            "visits": getattr(item, 'visits', 0),
            "value": getattr(item, 'value', 0),
        }
    else:
        jf = item.get("file_path") if isinstance(item, dict) else item
        node_data = {
            "node_id": item.get("node_id", "unknown") if isinstance(item, dict) else "unknown",
            "cost": item.get("cost", 0.0) if isinstance(item, dict) else 0.0,
            "visits": item.get("visits", 0) if isinstance(item, dict) else 0,
            "value": item.get("value", 0) if isinstance(item, dict) else 0,
        }
    return jf, node_data

def _get_display_path(jf, output_path):
    """Get display path for a result file."""
    jp = Path(jf).resolve()
    op_root = output_path.resolve()
    if hasattr(jp, "is_relative_to") and jp.is_relative_to(op_root):
        return str(jp.relative_to(op_root))
    else:
        return jp.name

def _add_frontier_info(result, item):
    """Add frontier information if available."""
    if hasattr(item, 'result_path'):
        result.update({
            "moar_accuracy": getattr(item, 'moar_accuracy', None),
            "on_frontier": getattr(item, 'on_frontier', False),
        })
    return result

def _process_evaluation_items(nodes_or_files, evaluate_func, output_path, method_name, result_fields, field_mapping):
    """Process evaluation items and return list of results."""
    eval_results = []
    for item in nodes_or_files:
        jf, node_data = _extract_node_data(item)
        
        if jf is None or not Path(jf).exists():
            continue
        
        try:
            metrics = evaluate_func(jf)
            display_path = _get_display_path(jf, output_path)
            
            # Build result dict
            result = {
                "file": display_path,
                **node_data
            }
            
            # Add result fields from metrics
            for field in result_fields:
                if field in field_mapping:
                    # Map field name
                    metric_key = field_mapping[field]
                    result[field] = metrics.get(metric_key)
                else:
                    # Use field name directly
                    result[field] = metrics.get(field)
            
            # Add frontier information
            result = _add_frontier_info(result, item)
            eval_results.append(result)
        except Exception as e:
            print(f"   ⚠️  Evaluation failed for {jf}: {e}")
    
    return eval_results

def identify_pareto_frontier(eval_results, dataset):
    """
    Identify the Pareto frontier for a given dataset based on accuracy vs cost.
    
    Args:
        eval_results (list): List of evaluation results with cost and accuracy metrics
        dataset (str): Dataset name to determine which accuracy metric to use
        
    Returns:
        list: Updated eval_results with 'on_frontier' field set to True/False
    """
    if not eval_results:
        return eval_results
    
    # Define the accuracy metric for each dataset
    dataset_metrics = {
        "cuad": "f1",
        "blackvault": "avg_distinct_locations", 
        "game_reviews": "combined_accuracy_score",
        "medec": "combined_score",
        "sustainability": "combined_score",
        "biodex": "avg_rp_at_5",
    }
    
    accuracy_metric = dataset_metrics.get(dataset.lower())
    if not accuracy_metric:
        print(f"⚠️  Unknown dataset '{dataset}', cannot identify Pareto frontier")
        return eval_results
    
    # Filter out results that don't have the required metrics
    valid_results = [r for r in eval_results if accuracy_metric in r and "cost" in r]
    if not valid_results:
        print(f"⚠️  No valid results with {accuracy_metric} and cost metrics")
        return eval_results
    
    # Sort by cost (ascending) and accuracy (descending for maximization)
    valid_results.sort(key=lambda x: (x["cost"], -x[accuracy_metric]))
    
    # Initialize frontier status
    for result in eval_results:
        result["on_frontier"] = False
    
    # Identify Pareto frontier points
    frontier_points = []
    for result in valid_results:
        # Check if this point is dominated by any existing frontier point
        is_dominated = False
        for frontier_point in frontier_points:
            # A point is dominated if another point has lower or equal cost AND higher or equal accuracy
            if (frontier_point["cost"] <= result["cost"] and 
                frontier_point[accuracy_metric] >= result[accuracy_metric]):
                is_dominated = True
                break
        
        if not is_dominated:
            frontier_points.append(result)
            # Mark this result as on the frontier
            for eval_result in eval_results:
                if (eval_result.get("file") == result.get("file") and 
                    eval_result.get("cost") == result.get("cost")):
                    eval_result["on_frontier"] = True
                    break
    
    print(f"📊 Identified {len(frontier_points)} Pareto frontier points for {dataset} dataset")
    print(f"   Accuracy metric: {accuracy_metric}")
    
    return eval_results

def print_pareto_frontier_summary(eval_results, dataset):
    """
    Print a summary of the Pareto frontier for a dataset.
    
    Args:
        eval_results (list): List of evaluation results with 'on_frontier' field
        dataset (str): Dataset name for display purposes
    """
    frontier_points = [r for r in eval_results if r.get("on_frontier", False)]
    
    if not frontier_points:
        print(f"📊 No Pareto frontier points found for {dataset} dataset")
        return
    
    # Define the accuracy metric for each dataset
    dataset_metrics = {
        "cuad": "f1",
        "blackvault": "avg_distinct_locations", 
        "game_reviews": "combined_accuracy_score",
        "medec": "combined_score",
        "sustainability": "combined_score",
        "biodex": "avg_rp_at_5",
    }
    
    accuracy_metric = dataset_metrics.get(dataset.lower(), "accuracy")
    
    # Sort frontier points by cost for better display
    frontier_points.sort(key=lambda x: x["cost"])
    
    print(f"\n🏆 Pareto Frontier Summary for {dataset.upper()} Dataset:")
    print(f"{'='*60}")
    print(f"{'Rank':<4} {'Cost ($)':<10} {accuracy_metric.upper():<15} {'File':<30}")
    print(f"{'='*60}")
    
    for i, point in enumerate(frontier_points, 1):
        cost = point.get("cost", 0.0)
        accuracy = point.get(accuracy_metric, 0.0)
        file_name = point.get("file", "unknown")
        
        print(f"{i:<4} ${cost:<9.4f} {accuracy:<15.4f} {file_name:<30}")
    
    print(f"{'='*60}")
    print(f"Total frontier points: {len(frontier_points)}")
    

def save_pareto_frontier_results(eval_results, dataset, output_path):
    """
    Save Pareto frontier results to a separate JSON file for analysis.
    
    Args:
        eval_results (list): List of evaluation results with 'on_frontier' field
        dataset (str): Dataset name
        output_path (Path): Output directory path
    """
    frontier_points = [r for r in eval_results if r.get("on_frontier", False)]
    
    if not frontier_points:
        return
    
    # Define the accuracy metric for each dataset
    dataset_metrics = {
        "cuad": "f1",
        "blackvault": "avg_distinct_locations", 
        "game_reviews": "combined_accuracy_score",
        "medec": "combined_score",
        "sustainability": "combined_score",
        "biodex": "avg_rp_at_5",
    }
    
    accuracy_metric = dataset_metrics.get(dataset.lower(), "accuracy")
    
    # Sort frontier points by cost
    frontier_points.sort(key=lambda x: x["cost"])
    
    # Add rank and cost-effectiveness information
    for i, point in enumerate(frontier_points):
        point["rank"] = i + 1
        point["accuracy_metric"] = accuracy_metric
    
    # Calculate cost-effectiveness ratios between consecutive points
    cost_effectiveness_analysis = []
    for i in range(len(frontier_points) - 1):
        curr = frontier_points[i]
        next_point = frontier_points[i + 1]
        
        cost_diff = next_point["cost"] - curr["cost"]
        accuracy_diff = next_point[accuracy_metric] - curr[accuracy_metric]
        
        if cost_diff > 0 and accuracy_diff > 0:
            cost_effectiveness = cost_diff / accuracy_diff
            cost_effectiveness_analysis.append({
                "from_file": curr["file"],
                "to_file": next_point["file"],
                "cost_increase": cost_diff,
                "accuracy_increase": accuracy_diff,
                "cost_per_unit_improvement": cost_effectiveness
            })
    
    # Create frontier summary
    frontier_summary = {
        "dataset": dataset,
        "accuracy_metric": accuracy_metric,
        "total_frontier_points": len(frontier_points),
        "frontier_points": frontier_points,
        "cost_effectiveness_analysis": cost_effectiveness_analysis,
        "timestamp": str(Path().cwd())
    }
    
    # Save to file
    frontier_file = output_path / f"pareto_frontier_{dataset}.json"
    with open(frontier_file, "w") as f:
        json.dump(frontier_summary, f, indent=2)
    
    print(f"💾 Pareto frontier results saved to: {frontier_file}")

def load_custom_evaluate_func(accuracy_function_path: str):
    """
    Load a custom evaluation function from a Python file.
    
    Args:
        accuracy_function_path (str): Path to a Python file containing an `evaluate_results` function
        
    Returns:
        callable: Evaluation function that takes (method_name, results_file_path, ground_truth_path=None, original_json_file=None)
        
    Raises:
        ValueError: If the file doesn't exist or doesn't contain `evaluate_results` function
    """
    
    
    func_path = Path(accuracy_function_path)
    if not func_path.exists():
        raise ValueError(f"Accuracy function file not found: {accuracy_function_path}")
    
    spec = importlib.util.spec_from_file_location("custom_evaluate", func_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load module from: {accuracy_function_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if not hasattr(module, 'evaluate_results'):
        raise ValueError(f"Module {accuracy_function_path} does not contain 'evaluate_results' function")
    
    evaluate_func = getattr(module, 'evaluate_results')
    
    def wrapped_eval_func(method_name, results_file_path):
        """
        Wrapper that calls the custom evaluate_results function.
        The custom function should have signature:
        evaluate_results(method_name, results_file, ground_truth_file=None, original_json_file=None)
        """
        return evaluate_func(method_name, results_file_path)
    
    return wrapped_eval_func

def get_evaluate_func(dataset, mode="train", custom_evaluate_func=None):
    """
    Get the appropriate evaluation function for a dataset.
    
    Args:
        dataset (str): Dataset name ('cuad' or 'blackvault')
        mode (str): 'train' or 'test' mode
        custom_evaluate_func (callable, optional): Custom evaluation function to use instead of predefined ones
        
    Returns:
        callable: Evaluation function that takes (method_name, results_file_path)
    """
    # If custom function provided, use it
    if custom_evaluate_func is not None:
        return custom_evaluate_func
    if dataset.lower() == "cuad":
        def cuad_eval_func(method_name, results_file_path):
            ground_truth_path = "experiments/reasoning/data/CUAD-master_clauses.csv"
            if mode == "train":
                original_json_path = "experiments/reasoning/data/train/cuad.json"
            else:
                original_json_path = "experiments/reasoning/data/test/cuad.json"
            return cuad_evaluate(method_name, results_file_path, ground_truth_path, original_json_path)
        return cuad_eval_func
    
    elif dataset.lower() == "blackvault":
        def blackvault_eval_func(method_name, results_file_path):
            if mode == "train":
                original_json_path = "experiments/reasoning/data/train/blackvault.json"
            else:
                original_json_path = "experiments/reasoning/data/test/blackvault.json"
            return blackvault_evaluate(method_name, results_file_path, original_json_path)
        return blackvault_eval_func
    
    elif dataset.lower() == "game_reviews":
        def game_reviews_eval_func(method_name, results_file_path):
            if mode == "train":
                original_json_path = "experiments/reasoning/data/train/game_reviews.json"
            else:
                original_json_path = "experiments/reasoning/data/test/game_reviews.json"
            return game_reviews_evaluate(method_name, results_file_path, original_json_path)
        return game_reviews_eval_func
    
    elif dataset.lower() == "medec":
        def medec_eval_func(method_name, results_file_path):
            if mode == "train":
                original_json_path = "experiments/reasoning/data/train/medec.json"
            else:
                original_json_path = "experiments/reasoning/data/test/medec.json"
            return medec_evaluate(method_name, results_file_path, original_json_path)
        return medec_eval_func
    
    elif dataset.lower() == "sustainability":
        def sustainability_eval_func(method_name, results_file_path):
            ground_truth_path = "experiments/reasoning/data/company_reports_gt.json"
            if mode == "train":
                original_json_path = "experiments/reasoning/data/train/sustainability.json"
            else:
                original_json_path = "experiments/reasoning/data/test/sustainability.json"
            return sustainability_evaluate(method_name, results_file_path, ground_truth_path, original_json_path)
        return sustainability_eval_func
    
    elif dataset.lower() == "biodex":
        def biodex_eval_func(method_name, results_file_path):
            if mode == "train":
                original_json_path = "experiments/reasoning/data/train/biodex.json"
            else:
                original_json_path = "experiments/reasoning/data/test/biodex.json"
            return biodex_evaluate(method_name, results_file_path, original_json_path)
        return biodex_eval_func
    
    elif dataset.lower() == "facility":
        def facility_eval_func(method_name, results_file_path):
            return facility_evaluate(method_name, results_file_path)
        return facility_eval_func
    
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
                # Skip if key starts with "GT "
                if key.startswith("GT "):
                    continue
                
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

def _get_dataset_config(dataset, ground_truth_path, method_name):
    """Get dataset configuration for evaluation."""
    dataset_lower = dataset.lower()
    
    configs = {
        "cuad": {
            "evaluate_func": lambda jf: cuad_evaluate(method_name, jf, 
                ground_truth_path or "experiments/reasoning/data/CUAD-master_clauses.csv",
                "experiments/reasoning/data/train/cuad.json"),
            "result_fields": ["precision", "recall", "f1"],
            "field_mapping": {"precision": "avg_precision", "recall": "avg_recall", "f1": "avg_f1"},
            "plot_func_name": "cuad"
        },
        "blackvault": {
            "evaluate_func": lambda jf: blackvault_evaluate(method_name, jf, 
                "experiments/reasoning/data/train/blackvault.json"),
            "result_fields": ["avg_distinct_locations", "total_documents", "total_distinct_locations"],
            "field_mapping": {},
            "plot_func_name": "blackvault"
        },
        "game_reviews": {
            "evaluate_func": lambda jf: game_reviews_evaluate(method_name, jf,
                "experiments/reasoning/data/train/game_reviews.json"),
            "result_fields": ["combined_accuracy_score", "sentiment_accuracy", "kendall_tau_score", "weighted_score"],
            "field_mapping": {"combined_accuracy_score": "weighted_score"},
            "plot_func_name": "game_reviews"
        },
        "medec": {
            "evaluate_func": lambda jf: medec_evaluate(method_name, jf,
                "experiments/reasoning/data/train/medec.json"),
            "result_fields": ["combined_score", "error_flag_accuracy", "avg_error_sentence_jaccard",
                             "avg_corrected_sentence_jaccard", "total_cases", "num_error_cases", "num_corrected_cases"],
            "field_mapping": {},
            "plot_func_name": "medec"
        },
        "sustainability": {
            "evaluate_func": lambda jf: sustainability_evaluate(method_name, jf,
                ground_truth_path or "experiments/reasoning/data/company_reports_gt.json",
                "experiments/reasoning/data/train/sustainability.json"),
            "result_fields": ["combined_score", "company_name_accuracy", "total_companies_processed",
                             "avg_findings_length", "total_economic_activities"],
            "field_mapping": {},
            "plot_func_name": "sustainability"
        },
        "biodex": {
            "evaluate_func": lambda jf: biodex_evaluate(method_name, jf,
                "experiments/reasoning/data/train/biodex.json"),
            "result_fields": ["avg_rp_at_5", "avg_rp_at_10", "avg_term_recall", "total_documents"],
            "field_mapping": {},
            "plot_func_name": "biodex"
        },
        "facility": {
            "evaluate_func": lambda jf: facility_evaluate(method_name, jf),
            "result_fields": ["combined_score"],
            "field_mapping": {},
            "plot_func_name": "facility"
        }
    }
    
    return configs.get(dataset_lower)

def _get_plot_func(plot_func_name):
    """Get plot function by name."""
    plot_funcs = {
        "cuad": _create_cuad_plots,
        "blackvault": _create_blackvault_plots,
        "game_reviews": _create_game_reviews_plots,
        "medec": _create_medec_plots,
        "sustainability": _create_sustainability_plots,
        "biodex": _create_biodex_plots,
        "facility": _create_facility_plots
    }
    return plot_funcs.get(plot_func_name)

def run_dataset_evaluation(dataset, nodes_or_files, output_path, ground_truth_path=None, method_name="docetl", root_cost=None, custom_evaluate_func=None, custom_metric_key=None):
    """
    Run evaluation for a specific dataset on a set of nodes or files.
    
    Args:
        dataset (str): Dataset name ('cuad' or 'blackvault')
        nodes_or_files (list): List of nodes (with result_path) or file paths
        output_path (Path): Path to save evaluation results
        ground_truth_path (str, optional): Path to ground truth file
        method_name (str): Method name for evaluation
        root_cost (float, optional): Root cost (deprecated, kept for compatibility)
        custom_evaluate_func (callable, optional): Custom evaluation function (method_name, results_file_path) -> dict
        custom_metric_key (str, optional): Key to extract from evaluation results for accuracy metric
        
    Returns:
        tuple: (eval_results, None) where eval_results is list of evaluation metrics
    """
    eval_results = []
    
    if custom_evaluate_func is not None:
        if custom_metric_key is None:
            raise ValueError("custom_metric_key must be provided when using custom_evaluate_func")
        
        # Process custom evaluation
        for item in nodes_or_files:
            jf, node_data = _extract_node_data(item)
            if jf is None or not Path(jf).exists():
                continue
            
            try:
                metrics = custom_evaluate_func(method_name, jf)
                display_path = _get_display_path(jf, output_path)
                
                # Extract the custom metric
                accuracy_value = metrics.get(custom_metric_key)
                if accuracy_value is None:
                    print(f"⚠️  Warning: Custom metric key '{custom_metric_key}' not found in evaluation results for {jf}")
                    accuracy_value = next((v for v in metrics.values() if isinstance(v, (int, float))), 0.0)
                
                result = {
                    "file": display_path,
                    custom_metric_key: accuracy_value,
                    **metrics,  # Include all metrics from custom function
                    **node_data
                }
                result = _add_frontier_info(result, item)
                eval_results.append(result)
            except Exception as e:
                print(f"   ⚠️  Evaluation failed for {jf}: {e}")
        
        # Create plots if we have results
        if eval_results:
            _create_generic_plots(eval_results, output_path, custom_metric_key)
    else:
        # Use dataset configuration
        config = _get_dataset_config(dataset, ground_truth_path, method_name)
        if config is None:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        # Process evaluation items
        eval_results = _process_evaluation_items(
            nodes_or_files, 
            config["evaluate_func"], 
            output_path, 
            method_name,
            config["result_fields"],
            config["field_mapping"]
        )
        
        # Create plots if we have results
        if eval_results:
            plot_func = _get_plot_func(config["plot_func_name"])
            if plot_func:
                plot_func(eval_results, output_path)
    
    # Identify Pareto frontier for all datasets
    if eval_results:
        print(f"\n🔍 Identifying Pareto frontier for {dataset} dataset...")
        eval_results = identify_pareto_frontier(eval_results, dataset)
        
        # Print Pareto frontier summary
        print_pareto_frontier_summary(eval_results, dataset)
        
        # Save Pareto frontier results to separate file
        save_pareto_frontier_results(eval_results, dataset, output_path)
    
    # Save evaluation results
    if eval_results:
        eval_out_file = output_path / "evaluation_metrics.json"
        with open(eval_out_file, "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"📊 Evaluation results written to {eval_out_file}")
    
    return eval_results, None

def _create_cuad_plots(eval_results, output_path):
    """Create plots for CUAD dataset"""
    
    # Plot F1 vs Cost scatter
    try:
        costs = [row["cost"] for row in eval_results]
        f1s = [row["f1"] for row in eval_results]
        colors = ["blue" if row.get("on_frontier", False) else "grey" for row in eval_results]

        plt.figure(figsize=(8,6))
        plt.scatter(costs, f1s, c=colors)
        for row in eval_results:
            moar_accuracy = row.get("moar_accuracy")
            if moar_accuracy is not None:
                label = f"{row['node_id']} ({moar_accuracy:.2f})"
            else:
                label = row.get("node_id", row.get("file", ""))
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

def _create_game_reviews_plots(eval_results, output_path):
    """Create plots for game reviews dataset"""
    # Plot Combined Accuracy Score vs Cost scatter
    try:
        costs = [row["cost"] for row in eval_results]
        combined_scores = [row["combined_accuracy_score"] for row in eval_results]
        colors = ["blue" if row.get("on_frontier", False) else "grey" for row in eval_results]

        plt.figure(figsize=(8,6))
        plt.scatter(costs, combined_scores, c=colors)
        for row in eval_results:
            moar_accuracy = row.get("moar_accuracy")
            if moar_accuracy is not None:
                label = f"{row['node_id']} ({moar_accuracy:.2f})"
            else:
                label = row.get("node_id", row.get("file", ""))
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

def _create_blackvault_plots(eval_results, output_path):
    """Create plots for BlackVault dataset"""
    # Plot Avg Distinct Locations vs Cost scatter
    try:
        costs = [row["cost"] for row in eval_results]
        avg_locations = [row["avg_distinct_locations"] for row in eval_results]
        colors = ["blue" if row.get("on_frontier", False) else "grey" for row in eval_results]

        plt.figure(figsize=(8,6))
        plt.scatter(costs, avg_locations, c=colors)
        for row in eval_results:
            moar_accuracy = row.get("moar_accuracy")
            if moar_accuracy is not None:
                label = f"{row['node_id']} ({moar_accuracy:.2f})"
            else:
                label = row.get("node_id", row.get("file", ""))
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

def _create_medec_plots(eval_results, output_path):
    """Create plots for MEDEC dataset"""
    # Plot Combined Score vs Cost scatter
    try:
        costs = [row["cost"] for row in eval_results]
        scores = [row["combined_score"] for row in eval_results]
        colors = ["blue" if row.get("on_frontier", False) else "grey" for row in eval_results]

        plt.figure(figsize=(8,6))
        plt.scatter(costs, scores, c=colors)
        for row in eval_results:
            moar_accuracy = row.get("moar_accuracy")
            if moar_accuracy is not None:
                label = f"{row['node_id']} ({moar_accuracy:.2f})"
            else:
                label = row.get("node_id", row.get("file", ""))
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

def _create_sustainability_plots(eval_results, output_path):
    """Create plots for sustainability dataset"""
    # Plot Economic Activity Accuracy vs Cost scatter
    try:
        costs = [row["cost"] for row in eval_results]
        accuracies = [row["combined_score"] for row in eval_results]
        colors = ["blue" if row.get("on_frontier", False) else "grey" for row in eval_results]

        plt.figure(figsize=(8,6))
        plt.scatter(costs, accuracies, c=colors)
        for row in eval_results:
            moar_accuracy = row.get("moar_accuracy")
            if moar_accuracy is not None:
                label = f"{row['node_id']} ({moar_accuracy:.2f})"
            else:
                label = row.get("node_id", row.get("file", ""))
            plt.annotate(label, (row["cost"], row["combined_score"]), textcoords="offset points", xytext=(4,4), fontsize=8)

        plt.xlabel("Cost ($)")
        plt.ylabel("Combined Score (Avg of Economic Activity Accuracy and Company Name Accuracy)")
        plt.title("Cost vs Combined Score for all plans")
        plt.grid(True, linestyle="--", alpha=0.5)
        plot_path = output_path / "cost_vs_combined_score.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📈 Scatter plot saved to: {plot_path}")
    except Exception as e:
        print(f"⚠️  Failed to create scatter plot: {e}")

def _create_facility_plots(eval_results, output_path):
    """Create plots for Facility dataset"""
    # Plot Combined Score vs Cost scatter
    try:
        costs = [row["cost"] for row in eval_results]
        scores = [row["combined_score"] for row in eval_results]
        colors = ["blue" if row.get("on_frontier", False) else "grey" for row in eval_results]

        plt.figure(figsize=(8,6))
        plt.scatter(costs, scores, c=colors)
        for row in eval_results:
            moar_accuracy = row.get("moar_accuracy")
            if moar_accuracy is not None:
                label = f"{row['node_id']} ({moar_accuracy:.2f})"
            else:
                label = row.get("node_id", row.get("file", ""))
            plt.annotate(label, (row["cost"], row["combined_score"]), textcoords="offset points", xytext=(4,4), fontsize=8)

        plt.xlabel("Cost ($)")
        plt.ylabel("Combined Score")
        plt.title("Cost vs Combined Score (Urgency + Sentiment + Categories) for all plans")
        plt.grid(True, linestyle="--", alpha=0.5)
        plot_path = output_path / "cost_vs_combined_score.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📈 Scatter plot saved to: {plot_path}")
    except Exception as e:
        print(f"⚠️  Failed to create scatter plot: {e}")

def _create_biodex_plots(eval_results, output_path):
    """Create plots for BioDEX dataset"""
    # Plot RP@10 vs Cost scatter (since we're optimizing for RP@10)
    try:
        costs = [row["cost"] for row in eval_results]
        rp_at_5_scores = [row["avg_rp_at_5"] for row in eval_results]
        colors = ["blue" if row.get("on_frontier", False) else "grey" for row in eval_results]

        plt.figure(figsize=(8,6))
        plt.scatter(costs, rp_at_5_scores, c=colors)
        for row in eval_results:
            moar_accuracy = row.get("moar_accuracy")
            if moar_accuracy is not None:
                label = f"{row['node_id']} ({moar_accuracy:.2f})"
            else:
                label = row.get("node_id", row.get("file", ""))
            plt.annotate(label, (row["cost"], row["avg_rp_at_5"]), textcoords="offset points", xytext=(4,4), fontsize=8)

        plt.xlabel("Cost ($)")
        plt.ylabel("Rank Precision @ 5")
        plt.title("Cost vs RP@5 for all plans")
        plt.grid(True, linestyle="--", alpha=0.5)
        plot_path = output_path / "cost_vs_rp_at_5.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📈 Scatter plot saved to: {plot_path}")
    except Exception as e:
        print(f"⚠️  Failed to create scatter plot: {e}")

def _create_generic_plots(eval_results, output_path, metric_key="accuracy"):
    """Create plots for custom evaluation functions"""
    # Plot metric vs Cost scatter
    try:
        costs = [row["cost"] for row in eval_results]
        accuracies = [row.get(metric_key, 0.0) for row in eval_results]
        colors = ["blue" if row.get("on_frontier", False) else "grey" for row in eval_results]

        plt.figure(figsize=(8,6))
        plt.scatter(costs, accuracies, c=colors)
        for row in eval_results:
            moar_accuracy = row.get("moar_accuracy")
            if moar_accuracy is not None:
                label = f"{row.get('node_id', row.get('file', ''))} ({moar_accuracy:.2f})"
            else:
                label = row.get("node_id", row.get("file", ""))
            plt.annotate(label, (row["cost"], row.get(metric_key, 0.0)), textcoords="offset points", xytext=(4,4), fontsize=8)

        plt.xlabel("Cost ($)")
        plt.ylabel(metric_key.replace("_", " ").title())
        plt.title(f"Cost vs {metric_key.replace('_', ' ').title()} for all plans")
        plt.grid(True, linestyle="--", alpha=0.5)
        plot_path = output_path / f"cost_vs_{metric_key}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📈 Scatter plot saved to: {plot_path}")
    except Exception as e:
        print(f"⚠️  Failed to create scatter plot: {e}")
