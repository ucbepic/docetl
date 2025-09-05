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

dataset_accuracy_metrics = {
    "cuad": "avg_f1",
    "blackvault": "avg_distinct_locations", 
    "game_reviews": "weighted_score",
    "medec": "combined_score",
    "sustainability": "economic_activity_accuracy",
    "biodex": "avg_rp_at_5"
}
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
    
    # Calculate cost-effectiveness ratios
    if len(frontier_points) > 1:
        print(f"\n💰 Cost-Effectiveness Analysis:")
        for i in range(len(frontier_points) - 1):
            curr = frontier_points[i]
            next_point = frontier_points[i + 1]
            
            cost_diff = next_point["cost"] - curr["cost"]
            accuracy_diff = next_point[accuracy_metric] - curr[accuracy_metric]
            
            if cost_diff > 0 and accuracy_diff > 0:
                cost_effectiveness = cost_diff / accuracy_diff
                print(f"   {curr['file']} → {next_point['file']}: +${cost_diff:.4f} for +{accuracy_diff:.4f} {accuracy_metric} (${cost_effectiveness:.4f} per unit improvement)")

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
    
    elif dataset.lower() == "sustainability":
        def sustainability_eval_func(method_name, results_file_path):
            ground_truth_path = "experiments/reasoning/data/company_reports_gt.json"
            return sustainability_evaluate(method_name, results_file_path, ground_truth_path)
        return sustainability_eval_func
    
    elif dataset.lower() == "biodex":
        def biodex_eval_func(method_name, results_file_path):
            return biodex_evaluate(method_name, results_file_path)
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

def run_dataset_evaluation(dataset, nodes_or_files, output_path, ground_truth_path=None, method_name="docetl", root_cost=None):
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
            pareto_auc = _create_cuad_plots_and_auc(eval_results, output_path, root_cost)
            
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
            pareto_auc = _create_blackvault_plots_and_auc(eval_results, output_path, root_cost)
    
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
            pareto_auc = _create_game_reviews_plots_and_auc(eval_results, output_path, root_cost)
    
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
            pareto_auc = _create_medec_plots_and_auc(eval_results, output_path, root_cost)
    
    elif dataset.lower() == "sustainability":
        if ground_truth_path is None:
            default_gt = Path("experiments/reasoning/data/company_reports_gt.json")
            ground_truth_path = str(default_gt)

        print(f"\n🧪 Evaluating sustainability analysis results ...")

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
                metrics = sustainability_evaluate(method_name, jf, ground_truth_path)
                jp = Path(jf).resolve()
                op_root = output_path.resolve()
                if hasattr(jp, "is_relative_to") and jp.is_relative_to(op_root):
                    display_path = str(jp.relative_to(op_root))
                else:
                    display_path = jp.name

                result = {
                    "file": display_path,
                    "combined_score": metrics["combined_score"],
                    "company_name_accuracy": metrics["company_name_accuracy"],
                    "total_companies_processed": metrics["total_companies_processed"],
                    "avg_findings_length": metrics["avg_findings_length"],
                    "total_economic_activities": metrics["total_economic_activities"],
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
            pareto_auc = _create_sustainability_plots_and_auc(eval_results, output_path, root_cost)
    
    elif dataset.lower() == "biodex":
        print(f"\n🧪 Evaluating BioDEX reaction extraction results ...")

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
                metrics = biodex_evaluate(method_name, jf)
                jp = Path(jf).resolve()
                op_root = output_path.resolve()
                if hasattr(jp, "is_relative_to") and jp.is_relative_to(op_root):
                    display_path = str(jp.relative_to(op_root))
                else:
                    display_path = jp.name

                result = {
                    "file": display_path,
                    "avg_rp_at_5": metrics["avg_rp_at_5"],
                    "avg_rp_at_10": metrics["avg_rp_at_10"],
                    "avg_term_recall": metrics["avg_term_recall"],
                    "total_documents": metrics["total_documents"],
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
            pareto_auc = _create_biodex_plots_and_auc(eval_results, output_path, root_cost)
    
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
    
    return eval_results, pareto_auc

def _create_cuad_plots_and_auc(eval_results, output_path, root_cost=None):
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
            mcts_accuracy = row.get("mcts_accuracy")
            if mcts_accuracy is not None:
                label = f"{row['node_id']} ({mcts_accuracy:.2f})"
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
    
    # Compute Hypervolume with reference point (accuracy=0, cost=baseline_cost*10)
    try:
        frontier_points = [row for row in eval_results if row.get("on_frontier", False)]
        if frontier_points:
            # Use highest cost across all points as reference
            ref_cost = max(row["cost"] for row in eval_results)
            print(f"Using highest cost across all points as reference: {ref_cost:.2f}")
            ref_accuracy = 0.0
            
            # Sort frontier points by cost (ascending)
            frontier_points.sort(key=lambda r: r["cost"])
            
            hypervolume = 0.0
            
            # Calculate trapezoid areas between consecutive frontier points
            for i in range(len(frontier_points) - 1):
                curr_point = frontier_points[i]
                next_point = frontier_points[i + 1]
                
                if (curr_point["cost"] < ref_cost and curr_point["f1"] > ref_accuracy and
                    next_point["cost"] < ref_cost and next_point["f1"] > ref_accuracy):
                    
                    # Trapezoid area: (height1 + height2) * width / 2
                    width = next_point["cost"] - curr_point["cost"]
                    height1 = curr_point["f1"] - ref_accuracy
                    height2 = next_point["f1"] - ref_accuracy
                    
                    if width > 0 and height1 > 0 and height2 > 0:
                        trapezoid_area = (height1 + height2) * width / 2
                        hypervolume += trapezoid_area
            
            # Add final rectangle from last point to reference cost
            if frontier_points:
                last_point = frontier_points[-1]
                if last_point["cost"] < ref_cost and last_point["f1"] > ref_accuracy:
                    final_width = ref_cost - last_point["cost"]
                    final_height = last_point["f1"] - ref_accuracy
                    if final_width > 0 and final_height > 0:
                        final_rectangle = final_width * final_height
                        hypervolume += final_rectangle
            
            pareto_auc = hypervolume
            print(f"📐 Hypervolume (ref_point=[{ref_accuracy}, {ref_cost:.2f}]): {hypervolume:.4f}")
        else:
            pareto_auc = 0.0
    except Exception as e:
        print(f"⚠️  Failed to compute Hypervolume: {e}")
        pareto_auc = None
    
    return pareto_auc

def _create_game_reviews_plots_and_auc(eval_results, output_path, root_cost=None):
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
            mcts_accuracy = row.get("mcts_accuracy")
            if mcts_accuracy is not None:
                label = f"{row['node_id']} ({mcts_accuracy:.2f})"
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
    
    # Compute Hypervolume with reference point (accuracy=0, cost=baseline_cost*10)
    try:
        frontier_points = [row for row in eval_results if row.get("on_frontier", False)]
        if frontier_points:
            # Use highest cost across all points as reference
            ref_cost = max(row["cost"] for row in eval_results)
            print(f"Using highest cost across all points as reference: {ref_cost:.2f}")
            ref_accuracy = 0.0
            
            # Sort frontier points by cost (ascending)
            frontier_points.sort(key=lambda r: r["cost"])
            
            hypervolume = 0.0
            
            # Calculate trapezoid areas between consecutive frontier points
            for i in range(len(frontier_points) - 1):
                curr_point = frontier_points[i]
                next_point = frontier_points[i + 1]
                
                if (curr_point["cost"] < ref_cost and curr_point["combined_accuracy_score"] > ref_accuracy and
                    next_point["cost"] < ref_cost and next_point["combined_accuracy_score"] > ref_accuracy):
                    
                    # Trapezoid area: (height1 + height2) * width / 2
                    width = next_point["cost"] - curr_point["cost"]
                    height1 = curr_point["combined_accuracy_score"] - ref_accuracy
                    height2 = next_point["combined_accuracy_score"] - ref_accuracy
                    
                    if width > 0 and height1 > 0 and height2 > 0:
                        trapezoid_area = (height1 + height2) * width / 2
                        hypervolume += trapezoid_area
            
            # Add final rectangle from last point to reference cost
            if frontier_points:
                last_point = frontier_points[-1]
                if last_point["cost"] < ref_cost and last_point["combined_accuracy_score"] > ref_accuracy:
                    final_width = ref_cost - last_point["cost"]
                    final_height = last_point["combined_accuracy_score"] - ref_accuracy
                    if final_width > 0 and final_height > 0:
                        final_rectangle = final_width * final_height
                        hypervolume += final_rectangle
            
            pareto_auc = hypervolume
            print(f"📐 Hypervolume (ref_point=[{ref_accuracy}, {ref_cost:.2f}]): {hypervolume:.4f}")
        else:
            pareto_auc = 0.0
    except Exception as e:
        print(f"⚠️  Failed to compute Hypervolume: {e}")
        pareto_auc = None
    
    return pareto_auc


def _create_blackvault_plots_and_auc(eval_results, output_path, root_cost=None):

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
            mcts_accuracy = row.get("mcts_accuracy")
            if mcts_accuracy is not None:
                label = f"{row['node_id']} ({mcts_accuracy:.2f})"
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
    
    # Compute Hypervolume with reference point (accuracy=0, cost=baseline_cost*10)
    try:
        frontier_points = [row for row in eval_results if row.get("on_frontier", False)]
        if frontier_points:
            # Use highest cost across all points as reference
            ref_cost = max(row["cost"] for row in eval_results)
            print(f"Using highest cost across all points as reference: {ref_cost:.2f}")
            ref_accuracy = 0.0
            
            # Sort frontier points by cost (ascending)
            frontier_points.sort(key=lambda r: r["cost"])
            
            hypervolume = 0.0
            
            # Calculate trapezoid areas between consecutive frontier points
            for i in range(len(frontier_points) - 1):
                curr_point = frontier_points[i]
                next_point = frontier_points[i + 1]
                
                if (curr_point["cost"] < ref_cost and curr_point["avg_distinct_locations"] > ref_accuracy and
                    next_point["cost"] < ref_cost and next_point["avg_distinct_locations"] > ref_accuracy):
                    
                    # Trapezoid area: (height1 + height2) * width / 2
                    width = next_point["cost"] - curr_point["cost"]
                    height1 = curr_point["avg_distinct_locations"] - ref_accuracy
                    height2 = next_point["avg_distinct_locations"] - ref_accuracy
                    
                    if width > 0 and height1 > 0 and height2 > 0:
                        trapezoid_area = (height1 + height2) * width / 2
                        hypervolume += trapezoid_area
            
            # Add final rectangle from last point to reference cost
            if frontier_points:
                last_point = frontier_points[-1]
                if last_point["cost"] < ref_cost and last_point["avg_distinct_locations"] > ref_accuracy:
                    final_width = ref_cost - last_point["cost"]
                    final_height = last_point["avg_distinct_locations"] - ref_accuracy
                    if final_width > 0 and final_height > 0:
                        final_rectangle = final_width * final_height
                        hypervolume += final_rectangle
            
            pareto_auc = hypervolume
            print(f"📐 Hypervolume (ref_point=[{ref_accuracy}, {ref_cost:.2f}]): {hypervolume:.4f}")
        else:
            pareto_auc = 0.0
    except Exception as e:
        print(f"⚠️  Failed to compute Hypervolume: {e}")
        pareto_auc = None
    
    return pareto_auc


def _create_medec_plots_and_auc(eval_results, output_path, root_cost=None):

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
            mcts_accuracy = row.get("mcts_accuracy")
            if mcts_accuracy is not None:
                label = f"{row['node_id']} ({mcts_accuracy:.2f})"
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
    
    # Compute Hypervolume with reference point (accuracy=0, cost=baseline_cost*10)
    try:
        frontier_points = [row for row in eval_results if row.get("on_frontier", False)]
        if frontier_points:
            # Use highest cost across all points as reference
            ref_cost = max(row["cost"] for row in eval_results)
            print(f"Using highest cost across all points as reference: {ref_cost:.2f}")
            ref_accuracy = 0.0
            
            # Sort frontier points by cost (ascending)
            frontier_points.sort(key=lambda r: r["cost"])
            
            hypervolume = 0.0
            
            # Calculate trapezoid areas between consecutive frontier points
            for i in range(len(frontier_points) - 1):
                curr_point = frontier_points[i]
                next_point = frontier_points[i + 1]
                
                if (curr_point["cost"] < ref_cost and curr_point["combined_score"] > ref_accuracy and
                    next_point["cost"] < ref_cost and next_point["combined_score"] > ref_accuracy):
                    
                    # Trapezoid area: (height1 + height2) * width / 2
                    width = next_point["cost"] - curr_point["cost"]
                    height1 = curr_point["combined_score"] - ref_accuracy
                    height2 = next_point["combined_score"] - ref_accuracy
                    
                    if width > 0 and height1 > 0 and height2 > 0:
                        trapezoid_area = (height1 + height2) * width / 2
                        hypervolume += trapezoid_area
            
            # Add final rectangle from last point to reference cost
            if frontier_points:
                last_point = frontier_points[-1]
                if last_point["cost"] < ref_cost and last_point["combined_score"] > ref_accuracy:
                    final_width = ref_cost - last_point["cost"]
                    final_height = last_point["combined_score"] - ref_accuracy
                    if final_width > 0 and final_height > 0:
                        final_rectangle = final_width * final_height
                        hypervolume += final_rectangle
                        
            
            pareto_auc = hypervolume
            print(f"📐 Hypervolume (ref_point=[{ref_accuracy}, {ref_cost:.2f}]): {hypervolume:.4f}")
        else:
            pareto_auc = 0.0
    except Exception as e:
        print(f"⚠️  Failed to compute Hypervolume: {e}")
        pareto_auc = None
    
    return pareto_auc


def _create_sustainability_plots_and_auc(eval_results, output_path, root_cost=None):
    """Create plots and compute AUC for sustainability dataset"""
    pareto_auc = None
    
    # Plot Economic Activity Accuracy vs Cost scatter
    try:
        costs = [row["cost"] for row in eval_results]
        accuracies = [row["combined_score"] for row in eval_results]
        colors = ["blue" if row.get("on_frontier", False) else "grey" for row in eval_results]

        plt.figure(figsize=(8,6))
        plt.scatter(costs, accuracies, c=colors)
        for row in eval_results:
            mcts_accuracy = row.get("mcts_accuracy")
            if mcts_accuracy is not None:
                label = f"{row['node_id']} ({mcts_accuracy:.2f})"
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
    
    # Compute Hypervolume with reference point (accuracy=0, cost=baseline_cost*10)
    try:
        frontier_points = [row for row in eval_results if row.get("on_frontier", False)]
        if frontier_points:
            # Use highest cost across all points as reference
            ref_cost = max(row["cost"] for row in eval_results)
            print(f"Using highest cost across all points as reference: {ref_cost:.2f}")
            ref_accuracy = 0.0
            
            # Sort frontier points by cost (ascending)
            frontier_points.sort(key=lambda r: r["cost"])
            
            hypervolume = 0.0
            
            # Calculate trapezoid areas between consecutive frontier points
            for i in range(len(frontier_points) - 1):
                curr_point = frontier_points[i]
                next_point = frontier_points[i + 1]
                
                if (curr_point["cost"] < ref_cost and curr_point["combined_score"] > ref_accuracy and
                    next_point["cost"] < ref_cost and next_point["combined_score"] > ref_accuracy):
                    
                    # Trapezoid area: (height1 + height2) * width / 2
                    width = next_point["cost"] - curr_point["cost"]
                    height1 = curr_point["combined_score"] - ref_accuracy
                    height2 = next_point["combined_score"] - ref_accuracy
                    
                    if width > 0 and height1 > 0 and height2 > 0:
                        trapezoid_area = (height1 + height2) * width / 2
                        hypervolume += trapezoid_area
            
            # Add final rectangle from last point to reference cost
            if frontier_points:
                last_point = frontier_points[-1]
                if last_point["cost"] < ref_cost and last_point["combined_score"] > ref_accuracy:
                    final_width = ref_cost - last_point["cost"]
                    final_height = last_point["combined_score"] - ref_accuracy
                    if final_width > 0 and final_height > 0:
                        final_rectangle = final_width * final_height
                        hypervolume += final_rectangle
            
            pareto_auc = hypervolume
            print(f"📐 Hypervolume (ref_point=[{ref_accuracy}, {ref_cost:.2f}]): {hypervolume:.4f}")
        else:
            pareto_auc = 0.0
    except Exception as e:
        print(f"⚠️  Failed to compute Hypervolume: {e}")
        pareto_auc = None
    
    return pareto_auc

def _create_facility_plots_and_auc(eval_results, output_path, root_cost=None):
    """Create plots and compute AUC for Facility dataset"""
    pareto_auc = None
    
    # Plot Combined Score vs Cost scatter
    try:
        costs = [row["cost"] for row in eval_results]
        scores = [row["combined_score"] for row in eval_results]
        colors = ["blue" if row.get("on_frontier", False) else "grey" for row in eval_results]

        plt.figure(figsize=(8,6))
        plt.scatter(costs, scores, c=colors)
        for row in eval_results:
            mcts_accuracy = row.get("mcts_accuracy")
            if mcts_accuracy is not None:
                label = f"{row['node_id']} ({mcts_accuracy:.2f})"
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
    
    # Compute Hypervolume with reference point (accuracy=0, cost=baseline_cost*10)
    try:
        frontier_points = [row for row in eval_results if row.get("on_frontier", False)]
        if frontier_points:
            # Use root cost if provided, otherwise fall back to minimum cost
            if root_cost is not None:
                ref_cost = root_cost * 10
                print(f"Using root cost reference: {root_cost} -> ref_cost: {ref_cost}")
            else:
                baseline_cost = min(row["cost"] for row in eval_results)
                ref_cost = baseline_cost * 10
                print(f"Using baseline cost reference: {baseline_cost} -> ref_cost: {ref_cost}")
            ref_accuracy = 0.0
            
            # Sort frontier points by cost (ascending)
            frontier_points.sort(key=lambda r: r["cost"])
            
            hypervolume = 0.0
            
            # Calculate trapezoid areas between consecutive frontier points
            for i in range(len(frontier_points) - 1):
                curr_point = frontier_points[i]
                next_point = frontier_points[i + 1]
                
                if (curr_point["cost"] < ref_cost and curr_point["combined_score"] > ref_accuracy and
                    next_point["cost"] < ref_cost and next_point["combined_score"] > ref_accuracy):
                    
                    # Trapezoid area: (height1 + height2) * width / 2
                    width = next_point["cost"] - curr_point["cost"]
                    height1 = curr_point["combined_score"] - ref_accuracy
                    height2 = next_point["combined_score"] - ref_accuracy
                    
                    if width > 0 and height1 > 0 and height2 > 0:
                        trapezoid_area = (height1 + height2) * width / 2
                        hypervolume += trapezoid_area
            
            # Add final rectangle from last point to reference cost
            if frontier_points:
                last_point = frontier_points[-1]
                if last_point["cost"] < ref_cost and last_point["combined_score"] > ref_accuracy:
                    final_width = ref_cost - last_point["cost"]
                    final_height = last_point["combined_score"] - ref_accuracy
                    if final_width > 0 and final_height > 0:
                        final_rectangle = final_width * final_height
                        hypervolume += final_rectangle
            
            pareto_auc = hypervolume
            print(f"📐 Hypervolume (ref_point=[{ref_accuracy}, {ref_cost:.2f}]): {hypervolume:.4f}")
        else:
            pareto_auc = 0.0
    except Exception as e:
        print(f"⚠️  Failed to compute Hypervolume: {e}")
        pareto_auc = None
    
    return pareto_auc

def _create_biodex_plots_and_auc(eval_results, output_path, root_cost=None):
    """Create plots and compute AUC for BioDEX dataset"""
    pareto_auc = None
    
    # Plot RP@10 vs Cost scatter (since we're optimizing for RP@10)
    try:
        costs = [row["cost"] for row in eval_results]
        rp_at_5_scores = [row["avg_rp_at_5"] for row in eval_results]
        colors = ["blue" if row.get("on_frontier", False) else "grey" for row in eval_results]

        plt.figure(figsize=(8,6))
        plt.scatter(costs, rp_at_5_scores, c=colors)
        for row in eval_results:
            mcts_accuracy = row.get("mcts_accuracy")
            if mcts_accuracy is not None:
                label = f"{row['node_id']} ({mcts_accuracy:.2f})"
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
    
    # Compute Hypervolume with reference point (accuracy=0, cost=baseline_cost*10)
    try:
        frontier_points = [row for row in eval_results if row.get("on_frontier", False)]
        if frontier_points:
            # Use highest cost across all points as reference
            ref_cost = max(row["cost"] for row in eval_results)
            print(f"Using highest cost across all points as reference: {ref_cost:.2f}")
            ref_accuracy = 0.0
            
            # Sort frontier points by cost (ascending)
            frontier_points.sort(key=lambda r: r["cost"])
            
            hypervolume = 0.0
            
            # Calculate trapezoid areas between consecutive frontier points
            for i in range(len(frontier_points) - 1):
                curr_point = frontier_points[i]
                next_point = frontier_points[i + 1]
                
                if (curr_point["cost"] < ref_cost and curr_point["avg_rp_at_5"] > ref_accuracy and
                    next_point["cost"] < ref_cost and next_point["avg_rp_at_5"] > ref_accuracy):
                    
                    # Trapezoid area: (height1 + height2) * width / 2
                    width = next_point["cost"] - curr_point["cost"]
                    height1 = curr_point["avg_rp_at_5"] - ref_accuracy
                    height2 = next_point["avg_rp_at_5"] - ref_accuracy
                    
                    if width > 0 and height1 > 0 and height2 > 0:
                        trapezoid_area = (height1 + height2) * width / 2
                        hypervolume += trapezoid_area
            
            # Add final rectangle from last point to reference cost
            if frontier_points:
                last_point = frontier_points[-1]
                if last_point["cost"] < ref_cost and last_point["avg_rp_at_5"] > ref_accuracy:
                    final_width = ref_cost - last_point["cost"]
                    final_height = last_point["avg_rp_at_5"] - ref_accuracy
                    if final_width > 0 and final_height > 0:
                        final_rectangle = final_width * final_height
                        hypervolume += final_rectangle
            
            pareto_auc = hypervolume
            print(f"📐 Hypervolume (ref_point=[{ref_accuracy}, {ref_cost:.2f}]): {hypervolume:.4f}")
        else:
            pareto_auc = 0.0
    except Exception as e:
        print(f"⚠️  Failed to compute Hypervolume: {e}")
        pareto_auc = None
    
    return pareto_auc

