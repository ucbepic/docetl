"""
Evaluation utility functions for DocETL.
"""

import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from docetl.console import DOCETL_CONSOLE


def register_eval(
    func: Callable[[str, str], Dict[str, Any]]
) -> Callable[[str, str], Dict[str, Any]]:
    """
    Decorator to mark a function as a DocETL evaluation function.

    The decorated function should take two arguments (dataset_file_path, results_file_path) and return
    a dictionary of evaluation metrics.

    Example:
        @docetl.register_eval
        def evaluate_results(dataset_file_path: str, results_file_path: str) -> Dict[str, Any]:
            # ... evaluation logic ...
            return {"score": 0.95}
    """
    func._docetl_eval = True
    return func


def load_custom_evaluate_func(
    evaluation_file_path: str, dataset_file_path: str
) -> Callable[[str], Dict[str, Any]]:
    """
    Load a custom evaluation function from a Python file and wrap it to pass dataset_file_path.

    The file should contain a function decorated with @docetl.register_eval.
    If multiple functions are decorated, an error is raised.

    Args:
        evaluation_file_path: Path to a Python file containing a function decorated with @docetl.register_eval
        dataset_file_path: Path to the dataset file to pass to the evaluation function

    Returns:
        callable: Wrapped evaluation function that takes (results_file_path: str) -> dict

    Raises:
        ValueError: If the file doesn't exist, doesn't contain a decorated function, or has multiple decorated functions
    """
    func_path = Path(evaluation_file_path)
    if not func_path.exists():
        raise ValueError(f"Evaluation file not found: {evaluation_file_path}")

    # Use a unique module name based on the file path to avoid conflicts
    module_name = f"docetl_eval_{func_path.stem}_{hash(str(func_path))}"
    spec = importlib.util.spec_from_file_location(module_name, func_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load module from: {evaluation_file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find all functions decorated with @docetl.register_eval
    eval_functions = []
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if hasattr(obj, "_docetl_eval") and obj._docetl_eval:
            eval_functions.append((name, obj))

    if len(eval_functions) == 0:
        raise ValueError(
            f"Module {evaluation_file_path} must contain a function decorated with @docetl.register_eval. "
            f"Found functions: {[name for name, _ in inspect.getmembers(module, inspect.isfunction)]}"
        )

    if len(eval_functions) > 1:
        function_names = [name for name, _ in eval_functions]
        raise ValueError(
            f"Module {evaluation_file_path} contains multiple functions decorated with @docetl.register_eval: {function_names}. "
            f"Only one evaluation function is allowed per file."
        )

    # Wrap the function to pass dataset_file_path
    original_func = eval_functions[0][1]

    def wrapped_func(results_file_path: str) -> Dict[str, Any]:
        return original_func(dataset_file_path, results_file_path)

    return wrapped_func


def _extract_node_data(item: Any) -> tuple[Optional[str], Dict[str, Any]]:
    """Extract node data from either a node object or a dict/file path."""
    if hasattr(item, "result_path"):
        jf = item.result_path
        node_data = {
            "node_id": item.get_id(),
            "cost": item.cost,
            "visits": getattr(item, "visits", 0),
            "value": getattr(item, "value", 0),
        }
    else:
        jf = item.get("file_path") if isinstance(item, dict) else item
        node_data = {
            "node_id": (
                item.get("node_id", "unknown") if isinstance(item, dict) else "unknown"
            ),
            "cost": item.get("cost", 0.0) if isinstance(item, dict) else 0.0,
            "visits": item.get("visits", 0) if isinstance(item, dict) else 0,
            "value": item.get("value", 0) if isinstance(item, dict) else 0,
        }
    return jf, node_data


def _get_display_path(jf: str, output_path: Path) -> str:
    """Get display path for a result file."""
    jp = Path(jf).resolve()
    op_root = output_path.resolve()
    if hasattr(jp, "is_relative_to") and jp.is_relative_to(op_root):
        return str(jp.relative_to(op_root))
    else:
        return jp.name


def _add_frontier_info(result: Dict[str, Any], item: Any) -> Dict[str, Any]:
    """Add frontier information if available."""
    if hasattr(item, "result_path"):
        result.update(
            {
                "moar_accuracy": getattr(item, "moar_accuracy", None),
                "on_frontier": getattr(item, "on_frontier", False),
            }
        )
    return result


def identify_pareto_frontier(
    eval_results: List[Dict[str, Any]], metric_key: str
) -> List[Dict[str, Any]]:
    """
    Identify the Pareto frontier for evaluation results based on accuracy vs cost.

    Args:
        eval_results: List of evaluation results with cost and accuracy metrics
        metric_key: Key to use for accuracy metric

    Returns:
        Updated eval_results with 'on_frontier' field set to True/False

    Raises:
        KeyError: If required metrics are missing from results
    """
    if not eval_results:
        return eval_results

    # Filter out results that don't have the required metrics
    valid_results = [r for r in eval_results if metric_key in r and "cost" in r]
    if not valid_results:
        DOCETL_CONSOLE.log(
            f"[yellow]⚠️  No valid results with {metric_key} and cost metrics[/yellow]"
        )
        return eval_results

    # Validate that all results have the required metrics
    for r in valid_results:
        if metric_key not in r:
            raise KeyError(
                f"Missing required accuracy metric '{metric_key}' in evaluation result. "
                f"Available keys: {list(r.keys())}. "
                f"This metric is required for Pareto frontier identification."
            )
        if "cost" not in r:
            raise KeyError(
                f"Missing required 'cost' metric in evaluation result. "
                f"Available keys: {list(r.keys())}"
            )

    # Sort by cost (ascending) and accuracy (descending for maximization)
    valid_results.sort(key=lambda x: (x["cost"], -x[metric_key]))

    # Identify Pareto frontier: points that are not dominated by any other point
    frontier = []
    for i, candidate in enumerate(valid_results):
        is_dominated = False
        for j, other in enumerate(valid_results):
            if i == j:
                continue
            # Check if other point dominates candidate
            # Dominated if: other has lower cost AND higher/equal accuracy, OR same cost AND higher accuracy
            if (
                other["cost"] < candidate["cost"]
                and other[metric_key] >= candidate[metric_key]
            ) or (
                other["cost"] == candidate["cost"]
                and other[metric_key] > candidate[metric_key]
            ):
                is_dominated = True
                break

        if not is_dominated:
            frontier.append(candidate)

    # Mark all results with frontier status
    frontier_set = set(id(f) for f in frontier)
    for r in eval_results:
        r["on_frontier"] = id(r) in frontier_set

    return eval_results


def print_pareto_frontier_summary(
    eval_results: List[Dict[str, Any]],
    metric_key: str,
    dataset_name: Optional[str] = None,
) -> None:
    """
    Print a summary of the Pareto frontier points.

    Args:
        eval_results: List of evaluation results with 'on_frontier' field
        metric_key: Key to use for accuracy metric
        dataset_name: Optional dataset name for display purposes

    Raises:
        KeyError: If required metrics are missing from frontier points
    """
    frontier_points = [r for r in eval_results if r.get("on_frontier", False)]

    if not frontier_points:
        dataset_str = f" for {dataset_name}" if dataset_name else ""
        DOCETL_CONSOLE.log(
            f"[yellow]📊 No Pareto frontier points found{dataset_str}[/yellow]"
        )
        return

    # Sort frontier points by cost for better display
    frontier_points.sort(key=lambda x: x["cost"])

    dataset_str = f" for {dataset_name.upper()}" if dataset_name else ""
    DOCETL_CONSOLE.log(f"\n🏆 Pareto Frontier Summary{dataset_str}:")
    DOCETL_CONSOLE.log("=" * 60)
    DOCETL_CONSOLE.log(
        f"{'Rank':<4} {'Cost ($)':<10} {metric_key.upper():<15} {'File':<30}"
    )
    DOCETL_CONSOLE.log("=" * 60)

    for i, point in enumerate(frontier_points, 1):
        if "cost" not in point:
            raise KeyError(
                f"Missing required 'cost' metric in frontier point. "
                f"Available keys: {list(point.keys())}"
            )
        if metric_key not in point:
            raise KeyError(
                f"Missing required accuracy metric '{metric_key}' in frontier point. "
                f"Available keys: {list(point.keys())}. "
                f"This metric is required for Pareto frontier summary."
            )
        cost = point["cost"]
        accuracy = point[metric_key]
        file_name = point.get("file", "unknown")
        DOCETL_CONSOLE.log(f"{i:<4} ${cost:<9.4f} {accuracy:<15.4f} {file_name:<30}")

    DOCETL_CONSOLE.log("=" * 60)
    DOCETL_CONSOLE.log(f"Total frontier points: {len(frontier_points)}")


def save_pareto_frontier_results(
    eval_results: List[Dict[str, Any]],
    output_path: Path,
    metric_key: str,
    dataset_name: Optional[str] = None,
) -> None:
    """
    Save Pareto frontier results to a separate JSON file for analysis.

    Args:
        eval_results: List of evaluation results with 'on_frontier' field
        output_path: Output directory path
        metric_key: Key to use for accuracy metric
        dataset_name: Optional dataset name

    Raises:
        KeyError: If required metrics are missing from frontier points
    """
    frontier_points = [r for r in eval_results if r.get("on_frontier", False)]

    if not frontier_points:
        return

    # Sort frontier points by cost
    frontier_points.sort(key=lambda x: x["cost"])

    # Add rank and accuracy metric information
    for i, point in enumerate(frontier_points):
        point["rank"] = i + 1
        point["accuracy_metric"] = metric_key

    # Calculate cost-effectiveness ratios between consecutive points
    cost_effectiveness_analysis = []
    for i in range(len(frontier_points) - 1):
        curr = frontier_points[i]
        next_point = frontier_points[i + 1]

        if "cost" not in curr or "cost" not in next_point:
            raise KeyError(
                f"Missing required 'cost' metric in frontier point. "
                f"Available keys: {list(curr.keys() if 'cost' not in curr else next_point.keys())}"
            )
        if metric_key not in curr or metric_key not in next_point:
            raise KeyError(
                f"Missing required accuracy metric '{metric_key}' in frontier point. "
                f"Available keys: {list(curr.keys() if metric_key not in curr else next_point.keys())}. "
                f"This metric is required for cost-effectiveness analysis."
            )
        cost_diff = next_point["cost"] - curr["cost"]
        accuracy_diff = next_point[metric_key] - curr[metric_key]

        if cost_diff > 0 and accuracy_diff > 0:
            cost_effectiveness = cost_diff / accuracy_diff
            cost_effectiveness_analysis.append(
                {
                    "from_file": curr["file"],
                    "to_file": next_point["file"],
                    "cost_increase": cost_diff,
                    "accuracy_increase": accuracy_diff,
                    "cost_per_unit_improvement": cost_effectiveness,
                }
            )

    # Create frontier summary
    frontier_summary = {
        "accuracy_metric": metric_key,
        "total_frontier_points": len(frontier_points),
        "frontier_points": frontier_points,
        "cost_effectiveness_analysis": cost_effectiveness_analysis,
    }
    if dataset_name:
        frontier_summary["dataset"] = dataset_name

    # Save to file
    frontier_file = output_path / "pareto_frontier.json"
    with open(frontier_file, "w") as f:
        json.dump(frontier_summary, f, indent=2)

    DOCETL_CONSOLE.log(f"📊 Pareto frontier results written to {frontier_file}")


def run_evaluation(
    nodes_or_files: List[Any],
    evaluate_func: Callable[[str], Dict[str, Any]],
    metric_key: str,
    output_path: Path,
    dataset_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run evaluation on a set of nodes or files using a custom evaluation function.

    This is a general-purpose evaluation function that does not depend on
    experiment-specific datasets. It processes nodes/files, extracts metrics,
    identifies the Pareto frontier, and saves results.

    Args:
        nodes_or_files: List of node objects (with result_path) or file paths
        evaluate_func: Evaluation function (results_file_path: str) -> dict
        metric_key: Key to extract from evaluation results for accuracy metric
        output_path: Path to save evaluation results
        dataset_name: Optional dataset name for display purposes

    Returns:
        List of evaluation results with 'on_frontier' field set

    Raises:
        ValueError: If metric_key is not provided
        KeyError: If required metrics are missing from evaluation results
    """
    if not metric_key:
        raise ValueError("metric_key must be provided")

    eval_results = []

    # Process evaluation items
    for item in nodes_or_files:
        jf, node_data = _extract_node_data(item)
        if jf is None or not Path(jf).exists():
            continue

        try:
            metrics = evaluate_func(jf)
            display_path = _get_display_path(jf, output_path)

            # Extract the custom metric
            accuracy_value = metrics.get(metric_key)
            if accuracy_value is None:
                DOCETL_CONSOLE.log(
                    f"[yellow]⚠️  Warning: Metric key '{metric_key}' not found in evaluation results for {jf}. "
                    f"Available keys: {list(metrics.keys())}[/yellow]"
                )
                # Try to find a numeric value as fallback
                accuracy_value = next(
                    (v for v in metrics.values() if isinstance(v, (int, float))), None
                )
                if accuracy_value is None:
                    DOCETL_CONSOLE.log(
                        f"[red]❌ Skipping {jf}: No valid accuracy metric found[/red]"
                    )
                    continue

            result = {
                "file": display_path,
                metric_key: accuracy_value,
                **metrics,  # Include all metrics from custom function
                **node_data,
            }
            result = _add_frontier_info(result, item)
            eval_results.append(result)
        except Exception as e:
            DOCETL_CONSOLE.log(f"[red]   ⚠️  Evaluation failed for {jf}: {e}[/red]")

    # Identify Pareto frontier
    if eval_results:
        DOCETL_CONSOLE.log("\n🔍 Identifying Pareto frontier...")
        eval_results = identify_pareto_frontier(eval_results, metric_key)

        # Print Pareto frontier summary
        print_pareto_frontier_summary(eval_results, metric_key, dataset_name)

        # Save Pareto frontier results to separate file
        save_pareto_frontier_results(
            eval_results, output_path, metric_key, dataset_name
        )

    # Save evaluation results
    if eval_results:
        eval_out_file = output_path / "evaluation_metrics.json"
        with open(eval_out_file, "w") as f:
            json.dump(eval_results, f, indent=2)
        DOCETL_CONSOLE.log(f"📊 Evaluation results written to {eval_out_file}")

    return eval_results
