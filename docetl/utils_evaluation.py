"""
Evaluation utility functions for DocETL.
"""

import importlib.util
import inspect
from pathlib import Path
from typing import Any, Callable, Dict


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
