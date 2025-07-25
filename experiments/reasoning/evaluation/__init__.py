"""Evaluation package.

Provides dataset-specific scorers and utility helpers.
"""

from pathlib import Path
from typing import Callable, Dict

# Mapping of dataset name -> scorer callable (results_file, ground_truth_path) -> metrics dict
_SCORERS: Dict[str, Callable[[str, str], dict]] = {}


def register_scorer(dataset: str, fn: Callable[[str, str], dict]):
    """Register a scorer for a dataset so evaluation CLI can discover it."""
    _SCORERS[dataset.lower()] = fn


def get_scorer(dataset: str) -> Callable[[str, str], dict]:
    dataset = dataset.lower()
    if dataset not in _SCORERS:
        raise ValueError(f"No scorer registered for dataset '{dataset}'.")
    return _SCORERS[dataset]


# ---------------------------------------------------------------------------
# Register built-in dataset scorers
# ---------------------------------------------------------------------------

from .cuad import evaluate_results as _cuad_evaluate


def _cuad_scorer(results_file: str, ground_truth: str):
    # method_name is informational only
    return _cuad_evaluate("docetl", results_file, ground_truth)


register_scorer("cuad", _cuad_scorer) 