"""
Dataset utility functions for DocETL.
"""

import json
from typing import Any, Dict, List

import yaml


def compute_dataset_stats(
    data: List[Dict[str, Any]], dataset_name: str = "data"
) -> str:
    """
    Compute statistics for a dataset by analyzing the actual data.

    Args:
        data: List of data records
        dataset_name: Name of the dataset

    Returns:
        str: Formatted dataset statistics
    """
    if not data:
        return (
            f"Dataset: {dataset_name}\nType: file\nRecords loaded: 0\nNo data available"
        )

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
                    field_stats[key] = {
                        "total_chars": 0,
                        "count": 0,
                        "type": type(value).__name__,
                    }

                if isinstance(value, str):
                    char_count = len(value)
                    field_stats[key]["total_chars"] += char_count
                    field_stats[key]["count"] += 1
                    total_tokens += (
                        char_count / 4
                    )  # 4 characters per token approximation
                elif isinstance(value, (int, float)):
                    # Numbers are typically short, estimate as ~5 characters
                    field_stats[key]["total_chars"] += 5
                    field_stats[key]["count"] += 1
                    total_tokens += 1.25
                elif isinstance(value, list):
                    # For lists, estimate based on string representation
                    str_repr = str(value)
                    char_count = len(str_repr)
                    field_stats[key]["total_chars"] += char_count
                    field_stats[key]["count"] += 1
                    total_tokens += char_count / 4

    # Format the output
    stats_lines = [
        f"Dataset: {dataset_name}",
        "Type: file",
        f"Records loaded: {num_records}",
        "Input schema:",
    ]

    for field, stats in field_stats.items():
        if stats["count"] > 0:
            avg_tokens = (stats["total_chars"] / stats["count"]) / 4
            field_type = "string" if stats["type"] in ["str"] else stats["type"]
            stats_lines.append(
                f"    {field}: {field_type} (avg: {avg_tokens:.1f} tokens)"
            )

    stats_lines.append(f"Total tokens: {int(total_tokens):,}")

    return "\n        ".join(stats_lines)


def get_dataset_stats(yaml_path: str, dataset_name: str | None = None) -> str:
    """
    Get dataset statistics by loading and analyzing the actual data from YAML config.

    Args:
        yaml_path: Path to the YAML configuration file
        dataset_name: Optional name of the dataset (if not provided, uses first dataset in config)

    Returns:
        str: Formatted dataset statistics
    """
    # Load the YAML config to get the data path
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract dataset info from config
    datasets = config.get("datasets", {})
    if not datasets:
        return f"Dataset: {dataset_name or 'unknown'}\nType: file\nRecords loaded: 0\nNo datasets found in config"

    # Get the first dataset (or specified dataset)
    if dataset_name and dataset_name in datasets:
        dataset_config = datasets[dataset_name]
        actual_dataset_name = dataset_name
    else:
        actual_dataset_name, dataset_config = next(iter(datasets.items()))

    data_path = dataset_config.get("path")

    if not data_path:
        return f"Dataset: {actual_dataset_name}\nType: file\nRecords loaded: 0\nNo data path found"

    # Load the data
    try:
        with open(data_path, "r") as f:
            data = json.load(f)

        return compute_dataset_stats(data, actual_dataset_name)

    except Exception as e:
        return f"Dataset: {actual_dataset_name}\nType: file\nRecords loaded: 0\nError loading data: {e}"
