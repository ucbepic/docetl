import json
from typing import Any, Dict, List

import tiktoken
from dotenv import load_dotenv

from docetl.dataset import Dataset, create_parsing_tool_map
from docetl.utils import load_config


def extract_input_schema(
    data: List[Dict[str, Any]], max_samples: int = 10
) -> Dict[str, Dict[str, Any]]:
    """
    Extract the input schema from JSON data by analyzing the structure.

    Args:
        data: List of dictionaries containing the dataset
        max_samples: Maximum number of records to analyze (default: 10)

    Returns:
        Dict[str, Dict[str, Any]]: Schema mapping field names to their type and token info
    """
    if not data:
        return {}

    # Sample records for analysis (to avoid processing entire large datasets)
    sample_size = min(max_samples, len(data))
    sample_data = data[:sample_size]

    schema = {}

    def count_field_tokens(value: Any, model="gpt-4o") -> int:
        """Count tokens for a single field value."""
        try:
            encoding = tiktoken.encoding_for_model(model)
            if value is None:
                return 0
            elif isinstance(value, (dict, list)):
                value_str = json.dumps(value, ensure_ascii=False)
            else:
                value_str = str(value)
            tokens = encoding.encode(value_str)
            return len(tokens)
        except Exception:
            return 0

    def infer_type(value: Any) -> str:
        """Infer the type of a value."""
        if value is None:
            return "string"  # Default to string for null values
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "number"
        elif isinstance(value, list):
            if not value:
                return "list[string]"  # Default for empty lists
            # Check first few elements to determine list type
            sample_elements = value[:3]
            element_types = [infer_type(elem) for elem in sample_elements]
            # If all elements are the same type, use that type
            if len(set(element_types)) == 1:
                return f"list[{element_types[0]}]"
            else:
                return "list[string]"  # Mixed types default to string
        elif isinstance(value, dict):
            return "dict"
        else:
            return "string"

    # Analyze each field across all sample records
    all_fields = set()
    for record in sample_data:
        all_fields.update(record.keys())

    # For each field, determine the most common type and token statistics
    for field in sorted(all_fields):
        field_values = []
        field_tokens = []

        for record in sample_data:
            if field in record:
                value = record[field]
                field_values.append(value)
                token_count = count_field_tokens(value)
                field_tokens.append(token_count)

        if not field_values:
            schema[field] = {"type": "string", "avg_tokens": 0}
            continue

        # Count type occurrences
        type_counts = {}
        for value in field_values:
            value_type = infer_type(value)
            type_counts[value_type] = type_counts.get(value_type, 0) + 1

        # Use the most common type
        most_common_type = max(type_counts.items(), key=lambda x: x[1])[0]
        avg_tokens = sum(field_tokens) / len(field_tokens)

        schema[field] = {"type": most_common_type, "avg_tokens": round(avg_tokens, 1)}

    return schema


def count_tokens_in_data(data, model="gpt-4o"):
    """
    Count the total number of tokens in the data using tiktoken.

    Args:
        data: List of dictionaries containing the dataset
        model: The model to use for tokenization (default: gpt-4o)

    Returns:
        int: Total number of tokens
    """
    try:
        # Get the encoding for the specified model
        encoding = tiktoken.encoding_for_model(model)

        total_tokens = 0

        for item in data:
            # Convert the entire item to a JSON string for tokenization
            item_str = json.dumps(item, ensure_ascii=False)
            tokens = encoding.encode(item_str)
            total_tokens += len(tokens)
        return total_tokens

    except Exception as e:
        print(f"  [WARNING] Could not count tokens: {e}")
        return None


def load_input_doc(yaml_path):
    doc_info = ""
    load_dotenv()
    try:
        config = load_config(yaml_path)
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        return doc_info

    parsing_tool_map = create_parsing_tool_map(config.get("parsing_tools", None))
    datasets_config = config.get("datasets", {})
    if not datasets_config:
        print("[ERROR] No datasets found in config.")
        return doc_info

    for name, dataset_config in datasets_config.items():
        doc_info += f"Dataset: {name}\n"
        try:
            ds = Dataset(
                runner=None,
                type=dataset_config["type"],
                path_or_data=dataset_config["path"],
                source=dataset_config.get("source", "local"),
                parsing=dataset_config.get("parsing", []),
                user_defined_parsing_tool_map=parsing_tool_map,
            )
            data = ds.load()

            if data:
                doc_info += f"  Type: {ds.type}\n"
                doc_info += f"  Records loaded: {len(data)}\n"
                schema = extract_input_schema(data)
                doc_info += "  Input schema:\n"
                for field, field_info in schema.items():
                    doc_info += f"    {field}: {field_info['type']} (avg: {field_info['avg_tokens']} tokens)\n"
                token_count = count_tokens_in_data(data)
                if token_count is not None:
                    doc_info += f"  Total tokens: {token_count:,}\n"

        except Exception as e:
            doc_info += f"  [ERROR] Failed to load dataset '{name}': {e}\n"
    return doc_info
