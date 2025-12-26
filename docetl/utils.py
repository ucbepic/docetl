import json
import math
import re
from enum import Enum
from typing import Any, Callable

import tiktoken
import yaml
from jinja2 import Environment, meta
from litellm import ModelResponse
from litellm import completion_cost as lcc
from lzstring import LZString
from rich.prompt import Confirm


class Decryptor:
    def __init__(self, secret_key: str):
        self.key = secret_key
        self.lz = LZString()

    def decrypt(self, encrypted_data: str) -> str:
        # First decompress the data
        compressed = self.lz.decompressFromBase64(encrypted_data)
        if not compressed:
            raise ValueError("Invalid compressed data")

        # Then decode using the key
        result = ""
        for i in range(len(compressed)):
            char_code = ord(compressed[i]) - ord(self.key[i % len(self.key)])
            result += chr(char_code)

        return result


def decrypt(encrypted_data: str, secret_key: str) -> str:
    if not secret_key:
        return encrypted_data
    return Decryptor(secret_key).decrypt(encrypted_data)


class StageType(Enum):
    SAMPLE_RUN = "sample_run"
    SHOULD_OPTIMIZE = "should_optimize"
    CANDIDATE_PLANS = "candidate_plans"
    EVALUATION_RESULTS = "evaluation_results"
    END = "end"


def get_stage_description(stage_type: StageType) -> str:
    if stage_type == StageType.SAMPLE_RUN:
        return "Running samples..."
    elif stage_type == StageType.SHOULD_OPTIMIZE:
        return "Checking if optimization is needed..."
    elif stage_type == StageType.CANDIDATE_PLANS:
        return "Generating candidate plans..."
    elif stage_type == StageType.EVALUATION_RESULTS:
        return "Evaluating candidate plans..."
    elif stage_type == StageType.END:
        return "Optimization complete!"
    raise ValueError(f"Unknown stage type: {stage_type}")


class CapturedOutput:
    def __init__(self) -> None:
        self.optimizer_output: dict[str, dict[StageType, Any]] = {}
        self.step: str | None = None

    def set_step(self, step: str) -> None:
        self.step = step

    def save_optimizer_output(self, stage_type: StageType, output: Any) -> None:
        if self.step is None:
            raise ValueError("Step must be set before saving optimizer output")

        # Save this to a file
        if self.step not in self.optimizer_output:
            self.optimizer_output[self.step] = {}

        self.optimizer_output[self.step][stage_type] = output


def has_jinja_syntax(template_string: str) -> bool:
    """
    Check if a string contains Jinja2 template syntax.

    Args:
        template_string (str): The string to check.

    Returns:
        bool: True if the string contains Jinja2 syntax ({{ }} or {% %}), False otherwise.
    """
    # Check for Jinja2 expression syntax {{ }}
    if re.search(r"\{\{.*?\}\}", template_string):
        return True
    # Check for Jinja2 statement syntax {% %}
    if re.search(r"\{%.*?%\}", template_string):
        return True
    return False


def prompt_user_for_non_jinja_confirmation(
    prompt_text: str, operation_name: str, prompt_field: str = "prompt"
) -> bool:
    """
    Prompt the user for confirmation when a prompt doesn't contain Jinja syntax.

    Args:
        prompt_text (str): The prompt text that doesn't contain Jinja syntax.
        operation_name (str): The name of the operation.
        prompt_field (str): The name of the prompt field (e.g., "prompt", "batch_prompt").

    Returns:
        bool: True if user confirms, False otherwise.
    """
    from docetl.console import DOCETL_CONSOLE

    console = DOCETL_CONSOLE
    console.print(
        f"\n[bold yellow]⚠ Warning:[/bold yellow] The '{prompt_field}' in operation '{operation_name}' "
        f"does not appear to be a Jinja2 template (no {{}} or {{% %}} syntax found)."
    )
    console.print(
        f"[dim]Prompt:[/dim] {prompt_text[:100]}{'...' if len(prompt_text) > 100 else ''}"
    )
    console.print(
        "\n[bold]We will automatically append the document(s) to your prompt during execution:[/bold]"
    )
    console.print(
        "  • For single-document operations: 'Here is the document: {{ input }}'"
    )
    console.print("  • For reduce operations: 'Here are the documents: {{ inputs }}'")
    console.print()

    try:
        return Confirm.ask(
            "Do you want to proceed with inserting all documents as-is?",
            default=True,
            console=console,
        )
    except Exception:
        # If Confirm fails (e.g., in non-interactive mode), default to True
        console.print(
            "[dim]Non-interactive mode: proceeding with document insertion[/dim]"
        )
        return True


def extract_jinja_variables(template_string: str) -> list[str]:
    """
    Extract variables from a Jinja2 template string.

    This function uses both Jinja2's AST parsing and regex to find all variables
    referenced in the given template string, including nested variables.

    Args:
        template_string (str): The Jinja2 template string to analyze.

    Returns:
        list[str]: A list of unique variable names found in the template.
    """
    # Create a Jinja2 environment
    env = Environment(autoescape=True)

    # Parse the template
    ast = env.parse(template_string)

    # Find all the variables referenced in the template
    variables = meta.find_undeclared_variables(ast)

    # Use regex to find any additional variables that might be missed
    # This regex looks for {{ variable }} patterns, including nested ones
    regex_variables = set(re.findall(r"{{\s*([\w.]+)\s*}}", template_string))

    # Combine both sets of variables
    all_variables = variables.union(regex_variables)

    # Special-case: remove "input"
    all_variables.discard("input")

    return list(all_variables)


def completion_cost(response: ModelResponse) -> float:
    try:
        return (
            response._completion_cost
            if hasattr(response, "_completion_cost")
            else lcc(response)
        )
    except Exception:
        return 0.0


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load and parse a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict[str, Any]: Parsed configuration as a dictionary.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    try:
        with open(config_path, "r") as config_file:
            config: dict[str, Any] = yaml.safe_load(config_file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML configuration: {e}")


def count_tokens(text: str, model: str) -> int:
    """
    Count the number of tokens in a string using the specified model.
    """
    model_name = model.replace("azure/", "")
    try:
        encoder = tiktoken.encoding_for_model(model_name)
        return len(encoder.encode(text))
    except Exception:
        # Use gpt-4o-mini to count tokens for other models
        encoder = tiktoken.encoding_for_model("gpt-4o")
        return len(encoder.encode(text))


def truncate_sample_data(
    data: dict[str, Any],
    available_tokens: int,
    key_lists: list[list[str]],
    model: str,
) -> dict[str, Any]:
    """
    Truncate sample data to fit within available tokens.

    Args:
        data (dict[str, Any]): The original data dictionary to truncate.
        available_tokens (int): The maximum number of tokens allowed.
        key_lists (list[list[str]]): Lists of keys to prioritize in the truncation process.
        model (str): The name of the model to use for token counting.

    Returns:
        dict[str, Any]: A new dictionary containing truncated data that fits within the token limit.
    """
    truncated_data = {}
    current_tokens = 0

    for key_list in key_lists:
        for key in sorted(
            key_list, key=lambda k: len(str(data.get(k, ""))), reverse=True
        ):
            if key in data:
                field_tokens = count_tokens(f'"{key}": {json.dumps(data[key])}', model)
                if current_tokens + field_tokens <= available_tokens:
                    truncated_data[key] = data[key]
                    current_tokens += field_tokens
                else:
                    # Calculate remaining tokens
                    remaining_tokens = available_tokens - current_tokens

                    # Encode the value
                    try:
                        encoder = tiktoken.encoding_for_model(model)
                    except Exception:
                        encoder = tiktoken.encoding_for_model("gpt-4o")
                    encoded_value = encoder.encode(str(data[key]))

                    # Calculate how many tokens to keep
                    tokens_to_keep = (
                        remaining_tokens - 20
                    )  # Reserve 20 tokens for truncation message
                    start_tokens = min(tokens_to_keep // 2, field_tokens // 2)
                    end_tokens = min(
                        tokens_to_keep - start_tokens, field_tokens - start_tokens
                    )

                    # Truncate the encoded value
                    truncated_encoded = (
                        encoded_value[:start_tokens]
                        + encoder.encode("[....truncated content...]")
                        + encoded_value[-end_tokens:]
                    )

                    # Decode the truncated value
                    truncated_value = encoder.decode(truncated_encoded)

                    # Add the truncated value to the result
                    truncated_data[key] = truncated_value
                    current_tokens += len(truncated_encoded)

                    return truncated_data

    return truncated_data


def smart_sample(
    input_data: list[dict], sample_size_needed: int, max_unique_values: int = 5
) -> list[dict]:
    """
    Smart sampling strategy that:
    1. Identifies categorical fields by checking for low cardinality (few unique values)
    2. Stratifies on up to 3 categorical fields
    3. Takes largest documents per stratum

    Args:
        input_data (list[dict]): List of input documents
        sample_size_needed (int): Number of samples needed
        max_unique_values (int): Maximum number of unique values for a field to be considered categorical

    Returns:
        list[dict]: Sampled documents
    """
    if not input_data or sample_size_needed >= len(input_data):
        return input_data

    # Find fields with low cardinality (categorical fields)
    field_unique_values = {}
    for field in input_data[0].keys():
        unique_values = set(str(doc.get(field, "")) for doc in input_data)
        if len(unique_values) <= max_unique_values:
            field_unique_values[field] = len(unique_values)

    # Sort by number of unique values and take top 3 categorical fields
    categorical_fields = sorted(field_unique_values.items(), key=lambda x: x[1])[:3]
    categorical_fields = [field for field, _ in categorical_fields]

    # If no categorical fields, return largest documents
    if not categorical_fields:
        return sorted(input_data, key=lambda x: len(json.dumps(x)), reverse=True)[
            :sample_size_needed
        ]

    # Group data by categorical fields
    groups: dict[tuple[str, ...], list[dict]] = {}
    for doc in input_data:
        key = tuple(str(doc.get(field, "")) for field in categorical_fields)
        if key not in groups:
            groups[key] = []
        groups[key].append(doc)

    # Calculate samples needed per group (evenly distributed)
    samples_per_group = math.ceil(sample_size_needed / len(groups))

    # Take largest documents from each group
    result = []
    for docs in groups.values():
        sorted_docs = sorted(docs, key=lambda x: len(json.dumps(x)), reverse=True)
        result.extend(sorted_docs[:samples_per_group])

    # If we have too many samples, trim to exact size needed
    # Sort by size again to ensure we keep the largest documents
    return sorted(result, key=lambda x: len(json.dumps(x)), reverse=True)[
        :sample_size_needed
    ]


class classproperty:
    def __init__(self, f: Callable[[Any], Any]) -> None:
        self.f = f

    def __get__(self, obj: Any | None, owner: type) -> Any:
        return self.f(owner)


def extract_output_from_json(yaml_file_path, json_output_path=None):
    """
    Extract output fields from JSON file based on the output schema defined in the YAML file.

    If the last operation doesn't have an output schema, returns all keys from the output data.

    Args:
        yaml_file_path (str): Path to the YAML configuration file
        json_output_path (str): Path to the JSON output file to extract from

    Returns:
        List[Dict]: Extracted data containing only the fields specified in the output schema,
                   or all fields if no output schema is defined
    """
    # Load YAML configuration
    with open(yaml_file_path, "r") as f:
        config = yaml.safe_load(f)

    if json_output_path is None:
        json_output_path = config.get("pipeline", {}).get("output", {}).get("path")
        if json_output_path is None:
            raise ValueError("No output path found in YAML file")

    # Load JSON output data
    with open(json_output_path, "r") as f:
        output_data = json.load(f)

    # Find the last operation in the pipeline
    pipeline = config.get("pipeline", {})
    steps = pipeline.get("steps", [])
    if not steps:
        raise ValueError("No pipeline steps found in YAML file")

    # Get the last step and its operations
    last_step = steps[-1]
    last_step_operations = last_step.get("operations", [])
    if not last_step_operations:
        raise ValueError("No operations found in the last pipeline step")

    # Get the name of the last operation in the last step
    last_operation_name = last_step_operations[-1]

    # Find this operation in the operations list
    operations = config.get("operations", [])
    last_operation = None
    for op in operations:
        if op.get("name") == last_operation_name:
            last_operation = op
            break

    if not last_operation:
        raise ValueError(
            f"Operation '{last_operation_name}' not found in operations list"
        )

    output_schema = last_operation.get("output", {}).get("schema", {})
    if not output_schema:
        # If no output schema, return all keys from the output data
        if isinstance(output_data, list) and len(output_data) > 0:
            # Get all unique keys from all items
            all_keys = set()
            for item in output_data:
                if isinstance(item, dict):
                    all_keys.update(item.keys())
            # Return all data with all keys
            return output_data
        else:
            # If output_data is not a list or is empty, return as-is
            return output_data if isinstance(output_data, list) else [output_data]

    # Extract the field names from the schema
    schema_fields = list(output_schema.keys())

    # Extract only the specified fields from each item in the output data
    extracted_data = []
    for item in output_data:
        extracted_item = {}
        for field in schema_fields:
            if field in item:
                extracted_item[field] = item[field]
        extracted_data.append(extracted_item)

    return extracted_data
