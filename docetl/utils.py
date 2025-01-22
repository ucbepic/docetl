import json
import math
import re
from enum import Enum
from typing import Any, Dict, List

import tiktoken
import yaml
from jinja2 import Environment, meta
from litellm import completion_cost as lcc
from lzstring import LZString


class Decryptor:
    def __init__(self, secret_key: str):
        self.key = secret_key
        self.lz = LZString()

    def decrypt(self, encrypted_data: str) -> str:
        try:
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

        except Exception as e:
            print(f"Decryption failed: {str(e)}")
            return None


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
    def __init__(self):
        self.optimizer_output = {}
        self.step = None

    def set_step(self, step: str):
        self.step = step

    def save_optimizer_output(self, stage_type: StageType, output: Any):
        if self.step is None:
            raise ValueError("Step must be set before saving optimizer output")

        # Save this to a file
        if self.step not in self.optimizer_output:
            self.optimizer_output[self.step] = {}

        self.optimizer_output[self.step][stage_type] = output


def extract_jinja_variables(template_string: str) -> List[str]:
    """
    Extract variables from a Jinja2 template string.

    This function uses both Jinja2's AST parsing and regex to find all variables
    referenced in the given template string, including nested variables.

    Args:
        template_string (str): The Jinja2 template string to analyze.

    Returns:
        List[str]: A list of unique variable names found in the template.
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


def completion_cost(response) -> float:
    try:
        return (
            response._completion_cost
            if hasattr(response, "_completion_cost")
            else lcc(response)
        )
    except Exception:
        return 0.0


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and parse a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Parsed configuration as a dictionary.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    try:
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)
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
    data: Dict[str, Any], available_tokens: int, key_lists: List[List[str]], model: str
) -> Dict[str, Any]:
    """
    Truncate sample data to fit within available tokens.

    Args:
        data (Dict[str, Any]): The original data dictionary to truncate.
        available_tokens (int): The maximum number of tokens allowed.
        key_lists (List[List[str]]): Lists of keys to prioritize in the truncation process.
        model (str): The name of the model to use for token counting.

    Returns:
        Dict[str, Any]: A new dictionary containing truncated data that fits within the token limit.
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
    input_data: List[Dict], sample_size_needed: int, max_unique_values: int = 5
) -> List[Dict]:
    """
    Smart sampling strategy that:
    1. Identifies categorical fields by checking for low cardinality (few unique values)
    2. Stratifies on up to 3 categorical fields
    3. Takes largest documents per stratum

    Args:
        input_data (List[Dict]): List of input documents
        sample_size_needed (int): Number of samples needed
        max_unique_values (int): Maximum number of unique values for a field to be considered categorical

    Returns:
        List[Dict]: Sampled documents
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
    groups = {}
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


class classproperty(object):
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)
