import json
import math
import os
import re
import sys
from enum import Enum
from typing import Any

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


class CapturedOutput:
    def __init__(self) -> None:
        self.optimizer_output: dict[str, dict[StageType, Any]] = {}
        self.step: str | None = None

    def set_step(self, step: str) -> None:
        self.step = step

    def save_optimizer_output(self, stage_type: StageType, output: Any) -> None:
        if self.step is None:
            raise ValueError("Step must be set before saving optimizer output")

        if self.step not in self.optimizer_output:
            self.optimizer_output[self.step] = {}

        self.optimizer_output[self.step][stage_type] = output


def has_jinja_syntax(template_string: str) -> bool:
    if re.search(r"\{\{.*?\}\}", template_string):
        return True
    if re.search(r"\{%.*?%\}", template_string):
        return True
    return False


def prompt_user_for_non_jinja_confirmation(
    prompt_text: str, operation_name: str, prompt_field: str = "prompt"
) -> bool:
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

    if os.environ.get("USE_FRONTEND") == "true" or not sys.stdin.isatty():
        console.print(
            "[dim]Non-interactive mode: proceeding with document insertion[/dim]"
        )
        return True

    try:
        return Confirm.ask(
            "Do you want to proceed with inserting all documents as-is?",
            default=True,
            console=console,
        )
    except Exception:
        console.print(
            "[dim]Non-interactive mode: proceeding with document insertion[/dim]"
        )
        return True


def extract_jinja_variables(template_string: str) -> list[str]:
    env = Environment(autoescape=True)
    ast = env.parse(template_string)
    variables = meta.find_undeclared_variables(ast)
    # Regex catches nested dot-access patterns that Jinja2 AST misses
    regex_variables = set(re.findall(r"{{\s*([\w.]+)\s*}}", template_string))
    all_variables = variables.union(regex_variables)
    all_variables.discard("input")
    return list(all_variables)


def extract_input_field_reads(
    template_string: Any, var: str = "input"
) -> "frozenset[str] | None":
    """The set of *var* fields a Jinja template reads, or None if unknown.

    Sound for plan-rewrite decisions, unlike ``extract_jinja_variables``:
    every use of *var* must be a static attribute access (``input.x``) or
    constant subscript (``input["x"]``) for a result to be returned. Any
    other use — bare ``{{ input }}``, a filter over the whole object, a
    dynamic subscript, aliasing through ``{% set %}`` or a loop — and the
    template may read the entire row, so this returns None (fail closed).

    Non-Jinja strings also return None: at runtime the operation appends
    the whole document to such prompts (``_append_document_to_prompt``).
    """
    from jinja2 import nodes

    if not isinstance(template_string, str) or not has_jinja_syntax(template_string):
        return None
    env = Environment(autoescape=True)
    try:
        ast = env.parse(template_string)
    except Exception:
        return None

    fields: set[str] = set()
    # Every Name node for *var* must be consumed by a static field access.
    consumed: set[int] = set()
    for node in ast.find_all((nodes.Getattr, nodes.Getitem)):
        target = node.node
        if not (isinstance(target, nodes.Name) and target.name == var):
            continue
        if isinstance(node, nodes.Getattr):
            fields.add(node.attr)
        else:
            arg = node.arg
            if not (isinstance(arg, nodes.Const) and isinstance(arg.value, str)):
                return None  # dynamic subscript: unknown field
            fields.add(arg.value)
        consumed.add(id(target))
    for name in ast.find_all(nodes.Name):
        if name.name == var and id(name) not in consumed:
            return None  # whole-object use of *var*
    return frozenset(fields)


def extract_template_field_reads(
    template_string: Any, var: str = "input"
) -> "frozenset[str] | None":
    """Like ``extract_input_field_reads`` but treats non-Jinja strings as
    reading nothing (∅) instead of everything (None).

    Use for auxiliary templates (gleaning prompts, conditions) that are
    rendered as-is when they contain no Jinja syntax — unlike main prompts,
    no document is appended to them at runtime.
    """
    if not isinstance(template_string, str):
        return None
    if not has_jinja_syntax(template_string):
        return frozenset()
    return extract_input_field_reads(template_string, var=var)


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
    try:
        with open(config_path, "r") as config_file:
            config: dict[str, Any] = yaml.safe_load(config_file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML configuration: {e}")


def op_ref_name(entry: "str | dict[str, Any]") -> str:
    """The operation name of a pipeline step's ``operations`` entry.

    An entry is either a plain operation name, or — for equijoins — a
    single-key ``{name: {"left": ..., "right": ...}}`` dict. This is the
    one place that knows that encoding; use it instead of re-parsing.
    """
    return entry if isinstance(entry, str) else next(iter(entry))


def count_tokens(text: str, model: str) -> int:
    model_name = model.replace("azure/", "")
    try:
        encoder = tiktoken.encoding_for_model(model_name)
        return len(encoder.encode(text))
    except Exception:
        encoder = tiktoken.encoding_for_model("gpt-4o")
        return len(encoder.encode(text))


def truncate_sample_data(
    data: dict[str, Any],
    available_tokens: int,
    key_lists: list[list[str]],
    model: str,
) -> dict[str, Any]:
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
                    remaining_tokens = available_tokens - current_tokens
                    try:
                        encoder = tiktoken.encoding_for_model(model)
                    except Exception:
                        encoder = tiktoken.encoding_for_model("gpt-4o")
                    encoded_value = encoder.encode(str(data[key]))
                    tokens_to_keep = remaining_tokens - 20
                    start_tokens = min(tokens_to_keep // 2, field_tokens // 2)
                    end_tokens = min(
                        tokens_to_keep - start_tokens, field_tokens - start_tokens
                    )

                    truncated_encoded = (
                        encoded_value[:start_tokens]
                        + encoder.encode("[....truncated content...]")
                        + encoded_value[-end_tokens:]
                    )

                    truncated_value = encoder.decode(truncated_encoded)
                    truncated_data[key] = truncated_value
                    current_tokens += len(truncated_encoded)

                    return truncated_data

    return truncated_data


def smart_sample(
    input_data: list[dict], sample_size_needed: int, max_unique_values: int = 5
) -> list[dict]:
    if not input_data or sample_size_needed >= len(input_data):
        return input_data

    field_unique_values = {}
    for field in input_data[0].keys():
        unique_values = set(str(doc.get(field, "")) for doc in input_data)
        if len(unique_values) <= max_unique_values:
            field_unique_values[field] = len(unique_values)

    categorical_fields = sorted(field_unique_values.items(), key=lambda x: x[1])[:3]
    categorical_fields = [field for field, _ in categorical_fields]

    if not categorical_fields:
        return sorted(input_data, key=lambda x: len(json.dumps(x)), reverse=True)[
            :sample_size_needed
        ]

    groups: dict[tuple[str, ...], list[dict]] = {}
    for doc in input_data:
        key = tuple(str(doc.get(field, "")) for field in categorical_fields)
        if key not in groups:
            groups[key] = []
        groups[key].append(doc)

    samples_per_group = math.ceil(sample_size_needed / len(groups))
    result = []
    for docs in groups.values():
        sorted_docs = sorted(docs, key=lambda x: len(json.dumps(x)), reverse=True)
        result.extend(sorted_docs[:samples_per_group])

    return sorted(result, key=lambda x: len(json.dumps(x)), reverse=True)[
        :sample_size_needed
    ]


def extract_output_from_json(yaml_file_path, json_output_path=None):
    with open(yaml_file_path, "r") as f:
        config = yaml.safe_load(f)

    if json_output_path is None:
        json_output_path = config.get("pipeline", {}).get("output", {}).get("path")
        if json_output_path is None:
            raise ValueError("No output path found in YAML file")

    with open(json_output_path, "r") as f:
        output_data = json.load(f)

    pipeline = config.get("pipeline", {})
    steps = pipeline.get("steps", [])
    if not steps:
        raise ValueError("No pipeline steps found in YAML file")

    last_step = steps[-1]
    last_step_operations = last_step.get("operations", [])
    if not last_step_operations:
        raise ValueError("No operations found in the last pipeline step")

    last_operation_name = last_step_operations[-1]
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
        if isinstance(output_data, list) and len(output_data) > 0:
            return output_data
        return output_data if isinstance(output_data, list) else [output_data]

    schema_fields = list(output_schema.keys())
    return [
        {field: item[field] for field in schema_fields if field in item}
        for item in output_data
    ]
