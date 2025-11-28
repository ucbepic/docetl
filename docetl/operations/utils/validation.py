import json
from typing import Any

from asteval import Interpreter
from jinja2 import Environment, StrictUndefined, Template
from jinja2.exceptions import UndefinedError
from rich import print as rprint
from rich.prompt import Prompt

from docetl.utils import has_jinja_syntax

aeval = Interpreter()


def strict_render(template: Template | str, context: dict[str, Any]) -> str:
    """
    Renders a Jinja template with strict undefined checking.

    Args:
        template: Either a Jinja2 Template object or a template string
        context: Dictionary containing the template variables

    Returns:
        The rendered template string

    Raises:
        UndefinedError: When any undefined variable, attribute or index is accessed
        ValueError: When template is invalid
    """
    # Create strict environment
    env = Environment(undefined=StrictUndefined)

    # Only process string templates for non-Jinja syntax check
    if isinstance(template, str):
        template_string = template

        # Check if template doesn't have Jinja syntax and append document statement
        if not has_jinja_syntax(template_string):
            # Determine the operation type based on context variables
            if "left" in context and "right" in context:
                # Equijoin operation - append both documents
                template_string = (
                    f"{template_string}\n\nHere are the documents:\n"
                    f"Left document: {{{{ left }}}}\n"
                    f"Right document: {{{{ right }}}}"
                )
            elif "input1" in context and "input2" in context:
                # Comparison operation (resolve) - append both documents
                template_string = (
                    f"{template_string}\n\nHere are the documents:\n"
                    f"Document 1: {{{{ input1 }}}}\n"
                    f"Document 2: {{{{ input2 }}}}"
                )
            elif "inputs" in context:
                # Reduce operation - append "Here are the documents: {{ inputs }}"
                template_string = (
                    f"{template_string}\n\nHere are the documents: {{{{ inputs }}}}"
                )
            elif "input" in context:
                # Regular operation - append "Here is the document: {{ input }}"
                template_string = (
                    f"{template_string}\n\nHere is the document: {{{{ input }}}}"
                )

        # Convert string template to Template object
        try:
            template = env.from_string(template_string)
        except Exception as e:
            raise ValueError(f"Invalid template: {str(e)}")
    # If template is already a Template object, use it as-is

    try:
        return template.render(context)
    except UndefinedError as e:
        # Get the available context keys for better error reporting
        available_vars = list(context.keys())

        # For each var in context, if it's a dict, get the keys
        var_attributes = {}
        for var in available_vars:
            if isinstance(context[var], dict):
                var_attributes[var] = list(context[var].keys())
            elif isinstance(context[var], list) and len(context[var]) > 0:
                var_attributes[var] = [
                    f"inputs[i].{k}"
                    for k in context[var][0].keys()
                    if "_observability" not in k
                ]

        raise UndefinedError(
            f"{str(e)}\n"
            f"Your prompt can include the following variables: {available_vars}\n"
            f"For dictionary variables, you can access keys using dot notation (e.g. input.key).\n"
            f"Available keys for each document: {var_attributes}\n"
        )


def safe_eval(expression: str, output: dict[str, Any]) -> bool:
    """Safely evaluate an expression with a given output dictionary."""
    try:
        aeval.symtable["output"] = output
        return bool(aeval(expression))
    except Exception:
        try:
            return bool(eval(expression, locals={"output": output}))
        except Exception:
            return False


def convert_val(value: Any, model: str = "gpt-4o-mini") -> dict[str, Any]:
    """Convert a string representation of a type to a dictionary representation."""
    value = value.strip().lower()
    if value in ["str", "text", "string", "varchar"]:
        return {"type": "string"}
    elif value in ["int", "integer"]:
        return {"type": "integer"}
    elif value in ["float", "decimal", "number"]:
        return {"type": "number"}
    elif value in ["bool", "boolean"]:
        return {"type": "boolean"}
    elif value.startswith("list["):
        inner_type = value[5:-1].strip()
        return {"type": "array", "items": convert_val(inner_type, model)}
    elif value == "list":
        raise ValueError("List type must specify its elements, e.g., 'list[str]'")
    elif value.startswith("{") and value.endswith("}"):
        properties = {}
        for item in value[1:-1].split(","):
            key, val = item.strip().split(":")
            properties[key.strip()] = convert_val(val.strip(), model)
        result = {
            "type": "object",
            "properties": properties,
            "required": list(properties.keys()),
        }
        if "gemini" not in model:
            result["additionalProperties"] = False
        return result
    elif value.startswith("enum[") and value.endswith("]"):
        enum_values = value[5:-1].strip().split(",")
        enum_values = [v.strip() for v in enum_values]
        return {"type": "string", "enum": enum_values}
    else:
        raise ValueError(f"Unsupported value type: {value}")


def _is_integer(value: Any) -> bool:
    """Return True if value is an int but not a bool."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    """Return True if value is a real number (int or float) but not a bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_scalar(value: Any, schema: dict[str, Any]) -> bool:
    """Validate a scalar value against a simple JSON-schema-like dict produced by convert_val."""
    expected_type = schema.get("type")
    if expected_type == "string":
        if not isinstance(value, str):
            return False
        # Enum constraint for strings
        if "enum" in schema and value not in schema["enum"]:
            return False
        return True
    if expected_type == "integer":
        return _is_integer(value)
    if expected_type == "number":
        return _is_number(value)
    if expected_type == "boolean":
        return isinstance(value, bool)
    return False


def _validate_value_against_schema(value: Any, schema: dict[str, Any]) -> bool:
    """Recursively validate value against JSON-schema-like dicts from convert_val."""
    expected_type = schema.get("type")

    if expected_type in {"string", "integer", "number", "boolean"}:
        return _validate_scalar(value, schema)

    if expected_type == "array":
        if not isinstance(value, list):
            return False
        item_schema = schema.get("items", {})
        # If items schema missing, accept any items
        if not item_schema:
            return True
        for item in value:
            if not _validate_value_against_schema(item, item_schema):
                return False
        return True

    if expected_type == "object":
        if not isinstance(value, dict):
            return False
        properties: dict[str, Any] = schema.get("properties", {})
        required_keys: list[str] = schema.get("required", [])
        additional_props_allowed = schema.get("additionalProperties", True)

        # Check required keys
        for req in required_keys:
            if req not in value:
                return False
        # Validate known properties
        for key, prop_schema in properties.items():
            if key in value and not _validate_value_against_schema(
                value[key], prop_schema
            ):
                return False
        # additionalProperties constraint
        if not additional_props_allowed:
            for key in value.keys():
                if key not in properties:
                    return False
        return True

    # Unknown schema type -> fail closed
    return False


def validate_output_types(
    output: dict[str, Any], output_schema: dict[str, Any], model: str = "gpt-4o-mini"
) -> tuple[bool, list[str]]:
    """
    Validate that each value in output conforms to the type specified by output_schema.

    output_schema is the user-friendly dict like {"field": "string", "nums": "list[int]"}.
    This function converts each entry via convert_val and checks values recursively.

    Returns (is_valid, errors)
    """
    errors: list[str] = []
    # Build per-field schemas from string declarations
    field_schemas: dict[str, dict[str, Any]] = {
        key: convert_val(value_type, model) for key, value_type in output_schema.items()
    }

    for field, field_schema in field_schemas.items():
        if field not in output:
            errors.append(f"Missing required field: {field}")
            continue
        if not _validate_value_against_schema(output[field], field_schema):
            errors.append(
                f"Field '{field}' has invalid type/value. Expected {field_schema}, got {type(output[field]).__name__}: {output[field]}"
            )

    return (len(errors) == 0), errors


def convert_dict_schema_to_list_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a dictionary schema to a list schema."""
    schema_str = "{" + ", ".join([f"{k}: {v}" for k, v in schema.items()]) + "}"
    return {"results": f"list[{schema_str}]"}


def get_user_input_for_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Prompt the user for input for each key in the schema."""
    user_input = {}

    for key, value_type in schema.items():
        prompt_text = f"Enter value for '{key}' ({value_type}): "
        user_value = Prompt.ask(prompt_text)

        try:
            parsed_value = json.loads(user_value)
            if isinstance(parsed_value, eval(value_type)):
                user_input[key] = parsed_value
            else:
                rprint(
                    f"[bold red]Error:[/bold red] Input for '{key}' does not match the expected type {value_type}."
                )
                return get_user_input_for_schema(schema)

        except json.JSONDecodeError:
            rprint(
                f"[bold red]Error:[/bold red] Invalid JSON input for '{key}'. Please try again."
            )
            return get_user_input_for_schema(schema)

    return user_input
