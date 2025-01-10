import json
from typing import Any, Dict, Union

from asteval import Interpreter
from jinja2 import Environment, StrictUndefined, Template
from jinja2.exceptions import UndefinedError
from rich import print as rprint
from rich.prompt import Prompt

aeval = Interpreter()


def strict_render(template: Union[Template, str], context: Dict[str, Any]) -> str:
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

    # Convert string to Template if needed
    if isinstance(template, str):

        # # If "inputs" in the context, make sure they are not accessing some attribute of inputs
        # if "inputs" in context and "{{ inputs." in template:
        #     raise UndefinedError("The inputs variable is a list, so you cannot access attributes of inputs. Use inputs[index].key instead.")

        try:
            template = env.from_string(template)
        except Exception as e:
            raise ValueError(f"Invalid template: {str(e)}")

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


def safe_eval(expression: str, output: Dict) -> bool:
    """Safely evaluate an expression with a given output dictionary."""
    try:
        aeval.symtable["output"] = output
        return bool(aeval(expression))
    except Exception:
        try:
            return bool(eval(expression, locals={"output": output}))
        except Exception:
            return False


def convert_val(value: Any, model: str = "gpt-4o-mini") -> Dict[str, Any]:
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


def convert_dict_schema_to_list_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a dictionary schema to a list schema."""
    schema_str = "{" + ", ".join([f"{k}: {v}" for k, v in schema.items()]) + "}"
    return {"results": f"list[{schema_str}]"}


def get_user_input_for_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
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
