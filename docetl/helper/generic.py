import functools
import json
import os
import threading
from typing import Any, Dict, List, Optional

import tiktoken
from asteval import Interpreter
from diskcache import Cache
from dotenv import load_dotenv
from frozendict import frozendict
from litellm import model_cost
from rich import print as rprint
from rich.prompt import Prompt
from pydantic import BaseModel

from docetl.utils import count_tokens

aeval = Interpreter()

load_dotenv()
# litellm.set_verbose = True
DOCETL_HOME_DIR = os.path.expanduser("~/.docetl")

CACHE_DIR = os.path.join(DOCETL_HOME_DIR, "cache")
LLM_CACHE_DIR = os.path.join(DOCETL_HOME_DIR, "llm_cache")
cache = Cache(LLM_CACHE_DIR)
cache.close()


class LLMResult(BaseModel):
    response: Any
    total_cost: float
    validated: bool


def freezeargs(func):
    """
    Decorator to convert mutable dictionary arguments into immutable.

    This decorator is useful for making functions compatible with caching mechanisms
    that require immutable arguments.

    Args:
        func (callable): The function to be wrapped.

    Returns:
        callable: The wrapped function with immutable dictionary arguments.
    """

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        args = tuple(
            (
                frozendict(arg)
                if isinstance(arg, dict)
                else json.dumps(arg) if isinstance(arg, list) else arg
            )
            for arg in args
        )
        kwargs = {
            k: (
                frozendict(v)
                if isinstance(v, dict)
                else json.dumps(v) if isinstance(v, list) else v
            )
            for k, v in kwargs.items()
        }
        return func(*args, **kwargs)

    return wrapped


def convert_dict_schema_to_list_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    schema_str = "{" + ", ".join([f"{k}: {v}" for k, v in schema.items()]) + "}"
    return {"results": f"list[{schema_str}]"}

def convert_val(value: Any, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    Convert a string representation of a type to a dictionary representation.

    This function takes a string value representing a data type and converts it
    into a dictionary format suitable for JSON schema.

    Args:
        value (Any): A string representing a data type.
        model (str): The model being used. Defaults to "gpt-4o-mini".

    Returns:
        Dict[str, Any]: A dictionary representing the type in JSON schema format.

    Raises:
        ValueError: If the input value is not a supported type or is improperly formatted.
    """
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
        # Handle dictionary type
        properties = {}
        for item in value[1:-1].split(","):
            key, val = item.strip().split(":")
            properties[key.strip()] = convert_val(val.strip(), model)
        result = {
            "type": "object",
            "properties": properties,
            "required": list(properties.keys()),
        }
        # TODO: this is a hack to get around the fact that gemini doesn't support additionalProperties
        if "gemini" not in model:
            result["additionalProperties"] = False
        return result
    else:
        raise ValueError(f"Unsupported value type: {value}")


def get_user_input_for_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prompt the user for input for each key in the schema using Rich,
    then parse the input values with json.loads().

    Args:
        schema (Dict[str, Any]): The schema dictionary.

    Returns:
        Dict[str, Any]: A dictionary with user inputs parsed according to the schema.
    """
    user_input = {}

    for key, value_type in schema.items():
        prompt_text = f"Enter value for '{key}' ({value_type}): "
        user_value = Prompt.ask(prompt_text)

        try:
            # Parse the input value using json.loads()
            parsed_value = json.loads(user_value)

            # Check if the parsed value matches the expected type
            if isinstance(parsed_value, eval(value_type)):
                user_input[key] = parsed_value
            else:
                rprint(
                    f"[bold red]Error:[/bold red] Input for '{key}' does not match the expected type {value_type}."
                )
                return get_user_input_for_schema(schema)  # Recursive call to retry

        except json.JSONDecodeError:
            rprint(
                f"[bold red]Error:[/bold red] Invalid JSON input for '{key}'. Please try again."
            )
            return get_user_input_for_schema(schema)  # Recursive call to retry

    return user_input


class InvalidOutputError(Exception):
    """
    Custom exception raised when the LLM output is invalid or cannot be parsed.

    Attributes:
        message (str): Explanation of the error.
        output (str): The invalid output that caused the exception.
        expected_schema (Dict[str, Any]): The expected schema for the output.
        messages (List[Dict[str, str]]): The messages sent to the LLM.
        tools (Optional[List[Dict[str, str]]]): The tool calls generated by the LLM.
    """

    def __init__(
        self,
        message: str,
        output: str,
        expected_schema: Dict[str, Any],
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, str]]] = None,
    ):
        self.message = message
        self.output = output
        self.expected_schema = expected_schema
        self.messages = messages
        self.tools = tools
        super().__init__(self.message)

    def __str__(self):
        return (
            f"{self.message}\n"
            f"Invalid output: {self.output}\n"
            f"Expected schema: {self.expected_schema}\n"
            f"Messages sent to LLM: {self.messages}\n"
            f"Tool calls generated by LLM: {self.tools}"
        )


def timeout(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [TimeoutError("Function call timed out")]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(seconds)
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]

        return wrapper

    return decorator


def truncate_messages(
    messages: List[Dict[str, str]], model: str, from_agent: bool = False
) -> List[Dict[str, str]]:
    """
    Truncate the messages to fit the model's context length.
    """
    model_input_context_length = model_cost.get(model.split("/")[-1], {}).get(
        "max_input_tokens", 8192
    )
    total_tokens = sum(count_tokens(json.dumps(msg), model) for msg in messages)

    if total_tokens <= model_input_context_length - 100:
        return messages

    truncated_messages = messages.copy()
    longest_message = max(truncated_messages, key=lambda x: len(x["content"]))
    content = longest_message["content"]
    excess_tokens = total_tokens - model_input_context_length + 200  # 200 token buffer

    try:
        encoder = tiktoken.encoding_for_model(model.split("/")[-1])
    except Exception:
        encoder = tiktoken.encoding_for_model("gpt-4o")
    encoded_content = encoder.encode(content)
    tokens_to_remove = min(len(encoded_content), excess_tokens)
    mid_point = len(encoded_content) // 2
    truncated_encoded = (
        encoded_content[: mid_point - tokens_to_remove // 2]
        + encoder.encode(f" ... [{tokens_to_remove} tokens truncated] ... ")
        + encoded_content[mid_point + tokens_to_remove // 2 :]
    )
    truncated_content = encoder.decode(truncated_encoded)
    # Calculate the total number of tokens in the original content
    total_tokens = len(encoded_content)

    # Print the warning message using rprint
    warning_type = "User" if not from_agent else "Agent"
    rprint(
        f"[yellow]{warning_type} Warning:[/yellow] Cutting {tokens_to_remove} tokens from a prompt with {total_tokens} tokens..."
    )

    longest_message["content"] = truncated_content

    return truncated_messages


def safe_eval(expression: str, output: Dict) -> bool:
    """
    Safely evaluate an expression with a given output dictionary.
    Uses asteval to evaluate the expression.
    https://lmfit.github.io/asteval/index.html
    """
    try:
        # Add the output dictionary to the symbol table
        aeval.symtable["output"] = output
        # Safely evaluate the expression
        return bool(aeval(expression))
    except Exception:
        # try to evaluate with python eval
        try:
            return bool(eval(expression, locals={"output": output}))
        except Exception:
            return False


