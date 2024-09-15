import functools
import os
import hashlib
import json
import shutil
import threading
from concurrent.futures import as_completed
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from openai import OpenAI

from dotenv import load_dotenv
from frozendict import frozendict
from jinja2 import Template
from litellm import completion, embedding, model_cost
from docetl.utils import completion_cost
from rich.console import Console
from tqdm import tqdm
from diskcache import Cache
import tiktoken
from rich import print as rprint
from pydantic import BaseModel, create_model

from docetl.utils import count_tokens

load_dotenv()
# litellm.set_verbose = True
DOCETL_HOME_DIR = os.path.expanduser("~/.docetl")

CACHE_DIR = os.path.join(DOCETL_HOME_DIR, "cache")
LLM_CACHE_DIR = os.path.join(DOCETL_HOME_DIR, "llm_cache")
cache = Cache(LLM_CACHE_DIR)

client = OpenAI()


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


@freezeargs
def gen_embedding(model: str, input: List[str]) -> List[float]:
    """
    A cached wrapper around litellm.embedding function.

    This function uses LRU (Least Recently Used) cache to store and retrieve
    embeddings for given inputs. It can significantly speed up repeated calls
    with the same model and input.

    Args:
        model (str): The name of the embedding model to use.
        input (str): The input text to generate an embedding for.

    Returns:
        List[float]: The embedding vector as a list of floats.

    Note:
        The cache size is set to 1000. Adjust this value based on your memory
        constraints and usage patterns.
    """
    # Create a unique key for the cache
    key = hashlib.md5(f"{model}_{input}".encode()).hexdigest()
    input = json.loads(input)

    # Try to get the result from cache
    result = cache.get(key)
    if result is None:
        # If not in cache, compute the embedding
        if not isinstance(input[0], str):
            input = [json.dumps(item) for item in input]

        input = [item if item else "None" for item in input]

        result = embedding(model=model, input=input)
        # Cache the result
        cache.set(key, result)

    return result


def flush_cache(console: Console = Console()):
    """
    Flush the cache to disk.
    """
    console.log("[bold green]Flushing cache to disk...[/bold green]")
    cache.close()
    console.log("[bold green]Cache flushed to disk.[/bold green]")


def clear_cache(console: Console = Console()):
    """
    Clear the LLM cache stored on disk.

    This function removes all cached items from the disk-based cache,
    effectively clearing the LLM's response history.

    Args:
        console (Console, optional): A Rich console object for logging.
            Defaults to a new Console instance.
    """
    console.log("[bold yellow]Clearing LLM cache...[/bold yellow]")
    try:
        cache.clear()
        cache.close()
        # Remove all files in the cache directory
        cache_dir = CACHE_DIR
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        for filename in os.listdir(cache_dir):
            file_path = os.path.join(cache_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                console.log(
                    f"[bold red]Error deleting {file_path}: {str(e)}[/bold red]"
                )
        console.log("[bold green]Cache cleared successfully.[/bold green]")
    except Exception as e:
        console.log(f"[bold red]Error clearing cache: {str(e)}[/bold red]")


def create_dynamic_model(schema: Dict[str, Any], model_name: str = "DynamicModel"):
    fields = {}

    def process_schema(s: Dict[str, Any], prefix: str = "") -> None:
        for key, value in s.items():
            field_name = f"{prefix}__{key}" if prefix else key
            if isinstance(value, dict):
                process_schema(value, field_name)
            else:
                fields[field_name] = parse_type(value, field_name)

    def parse_type(type_str: str, field_name: str) -> tuple:
        type_str = type_str.strip().lower()
        if type_str in ["str", "text", "string", "varchar"]:
            return (str, ...)
        elif type_str in ["int", "integer"]:
            return (int, ...)
        elif type_str in ["float", "decimal", "number"]:
            return (float, ...)
        elif type_str in ["bool", "boolean"]:
            return (bool, ...)
        elif type_str.startswith("list["):
            inner_type = type_str[5:-1].strip()
            item_type = parse_type(inner_type, f"{field_name}_item")[0]
            return (List[item_type], ...)
        elif type_str == "list":
            return (List[Any], ...)
        elif type_str.startswith("{") and type_str.endswith("}"):
            subfields = {}
            for item in type_str[1:-1].split(","):
                sub_key, sub_type = item.strip().split(":")
                subfields[sub_key.strip()] = parse_type(
                    sub_type.strip(), f"{field_name}_{sub_key}"
                )
            SubModel = create_model(f"{model_name}_{field_name}", **subfields)
            return (SubModel, ...)
        else:
            return (Any, ...)

    process_schema(schema)
    return create_model(model_name, **fields)


def convert_val(value: Any) -> Dict[str, Any]:
    """
    Convert a string representation of a type to a dictionary representation.

    This function takes a string value representing a data type and converts it
    into a dictionary format suitable for JSON schema.

    Args:
        value (Any): A string representing a data type.

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
        return {"type": "array", "items": convert_val(inner_type)}
    elif value == "list":
        raise ValueError("List type must specify its elements, e.g., 'list[str]'")
    elif value.startswith("{") and value.endswith("}"):
        # Handle dictionary type
        properties = {}
        for item in value[1:-1].split(","):
            key, val = item.strip().split(":")
            properties[key.strip()] = convert_val(val.strip())
        return {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
            "required": list(properties.keys()),
        }
    else:
        raise ValueError(f"Unsupported value type: {value}")


def cache_key(
    model: str,
    op_type: str,
    messages: List[Dict[str, str]],
    output_schema: Dict[str, str],
    scratchpad: Optional[str] = None,
) -> str:
    """
    Generate a unique cache key based on function arguments.

    This function creates a hash-based key using the input parameters, which can
    be used for caching purposes.

    Args:
        model (str): The model name.
        op_type (str): The operation type.
        messages (List[Dict[str, str]]): The messages to send to the LLM.
        output_schema (Dict[str, str]): The output schema dictionary.
        scratchpad (Optional[str]): The scratchpad to use for the operation.

    Returns:
        str: A unique hash string representing the cache key.
    """
    key_dict = {
        "model": model,
        "op_type": op_type,
        "messages": json.dumps(messages, sort_keys=True),
        "output_schema": json.dumps(output_schema, sort_keys=True),
        "scratchpad": scratchpad,
    }
    return hashlib.md5(json.dumps(key_dict, sort_keys=True).encode()).hexdigest()


# TODO: optimize this
@freezeargs
def cached_call_llm(
    cache_key: str,
    model: str,
    op_type: str,
    messages: List[Dict[str, str]],
    output_schema: Dict[str, str],
    tools: Optional[str] = None,
    scratchpad: Optional[str] = None,
) -> str:
    """
    Cached version of the call_llm function.

    This function serves as a cached wrapper around call_llm_with_cache. It uses
    the @freezeargs decorator to ensure immutable arguments and @functools.lru_cache
    for caching results.

    Args:
        cache_key (str): A unique key for caching.
        model (str): The model name.
        op_type (str): The operation type.
        messages (List[Dict[str, str]]): The messages to send to the LLM.
        output_schema (Dict[str, str]): The output schema dictionary.
        tools (Optional[str]): The tools to pass to the LLM.
        scratchpad (Optional[str]): The scratchpad to use for the operation.
    Returns:
        str: The result from call_llm_with_cache.
    """
    result = cache.get(cache_key)
    if result is None:
        result = call_llm_with_cache(
            model, op_type, messages, output_schema, tools, scratchpad
        )
        cache.set(cache_key, result)
    return result


def call_llm_with_validation(
    messages: List[str],
    llm_call_fn: Callable,
    validation_fn: Callable,
    val_rule: str,
    num_retries: int,
    console: Console,
) -> Tuple[Any, float, bool]:
    num_tries = num_retries + 1
    cost = 0.0
    for i in range(num_tries):
        response = llm_call_fn(messages)
        if isinstance(response, tuple):
            response, curr_cost = response
            cost += curr_cost

        cost += completion_cost(response)

        parsed_output, result = validation_fn(response)

        if result:
            return parsed_output, cost, True
        # Append the validation result to messages
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(parsed_output),
            }
        )
        messages.append(
            {
                "role": "user",
                "content": f"Your output {parsed_output} failed my validation rule: {str(val_rule)}\n\nPlease try again.",
            }
        )
        console.log(
            f"[bold red]Validation failed:[/bold red] {val_rule}\n"
            f"\t[yellow]Output:[/yellow] {parsed_output}\n"
            f"\tTrying again... ({i + 1}/{num_tries})"
        )

    return parsed_output, cost, False


def call_llm(
    model: str,
    op_type: str,
    messages: List[Dict[str, str]],
    output_schema: Dict[str, str],
    tools: Optional[List[Dict[str, str]]] = None,
    scratchpad: Optional[str] = None,
    console: Console = Console(),
) -> Any:
    """
    Wrapper function that uses caching for LLM calls.

    This function generates a cache key and calls the cached version of call_llm.
    It retries the call if it times out after 60 seconds.

    Args:
        model (str): The model name.
        op_type (str): The operation type.
        messages (List[Dict[str, str]]): The messages to send to the LLM.
        output_schema (Dict[str, str]): The output schema dictionary.
        tools (Optional[List[Dict[str, str]]]): The tools to pass to the LLM.
        scratchpad (Optional[str]): The scratchpad to use for the operation.
    Returns:
        str: The result from the cached LLM call.

    Raises:
        TimeoutError: If the call times out after retrying.
    """
    key = cache_key(model, op_type, messages, output_schema, scratchpad)

    max_retries = 2
    for attempt in range(max_retries):
        try:
            return timeout(120)(cached_call_llm)(
                key,
                model,
                op_type,
                messages,
                output_schema,
                json.dumps(tools) if tools else None,
                scratchpad,
            )
        except TimeoutError:
            if attempt == max_retries - 1:
                console.log(
                    f"[bold red]LLM call timed out after {max_retries} retries[/bold red]"
                )
                # TODO: HITL
                return {}


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


def call_llm_with_cache(
    model: str,
    op_type: str,
    messages: List[Dict[str, str]],
    output_schema: Dict[str, str],
    tools: Optional[str] = None,
    scratchpad: Optional[str] = None,
) -> str:
    """
    Make an LLM call with caching.

    This function prepares the necessary parameters and makes a call to the LLM
    using the provided model, operation type, prompt, and output schema.

    Args:
        model (str): The model name.
        op_type (str): The operation type.
        messages (List[Dict[str, str]]): The messages to send to the LLM.
        output_schema (Dict[str, str]): The output schema dictionary.
        tools (Optional[str]): The tools to pass to the LLM.
        scratchpad (Optional[str]): The scratchpad to use for the operation.
    Returns:
        str: The response from the LLM.
    """
    if tools is None:
        props = {key: convert_val(value) for key, value in output_schema.items()}
        if scratchpad is not None:
            props["updated_scratchpad"] = {"type": "string"}

        parameters = {"type": "object", "properties": props}
        parameters["required"] = list(props.keys())
        parameters["additionalProperties"] = False

        # response_format = {
        #     "type": "json_schema",
        #     "json_schema": {
        #         "name": "write_output",
        #         "description": "Write task output to a database",
        #         "strict": True,
        #         "schema": parameters,
        #         # "additionalProperties": False,
        #     },
        # }

        # tools = []
        # tool_choice = "auto"

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "write_output",
                    "description": "Write processing output to a database",
                    "strict": True,
                    "parameters": parameters,
                    "additionalProperties": False,
                },
            }
        ]
        tool_choice = {"type": "function", "function": {"name": "write_output"}}
        response_format = None

    else:
        tools = json.loads(tools)
        tool_choice = (
            "required" if any(tool.get("required", False) for tool in tools) else "auto"
        )
        tools = [{"type": "function", "function": tool["function"]} for tool in tools]
        response_format = None

    system_prompt = f"You are a helpful assistant, intelligently processing data. This is a {op_type} operation. You will perform the specified task on the provided data."
    if scratchpad:
        system_prompt += f"""

You are incrementally processing data across multiple batches. Maintain intermediate state between batches to accomplish this task effectively.

Current scratchpad: {scratchpad}

As you process each batch:
1. Update the scratchpad with crucial information for subsequent batches.
2. This may include partial results, counters, or data that doesn't fit into {list(output_schema.keys())}.
3. Example: For counting elements that appear more than twice, track all occurrences in the scratchpad until an item exceeds the threshold.

Keep the scratchpad concise (~500 chars) and easily parsable. Use clear structures like:
- Bullet points
- Key-value pairs
- JSON-like format

Update the 'updated_scratchpad' field in your output with the new scratchpad content.

Remember: The scratchpad should contain information necessary for processing future batches, not the final result."""
    messages = json.loads(messages)

    # Truncate messages if they exceed the model's context length
    messages = truncate_messages(messages, model)

    if response_format is None:
        response = completion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
            ]
            + messages,
            tools=tools,
            tool_choice=tool_choice,
        )
    else:
        response = completion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
            ]
            + messages,
            response_format=response_format,
        )

    return response


def truncate_messages(
    messages: List[Dict[str, str]], model: str, from_agent: bool = False
) -> List[Dict[str, str]]:
    """
    Truncate the messages to fit the model's context length.
    """
    if "gpt" not in model:
        model = "gpt-4o"

    model_input_context_length = model_cost.get(model, {}).get("max_input_tokens", 8192)
    total_tokens = sum(count_tokens(json.dumps(msg), model) for msg in messages)

    if total_tokens <= model_input_context_length - 100:
        return messages

    truncated_messages = messages.copy()
    longest_message = max(truncated_messages, key=lambda x: len(x["content"]))
    content = longest_message["content"]
    excess_tokens = total_tokens - model_input_context_length + 200  # 200 token buffer

    encoder = tiktoken.encoding_for_model(model)
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


def call_llm_with_gleaning(
    model: str,
    op_type: str,
    messages: List[Dict[str, str]],
    output_schema: Dict[str, str],
    validator_prompt_template: str,
    num_gleaning_rounds: int,
    console: Console = Console(),
) -> Tuple[str, float]:
    """
    Call LLM with a gleaning process, including validation and improvement rounds.

    This function performs an initial LLM call, followed by multiple rounds of
    validation and improvement based on the validator prompt template.

    Args:
        model (str): The model name.
        op_type (str): The operation type.
        messages (List[Dict[str, str]]): The messages to send to the LLM.
        output_schema (Dict[str, str]): The output schema dictionary.
        validator_prompt_template (str): Template for the validator prompt.
        num_gleaning_rounds (int): Number of gleaning rounds to perform.

    Returns:
        Tuple[str, float]: A tuple containing the final LLM response and the total cost.
    """
    props = {key: convert_val(value) for key, value in output_schema.items()}

    parameters = {"type": "object", "properties": props}
    parameters["required"] = list(props.keys())
    parameters["additionalProperties"] = False

    # Initial LLM call
    response = call_llm(model, op_type, messages, output_schema, console=console)

    cost = 0.0

    # Parse the response
    parsed_response = parse_llm_response(response)
    output = parsed_response[0]

    messages = (
        [
            {
                "role": "system",
                "content": f"You are a helpful assistant, intelligently processing data. This is a {op_type} operation.",
            }
        ]
        + messages
        + [
            {"role": "assistant", "content": json.dumps(output)},
        ]
    )

    for rnd in range(num_gleaning_rounds):
        cost += completion_cost(response)

        # Prepare validator prompt
        validator_template = Template(validator_prompt_template)
        validator_prompt = validator_template.render(output=output)

        # Call LLM for validation
        validator_response = completion(
            model="gpt-4o-mini",
            messages=truncate_messages(
                messages + [{"role": "user", "content": validator_prompt}], model
            ),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "should_refine": {"type": "boolean"},
                            "improvements": {"type": "string"},
                        },
                        "required": ["should_refine", "improvements"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        cost += completion_cost(validator_response)

        # Parse the validator response
        suggestion = json.loads(validator_response.choices[0].message.content)
        if suggestion["should_refine"] == False:
            break

        console.log(
            f"Validator improvements (gleaning round {rnd + 1}): {suggestion['improvements']}"
        )

        # Prompt for improvement
        improvement_prompt = f"""Based on the validation feedback:

```
{suggestion['improvements']}
```

Please improve your previous response. Ensure that the output adheres to the required schema and addresses any issues raised in the validation."""
        messages.append({"role": "user", "content": improvement_prompt})

        # Call LLM for improvement
        # TODO: support gleaning and tools
        response = completion(
            model=model,
            messages=truncate_messages(messages, model),
            # response_format={
            #     "type": "json_schema",
            #     "json_schema": {
            #         "name": "write_output",
            #         "description": "Write processing output to a database",
            #         "strict": True,
            #         "schema": parameters,
            #         # "additionalProperties": False,
            #     },
            # },
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "write_output",
                        "description": "Write processing output to a database",
                        "strict": True,
                        "parameters": parameters,
                        "additionalProperties": False,
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": "write_output"}},
        )

        # Update messages with the new response
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(parse_llm_response(response)[0]),
            }
        )

    return response, cost


def parse_llm_response(
    response: Any, tools: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, Any]]:
    """
    Parse the response from a language model.

    This function extracts the tool calls from the LLM response and returns the arguments
    of any 'write_output' function calls as a list of dictionaries.

    Args:
        response (Any): The response object from the language model.
        tools (Optional[List[Dict[str, str]]]): The tools that were passed to the LLM.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the parsed output.
    """
    if not response:
        return [{}]

    # Parse the response based on the provided tools
    if tools:
        # If custom tools are provided, parse accordingly
        tool_calls = response.choices[0].message.tool_calls
        results = []
        for tool_call in tool_calls:
            for tool in tools:
                if tool_call.function.name == tool["function"]["name"]:
                    try:
                        function_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        return [{}]
                    # Execute the function defined in the tool's code
                    local_scope = {}
                    exec(tool["code"].strip(), globals(), local_scope)
                    function_result = local_scope[tool["function"]["name"]](
                        **function_args
                    )
                    function_args.update(function_result)
                    results.append(function_args)
        return results
    else:
        if "tool_calls" in dir(response.choices[0].message):
            # Default behavior for write_output function
            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                raise ValueError("No tool calls found in response")

            outputs = []
            for tool_call in tool_calls:
                if tool_call.function.name == "write_output":
                    try:
                        outputs.append(json.loads(tool_call.function.arguments))
                    except json.JSONDecodeError:
                        return [{}]
            return outputs

        else:
            return [json.loads(response.choices[0].message.content)]

    # message = response.choices[0].message
    # return [json.loads(message.content)]


def validate_output(operation: Dict, output: Dict, console: Console) -> bool:
    """
    Validate the output against the specified validation rules in the operation.

    Args:
        operation (Dict): The operation dictionary containing validation rules.
        output (Dict): The output to be validated.
        console (Console): The console object for logging.

    Returns:
        bool: True if all validations pass, False otherwise.
    """
    if "validate" not in operation:
        return True
    for validation in operation["validate"]:
        try:
            if not eval(validation, {"output": output}):
                console.log(f"[bold red]Validation failed:[/bold red] {validation}")
                console.log(f"[yellow]Output:[/yellow] {output}")
                return False
        except Exception as e:
            console.log(f"[bold red]Validation error:[/bold red] {str(e)}")
            console.log(f"[yellow]Output:[/yellow] {output}")
            return False
    return True


class RichLoopBar:
    """
    A progress bar class that integrates with Rich console.

    This class provides a wrapper around tqdm to create progress bars that work
    with Rich console output.

    Args:
        iterable (Optional[Union[Iterable, range]]): An iterable to track progress.
        total (Optional[int]): The total number of iterations.
        desc (Optional[str]): Description to be displayed alongside the progress bar.
        leave (bool): Whether to leave the progress bar on screen after completion.
        console: The Rich console object to use for output.
    """

    def __init__(
        self,
        iterable: Optional[Union[Iterable, range]] = None,
        total: Optional[int] = None,
        desc: Optional[str] = None,
        leave: bool = True,
        console=None,
    ):
        if console is None:
            raise ValueError("Console must be provided")
        self.console = console
        self.iterable = iterable
        self.total = self._get_total(iterable, total)
        self.description = desc
        self.leave = leave
        self.tqdm = None

    def _get_total(self, iterable, total):
        """
        Determine the total number of iterations for the progress bar.

        Args:
            iterable: The iterable to be processed.
            total: The explicitly specified total, if any.

        Returns:
            int or None: The total number of iterations, or None if it can't be determined.
        """
        if total is not None:
            return total
        if isinstance(iterable, range):
            return len(iterable)
        try:
            return len(iterable)
        except TypeError:
            return None

    def __iter__(self):
        """
        Create and return an iterator with a progress bar.

        Returns:
            Iterator: An iterator that yields items from the wrapped iterable.
        """
        self.tqdm = tqdm(
            self.iterable,
            total=self.total,
            desc=self.description,
            file=self.console.file,
        )
        for item in self.tqdm:
            yield item

    def __enter__(self):
        """
        Enter the context manager, initializing the progress bar.

        Returns:
            RichLoopBar: The RichLoopBar instance.
        """
        self.tqdm = tqdm(
            total=self.total,
            desc=self.description,
            leave=self.leave,
            file=self.console.file,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager, closing the progress bar.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
            exc_val: The instance of the exception that caused the context to be exited.
            exc_tb: A traceback object encoding the stack trace.
        """
        self.tqdm.close()

    def update(self, n=1):
        """
        Update the progress bar.

        Args:
            n (int): The number of iterations to increment the progress bar by.
        """
        if self.tqdm:
            self.tqdm.update(n)


def rich_as_completed(futures, total=None, desc=None, leave=True, console=None):
    """
    Yield completed futures with a Rich progress bar.

    This function wraps concurrent.futures.as_completed with a Rich progress bar.

    Args:
        futures: An iterable of Future objects to monitor.
        total (Optional[int]): The total number of futures.
        desc (Optional[str]): Description for the progress bar.
        leave (bool): Whether to leave the progress bar on screen after completion.
        console: The Rich console object to use for output.

    Yields:
        Future: Completed future objects.

    Raises:
        ValueError: If no console object is provided.
    """
    if console is None:
        raise ValueError("Console must be provided")

    with RichLoopBar(total=total, desc=desc, leave=leave, console=console) as pbar:
        for future in as_completed(futures):
            yield future
            pbar.update()
