import json
import threading
from typing import Dict, List, Any, Optional, Tuple, Iterable, Union
from litellm import completion, embedding, completion_cost
import litellm
from dotenv import load_dotenv
from rich.console import Console
import hashlib
import functools
from rich.progress import Progress, TaskID
from concurrent.futures import as_completed
from tqdm import tqdm
from jinja2 import Template

load_dotenv()
# litellm.set_verbose = True

from frozendict import frozendict


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
        args = (frozendict(arg) if isinstance(arg, dict) else arg for arg in args)
        kwargs = {
            k: frozendict(v) if isinstance(v, dict) else v for k, v in kwargs.items()
        }
        return func(*args, **kwargs)

    return wrapped


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
    model: str, op_type: str, prompt: str, output_schema: Dict[str, str]
) -> str:
    """
    Generate a unique cache key based on function arguments.

    This function creates a hash-based key using the input parameters, which can
    be used for caching purposes.

    Args:
        model (str): The model name.
        op_type (str): The operation type.
        prompt (str): The prompt text.
        output_schema (Dict[str, str]): The output schema dictionary.

    Returns:
        str: A unique hash string representing the cache key.
    """
    key_dict = {
        "model": model,
        "op_type": op_type,
        "prompt": prompt,
        "output_schema": json.dumps(output_schema, sort_keys=True),
    }
    return hashlib.md5(json.dumps(key_dict, sort_keys=True).encode()).hexdigest()


# TODO: optimize this
@freezeargs
@functools.lru_cache(maxsize=100000)
def cached_call_llm(
    cache_key: str, model: str, op_type: str, prompt: str, output_schema: Dict[str, str]
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
        prompt (str): The prompt text.
        output_schema (Dict[str, str]): The output schema dictionary.

    Returns:
        str: The result from call_llm_with_cache.
    """
    return call_llm_with_cache(model, op_type, prompt, output_schema)


def call_llm(
    model: str,
    op_type: str,
    prompt: str,
    output_schema: Dict[str, str],
) -> str:
    """
    Wrapper function that uses caching for LLM calls.

    This function generates a cache key and calls the cached version of call_llm.
    It retries the call if it times out after 60 seconds.

    Args:
        model (str): The model name.
        op_type (str): The operation type.
        prompt (str): The prompt text.
        output_schema (Dict[str, str]): The output schema dictionary.

    Returns:
        str: The result from the cached LLM call.

    Raises:
        TimeoutError: If the call times out after retrying.
    """
    key = cache_key(model, op_type, prompt, output_schema)

    max_retries = 2
    for attempt in range(max_retries):
        try:
            return timeout(60)(cached_call_llm)(
                key, model, op_type, prompt, output_schema
            )
        except TimeoutError:
            if attempt == max_retries - 1:
                raise TimeoutError("LLM call timed out after multiple retries")


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
    prompt: str,
    output_schema: Dict[str, str],
) -> str:
    """
    Make an LLM call with caching.

    This function prepares the necessary parameters and makes a call to the LLM
    using the provided model, operation type, prompt, and output schema.

    Args:
        model (str): The model name.
        op_type (str): The operation type.
        prompt (str): The prompt text.
        output_schema (Dict[str, str]): The output schema dictionary.

    Returns:
        str: The response from the LLM.
    """
    props = {key: convert_val(value) for key, value in output_schema.items()}

    parameters = {"type": "object", "properties": props}
    parameters["required"] = list(props.keys())
    parameters["additionalProperties"] = False

    system_prompt = f"You are a helpful assistant to intelligently process data. This is a {op_type} operation."

    response = completion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        # response_format={
        #     "type": "json_schema",
        #     "json_schema": {
        #         "name": "output",
        #         "strict": True,
        #         "schema": parameters,
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
        # parallel_tool_calls=False,
        # num_retries=1,
        tool_choice={"type": "function", "function": {"name": "write_output"}},
    )

    return response


def call_llm_with_gleaning(
    model: str,
    op_type: str,
    prompt: str,
    output_schema: Dict[str, str],
    validator_prompt_template: str,
    num_gleaning_rounds: int,
) -> Tuple[str, float]:
    """
    Call LLM with a gleaning process, including validation and improvement rounds.

    This function performs an initial LLM call, followed by multiple rounds of
    validation and improvement based on the validator prompt template.

    Args:
        model (str): The model name.
        op_type (str): The operation type.
        prompt (str): The initial prompt text.
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
    response = call_llm(model, op_type, prompt, output_schema)

    cost = 0.0

    # Parse the response
    parsed_response = parse_llm_response(response)
    output = parsed_response[0]

    messages = [
        {
            "role": "system",
            "content": f"You are a helpful assistant to intelligently process data. This is a {op_type} operation.",
        },
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": json.dumps(output)},
    ]

    for _ in range(num_gleaning_rounds):
        cost += completion_cost(response)

        # Prepare validator prompt
        validator_template = Template(validator_prompt_template)
        validator_prompt = validator_template.render(output=output)

        # Call LLM for validation
        validator_response = completion(
            model=model,
            messages=messages + [{"role": "user", "content": validator_prompt}],
        )
        cost += completion_cost(validator_response)

        # Prompt for improvement
        improvement_prompt = f"""Based on the validation feedback:

```
{validator_response.choices[0].message.content}
```

Please improve your previous response. Ensure that the output adheres to the required schema and addresses any issues raised in the validation."""
        messages.append({"role": "user", "content": improvement_prompt})

        # Call LLM for improvement
        response = completion(
            model=model,
            messages=messages,
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


def parse_llm_response(response: Any) -> List[Dict[str, Any]]:
    """
    Parse the response from a language model.

    This function extracts the tool calls from the LLM response and returns the arguments
    of any 'write_output' function calls as a list of dictionaries.

    Args:
        response (Any): The response object from the language model.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the parsed output.
    """
    # This is a simplified parser
    tool_calls = response.choices[0].message.tool_calls
    tools = []
    for tool_call in tool_calls:
        if tool_call.function.name == "write_output":
            tools.append(json.loads(tool_call.function.arguments))
    return tools

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
        if not eval(validation, {"output": output}):
            console.log(f"[bold red]Validation failed:[/bold red] {validation}")
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
