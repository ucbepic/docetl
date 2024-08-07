import json
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
    """Convert a mutable dictionary into immutable.
    Useful to be compatible with cache
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
    value = value.lower()
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
    else:
        raise ValueError(f"Unsupported value type: {value}")


def cache_key(
    model: str, op_type: str, prompt: str, output_schema: Dict[str, str]
) -> str:
    """Generate a unique cache key based on function arguments."""
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
    """Cached version of call_llm function."""
    return call_llm_with_cache(model, op_type, prompt, output_schema)


def call_llm(
    model: str,
    op_type: str,
    prompt: str,
    output_schema: Dict[str, str],
) -> str:
    """Wrapper function that uses caching for LLM calls."""
    key = cache_key(model, op_type, prompt, output_schema)
    return cached_call_llm(key, model, op_type, prompt, output_schema)


def call_llm_with_cache(
    model: str,
    op_type: str,
    prompt: str,
    output_schema: Dict[str, str],
) -> str:
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
        num_retries=1,
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
    Calls LLM with gleaning process, including validation and improvement rounds.
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
    if "validate" not in operation:
        return True
    for validation in operation["validate"]:
        if not eval(validation, {"output": output}):
            console.log(f"[bold red]Validation failed:[/bold red] {validation}")
            console.log(f"[yellow]Output:[/yellow] {output}")
            return False
    return True


class RichLoopBar:
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
        if total is not None:
            return total
        if isinstance(iterable, range):
            return len(iterable)
        try:
            return len(iterable)
        except TypeError:
            return None

    def __iter__(self):
        self.tqdm = tqdm(
            self.iterable,
            total=self.total,
            desc=self.description,
            file=self.console.file,
        )
        for item in self.tqdm:
            yield item

    def __enter__(self):
        self.tqdm = tqdm(
            total=self.total,
            desc=self.description,
            leave=self.leave,
            file=self.console.file,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tqdm.close()

    def update(self, n=1):
        if self.tqdm:
            self.tqdm.update(n)


def rich_as_completed(futures, total=None, desc=None, leave=True, console=None):
    if console is None:
        raise ValueError("Console must be provided")

    with RichLoopBar(total=total, desc=desc, leave=leave, console=console) as pbar:
        for future in as_completed(futures):
            yield future
            pbar.update()
