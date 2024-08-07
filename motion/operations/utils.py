import json
from typing import Dict, List, Any, Optional, Tuple
from litellm import completion, embedding
import litellm
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
# litellm.set_verbose = True


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


def call_llm(
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
                    "description": "Write output to a database",
                    "strict": True,
                    "parameters": parameters,
                    "additionalProperties": False,
                },
            }
        ],
        parallel_tool_calls=False,
        # num_retries=1,
        tool_choice={"type": "function", "function": {"name": "write_output"}},
    )

    return response


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
            console.print(f"[bold red]Validation failed:[/bold red] {validation}")
            console.print(f"[yellow]Output:[/yellow] {output}")
            return False
    return True
