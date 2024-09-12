import json
import random
from typing import Any, Dict, List

import jinja2
from rich.console import Console

from docetl.operations import get_operation
from docetl.optimizers.utils import LLMClient


def select_evaluation_samples(
    input_data: List[Dict[str, Any]], num_samples: int
) -> List[Dict[str, Any]]:
    if len(input_data) <= num_samples:
        return input_data
    return random.sample(input_data, num_samples)


def generate_and_validate_prompt(
    llm_client: LLMClient,
    base_prompt: str,
    system_prompt: str,
    parameters: Dict[str, Any],
    op_config: Dict[str, Any],
    is_metadata: bool,
    config: Dict[str, Any],
    max_threads: int,
    console: Console,
) -> Dict[str, Any]:
    max_retries = 3
    attempt = 0
    chat_history = [
        {"role": "user", "content": base_prompt},
    ]

    while attempt < max_retries:
        try:
            response = llm_client.generate(
                chat_history,
                system_prompt,
                parameters,
            )
            result = json.loads(response.choices[0].message.content)
            chat_history += [
                {"role": "assistant", "content": result},
            ]

            # Create a dummy operation to test the prompt
            dummy_op_config = {**op_config}  # Create a deep copy
            if is_metadata:
                dummy_op_config.update(
                    {
                        "type": "map",
                        "prompt": result["metadata_prompt"],
                        "output": {"schema": result["output_schema"]},
                    }
                )
            else:
                dummy_op_config.update(
                    {
                        "type": "reduce",
                        "prompt": result["combine_prompt"],
                        "reduce_key": result["reduce_key"],
                    }
                )

            operation_class = get_operation(dummy_op_config["type"])
            operation_class(
                dummy_op_config,
                config.get("default_model", "gpt-4o-mini"),
                max_threads,
                console,
            )

            # If we reach here, the prompt is valid
            return result

        except jinja2.exceptions.TemplateError as e:
            error_message = f"Invalid Jinja2 template: {str(e)}"
        except Exception as e:
            # We only care about jinja errors
            console.log(f"Error: {e}")
            return result

        # Print the error message to the console
        console.log(f"[bold red]Error:[/bold red] {error_message}")

        chat_history.append(
            {
                "role": "user",
                "content": f"The previous attempt failed. Error: {error_message}\n\nPlease try again, ensuring the prompt is a valid Jinja2 template and meets all requirements.",
            }
        )
        attempt += 1

    raise Exception(f"Failed to generate a valid prompt after {max_retries} attempts.")
