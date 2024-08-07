from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from jinja2 import Template
from motion.operations.base import BaseOperation
from motion.operations.utils import call_llm, parse_llm_response
from motion.operations.utils import validate_output, rich_as_completed
from litellm import completion_cost
from rich.console import Console


class MapOperation(BaseOperation):
    def syntax_check(self) -> None:
        required_keys = ["prompt", "output"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(
                    f"Missing required key '{key}' in MapOperation configuration"
                )

        if "schema" not in self.config["output"]:
            raise ValueError("Missing 'schema' in 'output' configuration")

        if not isinstance(self.config["output"]["schema"], dict):
            raise TypeError("'schema' in 'output' configuration must be a dictionary")

        if not self.config["output"]["schema"]:
            raise ValueError("'schema' in 'output' configuration cannot be empty")

        # Check if the prompt is a valid Jinja2 template
        try:
            Template(self.config["prompt"])
        except Exception as e:
            raise ValueError(f"Invalid Jinja2 template in 'prompt': {str(e)}")

        # Check if the model is specified (optional)
        if "model" in self.config and not isinstance(self.config["model"], str):
            raise TypeError("'model' in configuration must be a string")

    def execute(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        def _process_map_item(item: Dict) -> Tuple[Optional[Dict], float]:
            prompt_template = Template(self.config["prompt"])
            prompt = prompt_template.render(input=item)
            response = call_llm(
                self.config.get("model", self.default_model),
                "map",
                prompt,
                self.config["output"]["schema"],
            )
            item_cost = completion_cost(response)
            output = parse_llm_response(response)[0]
            for key, value in item.items():
                if key not in self.config["output"]["schema"]:
                    output[key] = value
            if validate_output(self.config, output, self.console):
                return output, item_cost
            return None, item_cost

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [executor.submit(_process_map_item, item) for item in input_data]
            results = []
            total_cost = 0
            for future in rich_as_completed(
                futures,
                total=len(futures),
                desc="Processing map items",
                leave=True,
                console=self.console,
            ):
                result, item_cost = future.result()
                if result is not None:
                    results.append(result)
                total_cost += item_cost

        return results, total_cost

    def validate_output(self, output: Dict) -> bool:
        schema = self.config["output"]["schema"]
        for key in schema:
            if key not in output:
                self.console.log(f"[red]Error: Missing key '{key}' in output[/red]")
                return False
        return True


class ParallelMapOperation(BaseOperation):
    def syntax_check(self) -> None:
        if "prompts" not in self.config or not isinstance(self.config["prompts"], list):
            raise ValueError(
                "ParallelMapOperation requires a 'prompts' list in the configuration"
            )

        if not self.config["prompts"]:
            raise ValueError("The 'prompts' list cannot be empty")

        for i, prompt_config in enumerate(self.config["prompts"]):
            if not isinstance(prompt_config, dict):
                raise TypeError(f"Prompt configuration {i} must be a dictionary")

            required_keys = ["name", "prompt", "output_keys"]
            for key in required_keys:
                if key not in prompt_config:
                    raise ValueError(
                        f"Missing required key '{key}' in prompt configuration {i}"
                    )

            if not isinstance(prompt_config["name"], str):
                raise TypeError(f"'name' in prompt configuration {i} must be a string")

            if not isinstance(prompt_config["prompt"], str):
                raise TypeError(
                    f"'prompt' in prompt configuration {i} must be a string"
                )

            if not isinstance(prompt_config["output_keys"], list):
                raise TypeError(
                    f"'output_keys' in prompt configuration {i} must be a list"
                )

            if not prompt_config["output_keys"]:
                raise ValueError(
                    f"'output_keys' list in prompt configuration {i} cannot be empty"
                )

            # Check if the prompt is a valid Jinja2 template
            try:
                Template(prompt_config["prompt"])
            except Exception as e:
                raise ValueError(
                    f"Invalid Jinja2 template in prompt configuration {i}: {str(e)}"
                )

            # Check if the model is specified (optional)
            if "model" in prompt_config and not isinstance(prompt_config["model"], str):
                raise TypeError(f"'model' in prompt configuration {i} must be a string")

        # Check if all output schema keys are covered by the prompts
        output_schema = self.config["output"]["schema"]
        output_keys_covered = set()
        for prompt_config in self.config["prompts"]:
            output_keys_covered.update(prompt_config["output_keys"])

        missing_keys = set(output_schema.keys()) - output_keys_covered
        if missing_keys:
            raise ValueError(
                f"The following output schema keys are not covered by any prompt: {missing_keys}"
            )

    def execute(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        results = []
        total_cost = 0
        output_schema = self.config["output"]["schema"]

        def process_prompt(item, prompt_config):
            prompt_template = Template(prompt_config["prompt"])
            prompt = prompt_template.render(input=item)
            local_output_schema = {
                key: output_schema[key] for key in prompt_config["output_keys"]
            }
            response = call_llm(
                prompt_config.get("model", self.default_model),
                "parallel_map",
                prompt,
                local_output_schema,
            )
            output = parse_llm_response(response)[0]
            return output, completion_cost(response)

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            # Create all futures at once
            all_futures = [
                executor.submit(process_prompt, item, prompt_config)
                for item in input_data
                for prompt_config in self.config["prompts"]
            ]

            # Process results in order
            for i, future in enumerate(
                rich_as_completed(
                    all_futures,
                    total=len(all_futures),
                    desc="Processing parallel map items",
                    console=self.console,
                )
            ):
                output, cost = future.result()
                total_cost += cost

                # Determine which item this future corresponds to
                item_index = i // len(self.config["prompts"])
                prompt_index = i % len(self.config["prompts"])

                # Initialize or update the item_result
                if prompt_index == 0:
                    item_result = input_data[item_index].copy()
                    results.append(item_result)

                # Update the item_result with the output
                item_result.update(output)

                # Validate the item_result if this is the last prompt for this item
                if prompt_index == len(self.config["prompts"]) - 1:
                    if not validate_output(self.config, item_result, self.console):
                        results.pop()  # Remove the invalid result

        return results, total_cost
