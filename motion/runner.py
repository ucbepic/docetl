import yaml
import json
from typing import Dict, List, Any, Optional, Tuple
import tiktoken
from litellm import completion
from itertools import groupby
from operator import itemgetter
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from rich import print as rprint
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from litellm import completion_cost
from jinja2 import Template

load_dotenv()


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


class DSLRunner:
    def __init__(self, yaml_file: str, max_threads: int = None):
        with open(yaml_file, "r") as file:
            self.config = yaml.safe_load(file)
        self.default_model = self.config.get("default_model", "gpt-4o-mini")
        self.max_threads = max_threads or (os.cpu_count() or 1) * 4
        self.console = Console()
        self.datasets = {}

    def run(self):
        self.load_datasets()
        total_cost = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            for step in self.config["pipeline"]["steps"]:
                step_name = step["name"]
                step_task = progress.add_task(
                    f"Running step [cyan]{step_name}[/cyan]...", total=1
                )
                input_data = self.datasets[step["input"]] if "input" in step else None
                output_data, step_cost = self.execute_step(step, input_data, progress)
                self.datasets[step_name] = output_data
                total_cost += step_cost
                progress.update(
                    step_task,
                    advance=1,
                    description=f"Step [cyan]{step_name}[/cyan] completed. Cost: [green]${step_cost:.2f}[/green]",
                )

        self.save_output(self.datasets[self.config["pipeline"]["steps"][-1]["name"]])
        rprint(f"[bold green]Total cost: [green]${total_cost:.2f}[/green]")

    def load_datasets(self):
        for name, dataset_config in self.config["datasets"].items():
            if dataset_config["type"] == "file":
                with open(dataset_config["path"], "r") as file:
                    self.datasets[name] = json.load(file)
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_config['type']}")

    def save_output(self, data: List[Dict]):
        output_config = self.config["pipeline"]["output"]
        if output_config["type"] == "file":
            with open(output_config["path"], "w") as file:
                json.dump(data, file, indent=2)
        else:
            raise ValueError(f"Unsupported output type: {output_config['type']}")

    def execute_step(
        self, step: Dict, input_data: Optional[List[Dict]], progress: Progress
    ) -> Tuple[List[Dict], float]:
        total_cost = 0
        for operation in step["operations"]:
            if isinstance(operation, dict):
                operation_name = list(operation.keys())[0]
                operation_config = operation[operation_name]
            else:
                operation_name = operation
                operation_config = {}

            op_object = self.config["operations"][operation_name].copy()
            op_object.update(operation_config)

            op_task = progress.add_task(
                f"Running operation [cyan]{operation_name}[/cyan]...", total=1
            )
            input_data, cost = self.execute_operation(op_object, input_data)
            total_cost += cost
            progress.update(
                op_task,
                advance=1,
                description=f"Operation [cyan]{operation_name}[/cyan] completed. Cost: [green]${cost:.2f}[/green]",
            )

        return input_data, total_cost

    def execute_operation(
        self, operation: Dict, input_data: List[Dict]
    ) -> Tuple[List[Dict], float]:
        operation_type = operation["type"]

        if operation_type == "map":
            return self.execute_map(operation, input_data)
        elif operation_type == "filter":
            return self.execute_filter(operation, input_data)
        elif operation_type == "flatmap":
            return self.execute_flatmap(operation, input_data)
        elif operation_type == "parallel_flatmap":
            return self.execute_parallel_flatmap(operation, input_data)
        elif operation_type == "equijoin":
            left_data = self.datasets[operation["left"]]
            right_data = self.datasets[operation["right"]]
            return self.execute_equijoin(operation, left_data, right_data)
        elif operation_type == "splitter":
            return self.execute_splitter(operation, input_data)
        elif operation_type == "reduce":
            return self.execute_reduce(operation, input_data)
        else:
            raise ValueError(f"Unsupported operation type: {operation_type}")

    def execute_equijoin(
        self, operation: Dict, left_data: List[Dict], right_data: List[Dict]
    ) -> Tuple[List[Dict], float]:
        left_key = operation["left_key"]
        right_key = operation["right_key"]
        results = []

        # Create a dictionary for faster lookups
        right_dict = {item[right_key]: item for item in right_data}

        print([item[left_key] for item in left_data])
        print([item[right_key] for item in right_data])

        for left_item in left_data:
            if left_item[left_key] in right_dict:
                right_item = right_dict[left_item[left_key]]
                joined_item = {}
                for key, value in left_item.items():
                    joined_item[f"{key}_left" if key in right_item else key] = value
                for key, value in right_item.items():
                    joined_item[f"{key}_right" if key in left_item else key] = value
                if self.validate_output(operation, joined_item):
                    results.append(joined_item)

        return results, 0  # Assuming no cost for join operation

    def execute_map(
        self, operation: Dict, input_data: List[Dict]
    ) -> Tuple[List[Dict], float]:
        def _process_map_item(item: Dict) -> Tuple[Optional[Dict], float]:
            prompt_template = Template(operation["prompt"])
            prompt = prompt_template.render(**item)
            response = self.call_llm(
                operation.get("model", self.default_model),
                "map",
                prompt,
                operation["output"]["schema"],
                is_flatmap=False,
            )
            item_cost = completion_cost(response)
            output = self.parse_llm_response(response)[0]
            # Add key-value pairs from item that are not in output_schema
            for key, value in item.items():
                if key not in operation["output"]["schema"]:
                    output[key] = value
            if self.validate_output(operation, output):
                return output, item_cost
            return None, item_cost

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [executor.submit(_process_map_item, item) for item in input_data]
            results = []
            total_cost = 0
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing map items",
                leave=True,
            ):
                result, item_cost = future.result()
                if result is not None:
                    results.append(result)
                total_cost += item_cost

        return results, total_cost

    def execute_filter(
        self, operation: Dict, input_data: List[Dict]
    ) -> Tuple[List[Dict], float]:
        results = []
        total_cost = 0

        def _process_filter_item(item: Dict) -> Tuple[Optional[Dict], float]:
            prompt_template = Template(operation["prompt"])
            prompt = prompt_template.render(**item)
            response = self.call_llm(
                operation.get("model", self.default_model),
                "filter",
                prompt,
                operation["output"]["schema"],
            )
            item_cost = completion_cost(response)
            output = self.parse_llm_response(response)[0]
            # Add key-value pairs from item that are not in output_schema
            for key, value in item.items():
                if key not in operation["output"]["schema"]:
                    output[key] = value

            if self.validate_output(operation, output):
                return output, item_cost
            return None, item_cost

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [
                executor.submit(_process_filter_item, item) for item in input_data
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing filter items",
                leave=True,
            ):
                result, item_cost = future.result()
                total_cost += item_cost
                if result is not None:
                    results.append(result)

        return results, total_cost

    def execute_flatmap(
        self, operation: Dict, input_data: List[Dict]
    ) -> Tuple[List[Dict], float]:
        results = []
        total_cost = 0

        def process_item(item):
            prompt_template = Template(operation["prompt"])
            prompt = prompt_template.render(**item)
            response = self.call_llm(
                operation.get("model", self.default_model),
                "flatmap",
                prompt,
                operation["output"]["schema"],
                is_flatmap=True,
            )
            item_cost = completion_cost(response)
            outputs = self.parse_llm_response(response, is_flatmap=True)
            for output in outputs:
                # Add key-value pairs from item that are not in output_schema
                for key, value in item.items():
                    if key not in operation["output"]["schema"]:
                        output[key] = value
            return outputs, item_cost

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [executor.submit(process_item, item) for item in input_data]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing flatmap items",
                leave=True,
            ):
                outputs, item_cost = future.result()
                total_cost += item_cost
                if self.validate_output(operation, outputs):
                    results.extend(outputs)

        return results, total_cost

    def execute_parallel_flatmap(
        self, operation: Dict, input_data: List[Dict]
    ) -> Tuple[List[Dict], float]:
        results = []
        total_cost = 0

        def process_prompt(item, prompt_template, model, output_schema):
            prompt = Template(prompt_template).render(**item)
            response = self.call_llm(
                model,
                "map",
                prompt,
                output_schema,
            )
            output = self.parse_llm_response(response)[0]
            # Add key-value pairs from item that are not in output_schema
            for key, value in item.items():
                if key not in output_schema:
                    output[key] = value
            return output, completion_cost(response)

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = []
            for item in input_data:
                item_futures = []
                for i, prompt_template in enumerate(operation["prompts"]):
                    future = executor.submit(
                        process_prompt,
                        item,
                        prompt_template,
                        operation["models"][i],
                        operation["output"]["schema"],
                    )
                    item_futures.append(future)
                futures.append(item_futures)

            for item_futures in tqdm(
                futures, desc="Processing parallel flatmap items", leave=True
            ):
                item_results = []
                for future in as_completed(item_futures):
                    output, item_cost = future.result()
                    total_cost += item_cost
                    if self.validate_output(operation, output):
                        item_results.append(output)
                results.extend(item_results)

        return results, total_cost

    def execute_reduce(
        self, operation: Dict, input_data: List[Dict]
    ) -> Tuple[List[Dict], float]:
        reduce_key = operation["reduce_key"]
        sorted_data = sorted(input_data, key=itemgetter(reduce_key))
        grouped_data = groupby(sorted_data, key=itemgetter(reduce_key))

        results = []
        total_cost = 0

        def process_group(key, group):
            group_list = list(group)
            prompt_template = Template(operation["prompt"])
            prompt = prompt_template.render(reduce_key=key, values=group_list)
            response = self.call_llm(
                operation.get("model", self.default_model),
                "reduce",
                prompt,
                operation["output"]["schema"],
            )
            output = self.parse_llm_response(response)[0]
            output[reduce_key] = key
            return output, completion_cost(response)

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [
                executor.submit(process_group, key, group)
                for key, group in grouped_data
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing reduce items",
                leave=True,
            ):
                output, item_cost = future.result()
                total_cost += item_cost
                if self.validate_output(operation, output):
                    results.append(output)

        return results, total_cost

    def execute_splitter(
        self, operation: Dict, input_data: List[Dict]
    ) -> Tuple[List[Dict], float]:
        """
        EXAMPLE

        split_content:
            type: splitter
            chunk_size: 1000
            overlap_size: 100
            model: gpt-4o-mini
            split_key: content
        """
        results = []
        encoder = tiktoken.encoding_for_model(
            operation.get("model", self.default_model)
        )
        for item in input_data:
            content = item[operation["split_key"]]
            tokens = encoder.encode(content)
            chunks = []
            start = 0
            while start < len(tokens):
                end = start + operation["chunk_size"]
                chunk_tokens = tokens[start:end]
                chunk = encoder.decode(chunk_tokens)
                chunk_id = f"chunk_{start}_{end}"
                chunk_data = {"chunk_id": chunk_id, "chunk_content": chunk, **item}
                if self.validate_output(operation, chunk_data):
                    chunks.append(chunk_data)
                start = end - operation["overlap_size"]
            results.extend(chunks)
        return results, 0

    def call_llm(
        self,
        model: str,
        op_type: str,
        prompt: str,
        output_schema: Dict[str, str],
        is_flatmap: bool = False,
    ) -> str:
        props = {key: convert_val(value) for key, value in output_schema.items()}

        if is_flatmap:
            props = {
                "output": {
                    "type": "array",
                    "items": {"type": "object", "properties": props},
                }
            }

        parameters = {"type": "object", "properties": props}
        parameters["required"] = list(props.keys())

        system_prompt = f"You are a helpful assistant to intelligently process data, writing outputs to a database. This is a {op_type} operation."
        if is_flatmap:
            system_prompt += " You may write outputs multiple times for the same input."

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
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "write_output",
                        "description": "Write output to a database",
                        "parameters": parameters,
                    },
                }
            ],
            parallel_tool_calls=False,
            tool_choice={"type": "function", "function": {"name": "write_output"}},
        )
        return response

    def parse_llm_response(
        self, response: Any, is_flatmap: bool = False
    ) -> List[Dict[str, Any]]:
        # This is a simplified parser
        tool_calls = response.choices[0].message.tool_calls
        tools = []
        for tool_call in tool_calls:
            if tool_call.function.name == "write_output":
                if not is_flatmap:
                    tools.append(json.loads(tool_call.function.arguments))
                else:
                    args = json.loads(tool_call.function.arguments)
                    for arg in args["output"]:
                        tools.append(arg)
        return tools

    def validate_output(self, operation: Dict, output: Dict) -> bool:
        if "validate" not in operation:
            return True
        for validation in operation["validate"]:
            if not eval(validation, {"output": output}):
                rprint(f"[bold red]Validation failed:[/bold red] {validation}")
                rprint(f"[yellow]Output:[/yellow] {output}")
                return False
        return True

    def load_input_for_join(self, input_config: Dict) -> List[Dict]:
        if input_config["type"] == "file":
            with open(input_config["path"], "r") as file:
                return json.load(file)
        else:
            raise ValueError(f"Unsupported input type for join: {input_config['type']}")
