from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import copy
import yaml
from typing import Dict, List, Any, Optional, Tuple, Union
from motion.operations import get_operation
from motion.operations.base import BaseOperation
from motion.optimizers.map_optimizer import MapOptimizer
from motion.optimizers.reduce_optimizer import ReduceOptimizer
from motion.optimizers.join_optimizer import JoinOptimizer
from motion.utils import load_config
from rich.console import Console
from rich.table import Table
import random
import json
from litellm import completion, completion_cost
import os
import jinja2
from jinja2 import Environment, meta
import re
from tqdm import tqdm
from motion.optimizers.utils import extract_jinja_variables, LLMClient


SUPPORTED_OPS = ["map", "resolve", "reduce", "equijoin"]


SAMPLE_SIZE_MAP = {
    "reduce": 40,
    "map": 10,
    "resolve": 100,
    "equijoin": 100,
}


class Optimizer:
    def __init__(
        self,
        yaml_file: str,
        max_threads: Optional[int] = None,
        model: str = "gpt-4o",
        timeout: int = 60,
    ):
        self.yaml_file_path = yaml_file
        self.config = load_config(yaml_file)
        self.console = Console()
        self.optimized_config = self.config.copy()
        self.llm_client = LLMClient(model)
        self.max_threads = max_threads or (os.cpu_count() or 1) * 4
        self.operations_cost = 0
        self.timeout = timeout
        self.sample_sizes = defaultdict(dict)
        self.datasets = {}

        self.print_optimizer_config()

    def print_optimizer_config(self):
        self.console.print("[bold cyan]Optimizer Configuration:[/bold cyan]")
        self.console.print("─" * 40)
        self.console.print(f"[yellow]YAML File:[/yellow] {self.yaml_file_path}")
        self.console.print(f"[yellow]Sample Size:[/yellow] {SAMPLE_SIZE_MAP}")
        self.console.print(f"[yellow]Max Threads:[/yellow] {self.max_threads}")
        self.console.print(f"[yellow]Model:[/yellow] {self.llm_client.model}")
        self.console.print(f"[yellow]Timeout:[/yellow] {self.timeout} seconds")
        self.console.print("─" * 40)

    def analyze_pipeline(self):
        steps = self.config["pipeline"]["steps"]
        dependencies = defaultdict(list)

        # Analyze dependencies
        for i, step in enumerate(steps):
            for j in range(i + 1, len(steps)):
                if steps[j].get("input") == step.get("name"):
                    dependencies[i].append(j)

        # Determine sample sizes for each operation in each step
        for i in range(len(steps) - 1, -1, -1):
            step = steps[i]
            step_name = step.get("name")
            downstream_samples = max(
                [max(self.sample_sizes[j].values(), default=0) for j in dependencies[i]]
                + [0]
            )

            for operation in step["operations"]:
                if isinstance(operation, dict):
                    operation_name = list(operation.keys())[0]
                    operation_config = self.config["operations"][operation_name]
                else:
                    operation_name = operation
                    operation_config = self.config["operations"][operation_name]

                op_type = operation_config.get("type")
                self.sample_sizes[step_name][operation_name] = max(
                    SAMPLE_SIZE_MAP.get(op_type, float("inf")), downstream_samples
                )

        self.print_sample_sizes()

    def print_sample_sizes(self):
        self.console.print("[bold cyan]Sample Sizes:[/bold cyan]")
        self.console.print("─" * 40)
        for step_name, operations in self.sample_sizes.items():
            self.console.print(f"[yellow]Step:[/yellow] {step_name}")
            for operation_name, sample_size in operations.items():
                self.console.print(f"  [green]{operation_name}:[/green] {sample_size}")
            self.console.print("─" * 40)

    def optimize(self):
        self.analyze_pipeline()

        optimized_steps = []
        optimized_operations = {}
        for step in self.config["pipeline"]["steps"]:
            step_name = step.get("name")
            if not step_name:
                raise ValueError(
                    f"Step does not have a name. Each step must have a unique name."
                )

            optimized_step, step_operations, input_data = self._optimize_step(step)
            optimized_steps.append(optimized_step)
            optimized_operations.update(step_operations)

            # Save the result to datasets using the step name
            self.datasets[step_name] = copy.deepcopy(input_data)

        self.optimized_config["operations"] = optimized_operations
        self.optimized_config["pipeline"]["steps"] = optimized_steps
        self._save_optimized_config()

        self.console.print(
            f"[bold]Total agent cost: ${self.llm_client.total_cost:.2f}[/bold]"
        )
        self.console.print(
            f"[bold]Total operations cost: ${self.operations_cost:.2f}[/bold]"
        )
        self.console.print(
            f"[bold]Total cost: ${self.llm_client.total_cost + self.operations_cost:.2f}[/bold]"
        )

    def _optimize_step(
        self, step: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
        optimized_operations = {}
        input_data = None

        for operation in step["operations"]:
            if isinstance(operation, dict):
                operation_name = list(operation.keys())[0]
                operation_config = operation[operation_name]
            else:
                operation_name = operation
                operation_config = {}

            op_object = self.config["operations"][operation_name].copy()
            op_object.update(operation_config)
            op_object["name"] = operation_name
            sample_size = self.sample_sizes[step.get("name")][operation_name]

            # For equijoin operations, load both left and right datasets
            if op_object.get("type") == "equijoin":
                left_data = self._get_sample_data(
                    op_object.get("left"), op_object, sample_size
                )
                right_data = self._get_sample_data(
                    op_object.get("right"), op_object, sample_size
                )
                input_data = {"left": left_data, "right": right_data}
            else:
                if input_data is None:
                    input_data = self._get_sample_data(
                        step.get("input"), op_object, sample_size
                    )

            if (
                op_object.get("optimize", True) == False
                or op_object.get("type") not in SUPPORTED_OPS
            ):
                # If optimize is False or operation type is not supported, just run the operation without optimization
                # Use rich console status to indicate running the operation
                with self.console.status(
                    f"[bold green]Running operation: {operation_name} (Type: {op_object['type']})[/bold green]"
                ):
                    # Print the number of elements in input_data
                    self.console.print(f"[yellow]Running Operation:[/yellow]")
                    self.console.print(f"[yellow]  Type: {op_object['type']}[/yellow]")
                    self.console.print(f"[yellow]  Name: {operation_name}[/yellow]")
                    if op_object.get("type") == "equijoin":
                        self.console.print(
                            f"[yellow]  Sample size (left): {len(input_data['left'])}[/yellow]"
                        )
                        self.console.print(
                            f"[yellow]  Sample size (right): {len(input_data['right'])}[/yellow]"
                        )
                    else:
                        self.console.print(
                            f"[yellow]  Sample size: {len(input_data)}[/yellow]"
                        )
                    input_data = self._run_operation(op_object, input_data)
                    optimized_operations[operation_name] = op_object
            else:
                # Use rich console status to indicate optimization of the operation
                with self.console.status(
                    f"[bold blue]Optimizing operation: {operation_name} (Type: {op_object['type']})[/bold blue]"
                ):
                    # Print the number of elements in input_data
                    self.console.print(f"[yellow]Optimizing Operation:[/yellow]")
                    self.console.print(f"[yellow]  Type: {op_object['type']}[/yellow]")
                    self.console.print(f"[yellow]  Name: {operation_name}[/yellow]")
                    if op_object.get("type") == "equijoin":
                        self.console.print(
                            f"[yellow]  Sample size (left): {len(input_data['left'])}[/yellow]"
                        )
                        self.console.print(
                            f"[yellow]  Sample size (right): {len(input_data['right'])}[/yellow]"
                        )
                    else:
                        self.console.print(
                            f"[yellow]  Sample size: {len(input_data)}[/yellow]"
                        )

                    # Run optimization
                    if op_object.get("type") == "map":
                        optimized_ops, input_data = self._optimize_map(
                            op_object, input_data
                        )
                    elif op_object.get("type") == "reduce":
                        optimized_ops, input_data = self._optimize_reduce(
                            op_object, input_data
                        )
                    elif op_object.get("type") == "resolve":
                        optimized_ops, input_data = self._optimize_resolve(
                            op_object, input_data
                        )
                    elif op_object.get("type") == "equijoin":
                        optimized_ops, input_data = self._optimize_equijoin(
                            op_object, input_data["left"], input_data["right"]
                        )
                    else:
                        raise ValueError(
                            f"Unsupported operation type: {op_object['type']}"
                        )

                    for op in optimized_ops:
                        op_name = op.pop("name")
                        optimized_operations[op_name] = op

        optimized_step = step.copy()
        optimized_step["operations"] = list(optimized_operations.keys())
        return optimized_step, optimized_operations, input_data

    def _get_sample_data(
        self, dataset_name: str, op_config: Optional[Dict[str, Any]], sample_size: int
    ) -> List[Dict[str, Any]]:
        if dataset_name is None:
            return []

        if dataset_name in self.datasets:
            data = self.datasets[dataset_name]
        else:
            dataset = self.config["datasets"].get(dataset_name)
            if dataset is None:
                raise ValueError(
                    f"Dataset '{dataset_name}' not found in config or previous steps."
                )
            if dataset["type"] == "file":
                with open(dataset["path"], "r") as f:
                    data = json.load(f)
            else:
                raise ValueError(f"Unsupported dataset type: {dataset['type']}")

        if op_config:
            if sample_size == float("inf"):
                return data

            if op_config.get("type") == "reduce":
                return self._get_reduce_sample(
                    data, op_config.get("reduce_key"), sample_size
                )

        return random.sample(data, min(sample_size, len(data)))

    def _get_reduce_sample(
        self, data: List[Dict[str, Any]], reduce_key: str, sample_size: int
    ) -> List[Dict[str, Any]]:
        # Group data by reduce key
        grouped_data = defaultdict(list)
        for item in data:
            grouped_data[item[reduce_key]].append(item)

        # Sort groups by size in descending order
        sorted_groups = sorted(
            grouped_data.items(), key=lambda x: len(x[1]), reverse=True
        )

        sample = []

        # Take the top 5 groups
        top_5_groups = sorted_groups[:5]

        # Calculate the total count of items in the top 5 groups
        total_count = sum(len(items) for _, items in top_5_groups)

        sample = []
        for _, items in top_5_groups:
            # Calculate the proportion of items to sample from this group
            group_proportion = len(items) / total_count
            group_sample_size = int(sample_size * group_proportion)

            # Sample from the group
            group_sample = random.sample(items, min(group_sample_size, len(items)))
            sample.extend(group_sample)

        # If we haven't reached the desired sample size, add more items randomly
        if len(sample) < sample_size:
            remaining_items = [
                item
                for _, items in top_5_groups
                for item in items
                if item not in sample
            ]
            additional_sample = random.sample(
                remaining_items,
                min(sample_size - len(sample), len(remaining_items)),
            )
            sample.extend(additional_sample)

        # Add items randomly from non-top groups to meet the sample size
        if len(sample) < sample_size:
            remaining_items = [
                item
                for _, items in grouped_data.items()
                for item in items
                if item not in sample
            ]
            additional_sample = random.sample(
                remaining_items,
                min(sample_size - len(sample), len(remaining_items)),
            )
            sample.extend(additional_sample)

        # Create a histogram of group sizes
        group_sizes = [len(items) for _, items in grouped_data.items()]
        size_counts = Counter(group_sizes)

        # Sort the sizes for a more readable output
        sorted_sizes = sorted(size_counts.items())

        # Print the histogram
        self.console.print("\n[bold]Histogram of Group Sizes:[/bold]")
        max_bar_width, max_count = 2, max(size_counts.values())
        for size, count in sorted_sizes[:5]:
            normalized_count = int(count / max_count * max_bar_width)
            bar = "█" * normalized_count
            self.console.print(f"{size:3d}: {bar} ({count})")
        self.console.print("\n")

        return sample

    def _optimize_reduce(
        self, op_config: Dict[str, Any], input_data: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        reduce_optimizer = ReduceOptimizer(
            self.config,
            self.console,
            self.llm_client,
            self.max_threads,
            self._run_operation,
        )
        optimized_op, input_data = reduce_optimizer.optimize(op_config, input_data)
        return [optimized_op], input_data

    def _optimize_equijoin(
        self,
        op_config: Dict[str, Any],
        left_data: List[Dict[str, Any]],
        right_data: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        join_optimizer = JoinOptimizer(
            self.config, op_config, self.console, self.llm_client, self.max_threads
        )
        optimized_config, cost = join_optimizer.optimize_equijoin(left_data, right_data)
        self.operations_cost += cost

        # Update the operation config with the optimized values
        op_config.update(optimized_config)

        # Run the optimized operation
        output_data = self._run_operation(
            op_config, {"left": left_data, "right": right_data}
        )

        return [op_config], output_data

    def _optimize_map(
        self, op_config: Dict[str, Any], input_data: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        map_optimizer = MapOptimizer(
            self.config,
            self.console,
            self.llm_client,
            self.max_threads,
            self._run_operation,
            timeout=self.timeout,
        )
        return map_optimizer.optimize(op_config, input_data)

    def _optimize_resolve(
        self, op_config: Dict[str, Any], input_data: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        optimized_config, cost = JoinOptimizer(
            self.config, op_config, self.console, self.llm_client, self.max_threads
        ).optimize_resolve(input_data)
        self.operations_cost += cost

        # Update the operation config with the optimized values
        op_config.update(optimized_config)

        # Run the optimized operation
        output_data = self._run_operation(op_config, input_data)

        return [op_config], output_data

    def _run_operation(
        self,
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
        return_instance: bool = False,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], BaseOperation]]:
        operation_class = get_operation(op_config["type"])
        operation_instance = operation_class(
            op_config, self.config["default_model"], self.max_threads, self.console
        )
        if op_config["type"] == "equijoin":
            left_data = input_data["left"]
            right_data = input_data["right"]
            output_data, cost = operation_instance.execute(left_data, right_data)
        else:
            output_data, cost = operation_instance.execute(input_data)
        self.operations_cost += cost
        if return_instance:
            return output_data, operation_instance
        else:
            return output_data

    def _save_optimized_config(self):
        # Create a copy of the optimized config to modify
        config_to_save = self.optimized_config.copy()

        # Recursively resolve all anchors and aliases
        def resolve_anchors(data):
            if isinstance(data, dict):
                return {k: resolve_anchors(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [resolve_anchors(item) for item in data]
            else:
                return data

        resolved_config = resolve_anchors(config_to_save)

        # Use safe_dump to avoid creating anchors and aliases
        # Get the base filename without extension
        base_filename = os.path.splitext(self.yaml_file_path)[0]

        # Append '_opt' to the base filename
        optimized_filename = f"{base_filename}_opt.yaml"

        with open(optimized_filename, "w") as f:
            yaml.safe_dump(resolved_config, f, default_flow_style=False)

        self.console.print(
            f"[green italic]Optimized config saved to {optimized_filename}[/green italic]"
        )


if __name__ == "__main__":
    optimizer = Optimizer("workloads/medical/full.yaml", model="gpt-4o")
    optimizer.optimize()
