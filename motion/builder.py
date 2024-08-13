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
import os
import jinja2
from jinja2 import Environment, meta
import re
from motion.optimizers.utils import extract_jinja_variables, LLMClient


SUPPORTED_OPS = ["map", "resolve", "reduce", "equijoin"]


SAMPLE_SIZE_MAP = {
    "reduce": 40,
    "map": 5,
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
        """
        Initialize the Optimizer class.

        This method sets up the optimizer with the given configuration file and parameters.
        It loads the configuration, initializes the console for output, sets up the LLM client,
        and prepares various attributes for optimization.

        Args:
            yaml_file (str): Path to the YAML configuration file.
            max_threads (Optional[int]): Maximum number of threads to use for parallel processing.
                If None, it will be set to (number of CPUs * 4).
            model (str): The name of the language model to use. Defaults to "gpt-4o".
            timeout (int): Timeout in seconds for operations. Defaults to 60.

        Attributes:
            yaml_file_path (str): Stores the path to the YAML file.
            config (Dict): Stores the loaded configuration from the YAML file.
            console (Console): Rich console for formatted output.
            optimized_config (Dict): A copy of the original config to be optimized.
            llm_client (LLMClient): Client for interacting with the language model.
            max_threads (int): Maximum number of threads for parallel processing.
            operations_cost (float): Tracks the total cost of operations.
            timeout (int): Timeout for operations in seconds.
            selectivities (defaultdict): Stores selectivity information for operations.
                Selectivity is the ratio of output size to input size for an operation.
                It's used to estimate how much data will flow through the pipeline after
                each operation, which helps in optimizing subsequent operations and
                determining appropriate sample sizes. For example, a selectivity of 0.5
                means an operation halves the size of its input data.
            datasets (Dict): Stores loaded datasets.

        The method also calls print_optimizer_config() to display the initial configuration.
        """
        self.yaml_file_path = yaml_file
        self.config = load_config(yaml_file)
        self.console = Console()
        self.optimized_config = self.config.copy()
        self.llm_client = LLMClient(model)
        self.max_threads = max_threads or (os.cpu_count() or 1) * 4
        self.operations_cost = 0
        self.timeout = timeout
        self.selectivities = defaultdict(dict)
        self.datasets = {}
        self.optimized_config_path = f"{yaml_file}_opt.yaml"

        self.print_optimizer_config()

    def syntax_check(self):
        """
        Perform a syntax check on all operations defined in the configuration.

        This method validates each operation by attempting to instantiate it.
        If any operation fails to instantiate, a ValueError is raised.

        Raises:
            ValueError: If any operation fails the syntax check.
        """
        for operation in self.config["operations"]:
            operation_config = self.config["operations"][operation]
            operation_type = operation_config["type"]

            try:
                operation_class = get_operation(operation_type)
                operation_class(
                    operation_config,
                    self.config.get("default_model", "gpt-4o-mini"),
                    self.max_threads,
                    self.console,
                )
            except Exception as e:
                raise ValueError(
                    f"Syntax check failed for operation '{operation}': {str(e)}"
                )

        self.console.log("[green]Syntax check passed for all operations.[/green]")

    def print_optimizer_config(self):
        """
        Print the current configuration of the optimizer.

        This method uses the Rich console to display a formatted output of the optimizer's
        configuration. It includes details such as the YAML file path, sample sizes for
        different operation types, maximum number of threads, the language model being used,
        and the timeout setting.

        The output is color-coded and formatted for easy readability, with a header and
        separator lines to clearly delineate the configuration information.
        """
        self.console.log("[bold cyan]Optimizer Configuration:[/bold cyan]")
        self.console.log("─" * 40)
        self.console.log(f"[yellow]YAML File:[/yellow] {self.yaml_file_path}")
        self.console.log(f"[yellow]Sample Size:[/yellow] {SAMPLE_SIZE_MAP}")
        self.console.log(f"[yellow]Max Threads:[/yellow] {self.max_threads}")
        self.console.log(f"[yellow]Model:[/yellow] {self.llm_client.model}")
        self.console.log(f"[yellow]Timeout:[/yellow] {self.timeout} seconds")
        self.console.log("─" * 40)

    def compute_sample_size(
        self,
        step_name: str,
        step_ops: List[str],
        op_config: Dict[str, Any],
    ) -> int:
        """
        Compute the sample size necessary for optimizing given operation based on upstream operations.

        This method calculates an appropriate sample size for an operation, taking into
        account the selectivities of upstream operations in the same step. It uses a
        predefined sample size map (SAMPLE_SIZE_MAP) as a starting point.

        For example, if we have a 'map' operation with a default sample size of 10,
        and one upstream operation with a selectivity of 0.5, the computed sample size for the upstream operation would be:
        10 / 0.5 = 20

        This ensures that after applying the selectivity of the upstream operation,
        we still have a representative sample size for the current operation.

        Args:
            step_name (str): The name of the current step in the pipeline.
            step_ops (List[str]): A list of all operations in the current step.
            op_config (Dict[str, Any]): The configuration dictionary for the current operation.

        Returns:
            int: The computed sample size for the operation.

        The method works as follows:
        1. If there are no upstream operations, it returns the default sample size for the operation type.
        2. Otherwise, it starts with the default sample size and adjusts it based on the selectivities
           of upstream operations.
        3. It iterates through upstream operations in reverse order, dividing the sample size by
           each operation's selectivity.
        4. The final result is rounded to the nearest integer.

        Raises:
            ValueError: If the selectivity for any upstream operation is not found.

        Note:
            - The method assumes that selectivities for all upstream operations have been
              previously computed and stored in self.selectivities.
            - The sample size is always at least 1, even after all adjustments.
        """
        # If an equijoin, load the default. Equijoins are always first
        if op_config.get("type") == "equijoin":
            return SAMPLE_SIZE_MAP.get(op_config.get("type"))

        # If there are no upstream operations, use the default sample_size
        upstream_ops = []
        for step_op in step_ops:
            if step_op != op_config.get("name"):
                upstream_ops.append(step_op)
            else:
                break

        if len(upstream_ops) == 0:
            return SAMPLE_SIZE_MAP.get(op_config.get("type"), float("inf"))

        # Otherwise, compute the sample size based on the upstream operations
        sample_size = SAMPLE_SIZE_MAP.get(op_config.get("type"), 1)

        for op in reversed(upstream_ops):
            # Use the selectivity of the upstream operation to compute the sample size
            if op not in self.selectivities[step_name]:
                raise ValueError(
                    f"Selectivity for operation {op} not found in selectivities. Other ops are {self.selectivities[step_name]}"
                )
            sample_size = sample_size / self.selectivities[step_name].get(op)

        return int(round(sample_size))

    def _insert_empty_resolve_operations(self):
        """
        Determines whether to insert resolve operations in the pipeline.

        This method iterates through each step in the pipeline and checks if there's a reduce
        operation that follows a map operation with no resolver in between. If such a case is
        found, it synthesizes an empty resolver operation and inserts it into the pipeline.

        The method modifies the pipeline configuration in-place.

        Returns:
            None

        Side effects:
        - Modifies self.config["pipeline"]["steps"] by potentially inserting new resolve operations.
        - Adds new resolve operations to self.config["operations"] if necessary.
        """
        for i, step in enumerate(self.config["pipeline"]["steps"]):
            operations = step.get("operations", [])
            has_map = False
            has_reduce = False
            has_resolve = False
            map_op = None
            reduce_op = None

            for op in operations:
                op_type = self.config["operations"][op].get("type")
                if op_type == "map":
                    has_map = True
                    map_op = op
                elif op_type == "reduce":
                    has_reduce = True
                    reduce_op = op
                elif op_type == "resolve":
                    has_resolve = True

            if has_map and has_reduce and not has_resolve:
                # Synthesize an empty resolver
                self.console.log(
                    f"[yellow]Synthesizing empty resolver operation:[/yellow]"
                )
                self.console.log(
                    f"  • [cyan]Reduce operation:[/cyan] [bold]{reduce_op}[/bold]"
                )
                self.console.log(f"  • [cyan]Step:[/cyan] [bold]{step['name']}[/bold]")

                new_resolve_op = f"synthesized_resolve_{i}"
                self.config["operations"][new_resolve_op] = {
                    "type": "resolve",
                    "empty": True,
                    "embedding_model": "text-embedding-3-small",
                    "resolution_model": self.config.get("default_model", "gpt-4o-mini"),
                    "comparison_model": self.config.get("default_model", "gpt-4o-mini"),
                    "_intermediates": {
                        "map_prompt": self.config["operations"][map_op].get("prompt"),
                        "reduce_key": self.config["operations"][reduce_op].get(
                            "reduce_key"
                        ),
                    },
                }

                # Insert the new resolve operation before the reduce operation
                reduce_index = next(
                    i
                    for i, op in enumerate(operations)
                    if self.config["operations"][op].get("type") == "reduce"
                )
                operations.insert(reduce_index, new_resolve_op)

                has_resolve = True

        # Update the pipeline configuration
        self.config["pipeline"]["steps"] = self.config["pipeline"]["steps"]

    def optimize(self):
        """
        Optimize the entire pipeline defined in the configuration.

        This method is the main entry point for the optimization process. It iterates through
        each step in the pipeline, optimizing from upstream to downstream, and constructs an
        optimized version of the configuration.

        The optimization process includes:
        1. Iterating through each step in the pipeline, from upstream to downstream.
        2. Optimizing each step using the _optimize_step method.
        3. Updating the optimized configuration with the new operations and steps.
        4. Saving the optimized configuration to a file.
        5. Logging the total costs (agent cost, operations cost, and total cost).

        Returns:
            None

        Side effects:
        - Modifies self.optimized_config with the optimized pipeline and operations.
        - Updates self.datasets with the results of each step.
        - Calls _save_optimized_config to save the optimized configuration to a file.
        - Logs cost information to the console.

        Raises:
            ValueError: If a step in the pipeline does not have a name.

        Note:
        - This method assumes that all necessary data and configurations are already
          loaded and initialized in the Optimizer instance.
        - The optimization process is performed step by step, from upstream to downstream,
          with each step potentially depending on the results of previous steps.
        """
        self.syntax_check()
        self._insert_empty_resolve_operations()

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

        self.console.log(
            f"[bold]Total agent cost: ${self.llm_client.total_cost:.2f}[/bold]"
        )
        self.console.log(
            f"[bold]Total operations cost: ${self.operations_cost:.2f}[/bold]"
        )
        self.console.log(
            f"[bold]Total cost: ${self.llm_client.total_cost + self.operations_cost:.2f}[/bold]"
        )

    def _run_partial_step(
        self,
        step: Dict[str, Any],
        ops_to_run: List[str],
        sample_size: int,
        optimized_operations: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Execute a partial step of the pipeline on a sample of the input data.

        This internal method runs a subset of operations for a given step on a sample
        of the input data. It's used as part of the optimization process to evaluate
        and optimize individual operations within a step.

        Args:
            step (Dict[str, Any]): The step configuration dictionary.
            ops_to_run (List[str]): List of operation names to execute in this partial step.
            sample_size (int): The number of items to include in the input sample.
            optimized_operations (Dict[str, Dict[str, Any]]): Dictionary of optimized operations.

        Returns:
            List[Dict[str, Any]]: The output data after running the specified operations.

        The method performs the following steps:
        1. Retrieves a sample of the input data using _get_sample_data.
        2. For equijoin operations, it loads both left and right datasets.
        3. Iterates through the specified operations, running each on the input sample.
        4. Returns the final output after all specified operations have been applied.

        Note:
        - The method handles both regular steps and equijoin steps differently.

        Raises:
            Any exceptions raised by _get_sample_data or _run_operation methods.
        """
        # Take the input data and run the operations in ops_to_run
        # Return the output data
        input_sample = self._get_sample_data(step.get("input"), None, sample_size)

        if step.get("input") is None:
            join_op_name = list(step.get("operations")[0].keys())[0]
            # this is an equijoin step, load left and right datasets
            left_data = self._get_sample_data(
                step.get("operations")[0][join_op_name].get("left"), None, sample_size
            )
            right_data = self._get_sample_data(
                step.get("operations")[0][join_op_name].get("right"), None, sample_size
            )
            input_sample = {"left": left_data, "right": right_data}

        for op in ops_to_run:
            op_object = optimized_operations[op]
            input_sample = self._run_operation(op_object, input_sample)
        return input_sample

    def _optimize_step(
        self, step: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Optimize a single step in the pipeline.

        This method takes a step configuration and optimizes each operation within it.
        It handles different types of operations, including those that require optimization
        and those that don't.

        Args:
            step (Dict[str, Any]): The configuration dictionary for the step to be optimized.

        Returns:
            Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
                - The optimized step configuration.
                - A list of optimized operations.
                - The output data after running all operations in the step.

        The method performs the following for each operation in the step:
        1. Extracts the operation configuration.
        2. Computes the appropriate sample size for the operation.
        3. Runs the operation on a sample of the input data.
        4. If the operation is optimizable and of a supported type, it calls the appropriate
           optimization method (e.g., _optimize_map, _optimize_reduce).
        5. If not optimizable or not supported, it runs the operation as-is.
        6. Calculates and stores the selectivity of each operation.
        7. Updates the list of optimized operations and their configurations.

        The method uses rich console to provide status updates during the optimization process.

        Note:
        - This method is a key part of the overall optimization process, focusing on
          individual steps in the pipeline.
        - It relies on several helper methods like _run_partial_step, compute_sample_size,
          and various _optimize_* methods for specific operation types.
        - When optimizing an operation in the step, all previous operations are run on the
          sample size needed for the current operation. This ensures that the input to the
          operation being optimized is representative of what it would receive in the full pipeline.

        Raises:
            ValueError: If an unsupported operation type is encountered.
        """
        optimized_operations = {}
        optimized_operation_names = []

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

            # Run the pipeline
            step_ops = step.get("operations")

            sample_size = self.compute_sample_size(
                step.get("name"), step_ops, op_object
            )
            input_data = self._run_partial_step(
                step, optimized_operation_names, sample_size, optimized_operations
            )

            if (
                op_object.get("optimize", True) == False
                or op_object.get("type") not in SUPPORTED_OPS
            ):
                # If optimize is False or operation type is not supported, just use the operation without optimization
                output_data = self._run_operation(op_object, input_data)
                optimized_operations[operation_name] = op_object
                optimized_operation_names.append(operation_name)

                selectivity = len(output_data) / len(input_data)

                self.selectivities[step.get("name")][operation_name] = selectivity
            else:
                # Use rich console status to indicate optimization of the operation
                with self.console.status(
                    f"[bold blue]Optimizing operation: {operation_name} (Type: {op_object['type']})[/bold blue]"
                ):
                    # Print the number of elements in input_data
                    self.console.log(f"[yellow]Optimizing Operation:[/yellow]")
                    self.console.log(f"[yellow]  Type: {op_object['type']}[/yellow]")
                    self.console.log(f"[yellow]  Name: {operation_name}[/yellow]")
                    if op_object.get("type") == "equijoin":
                        self.console.log(
                            f"[yellow]  Sample size (left): {len(input_data['left'])}[/yellow]"
                        )
                        self.console.log(
                            f"[yellow]  Sample size (right): {len(input_data['right'])}[/yellow]"
                        )
                    else:
                        self.console.log(
                            f"[yellow]  Sample size: {len(input_data)}[/yellow]"
                        )

                    # Run optimization
                    if op_object.get("type") == "map":
                        optimized_ops, output_data = self._optimize_map(
                            op_object, input_data
                        )
                    elif op_object.get("type") == "reduce":
                        optimized_ops, output_data = self._optimize_reduce(
                            op_object, input_data
                        )
                    elif op_object.get("type") == "resolve":
                        optimized_ops, output_data = self._optimize_resolve(
                            op_object, input_data
                        )
                    elif op_object.get("type") == "equijoin":
                        optimized_ops, output_data = self._optimize_equijoin(
                            op_object, input_data["left"], input_data["right"]
                        )
                    else:
                        raise ValueError(
                            f"Unsupported operation type: {op_object['type']}"
                        )

                    for op in optimized_ops:
                        op_name = op.pop("name")
                        optimized_operations[op_name] = op
                        optimized_operation_names.append(op_name)

                        old_input_data_size = len(input_data)
                        input_data = self._run_operation(op, input_data)
                        new_input_data_size = len(input_data)
                        selectivity = new_input_data_size / old_input_data_size
                        self.selectivities[step.get("name")][op_name] = selectivity

        optimized_step = step.copy()
        optimized_step["operations"] = optimized_operation_names
        return optimized_step, optimized_operations, output_data

    def _get_sample_data(
        self, dataset_name: str, op_config: Optional[Dict[str, Any]], sample_size: int
    ) -> List[Dict[str, Any]]:
        """
        Retrieve a sample of data from a specified dataset.

        This method loads data from either a previously processed dataset or from a file,
        and returns a sample of the data based on the given sample size and operation configuration.

        Args:
            dataset_name (str): The name of the dataset to sample from.
            op_config (Optional[Dict[str, Any]]): The configuration of the operation to be performed.
                                                  This is used to determine if special sampling is needed.
            sample_size (int): The desired size of the sample. If set to float('inf'), all data is returned.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the sampled data.

        Raises:
            ValueError: If the dataset is not found or if the dataset type is unsupported.
        """
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

        if sample_size == float("inf"):
            return data

        if op_config:
            if op_config.get("type") == "reduce":
                return self._get_reduce_sample(
                    data, op_config.get("reduce_key"), sample_size
                )

        return random.sample(data, min(sample_size, len(data)))

    def _get_reduce_sample(
        self, data: List[Dict[str, Any]], reduce_key: str, sample_size: int
    ) -> List[Dict[str, Any]]:
        """
        Get a representative sample for a reduce operation.

        This method creates a sample that preserves the distribution of groups in the data,
        focusing on the top 5 largest groups. It also generates and prints a histogram of group sizes.

        Args:
            data (List[Dict[str, Any]]): The full dataset to sample from.
            reduce_key (str): The key used for grouping in the reduce operation.
            sample_size (int): The desired size of the sample.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the sampled data.
        """
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
        self.console.log("\n[bold]Histogram of Group Sizes:[/bold]")
        max_bar_width, max_count = 2, max(size_counts.values())
        for size, count in sorted_sizes[:5]:
            normalized_count = int(count / max_count * max_bar_width)
            bar = "█" * normalized_count
            self.console.log(f"{size:3d}: {bar} ({count})")
        self.console.log("\n")

        return sample

    def _optimize_reduce(
        self, op_config: Dict[str, Any], input_data: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Optimize a reduce operation.

        This method creates a ReduceOptimizer instance and uses it to optimize the reduce operation.

        Args:
            op_config (Dict[str, Any]): The configuration of the reduce operation.
            input_data (List[Dict[str, Any]]): The input data for the reduce operation.

        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: A tuple containing the optimized operation
            configuration and the output data after applying the optimized operation.
        """
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
        """
        Optimize an equijoin operation.

        This method creates a JoinOptimizer instance and uses it to optimize the equijoin operation.
        It updates the operation cost and runs the optimized operation.

        Args:
            op_config (Dict[str, Any]): The configuration of the equijoin operation.
            left_data (List[Dict[str, Any]]): The left dataset for the join.
            right_data (List[Dict[str, Any]]): The right dataset for the join.

        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: A tuple containing the optimized operation
            configuration and the output data after applying the optimized operation.
        """
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
        """
        Optimize a map operation.

        This method creates a MapOptimizer instance and uses it to optimize the map operation.

        Args:
            op_config (Dict[str, Any]): The configuration of the map operation.
            input_data (List[Dict[str, Any]]): The input data for the map operation.

        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: A tuple containing the optimized operation
            configuration and the output data after applying the optimized operation.
        """
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
        """
        Optimize a resolve operation.

        This method creates a JoinOptimizer instance and uses it to optimize the resolve operation.
        It updates the operation cost and runs the optimized operation.

        Args:
            op_config (Dict[str, Any]): The configuration of the resolve operation.
            input_data (List[Dict[str, Any]]): The input data for the resolve operation.

        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: A tuple containing the optimized operation
            configuration and the output data after applying the optimized operation.
        """
        optimized_config, cost = JoinOptimizer(
            self.config, op_config, self.console, self.llm_client, self.max_threads
        ).optimize_resolve(input_data)

        if optimized_config.get("empty", False) == True:
            # Remove this operation from the pipeline and just return input data
            return [], input_data

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
        """
        Run a single operation based on its configuration.

        This method creates an instance of the appropriate operation class and executes it.
        It also updates the total operation cost.

        Args:
            op_config (Dict[str, Any]): The configuration of the operation to run.
            input_data (List[Dict[str, Any]]): The input data for the operation.
            return_instance (bool, optional): If True, return the operation instance along with the output data.

        Returns:
            Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], BaseOperation]]:
            If return_instance is False, returns the output data.
            If return_instance is True, returns a tuple of the output data and the operation instance.
        """
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

    # Recursively resolve all anchors and aliases
    @staticmethod
    def resolve_anchors(data):
        """
        Recursively resolve all anchors and aliases in a nested data structure.

        This static method traverses through dictionaries and lists, resolving any YAML anchors and aliases.

        Args:
            data: The data structure to resolve. Can be a dictionary, list, or any other type.

        Returns:
            The resolved data structure with all anchors and aliases replaced by their actual values.
        """
        if isinstance(data, dict):
            return {k: Optimizer.resolve_anchors(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [Optimizer.resolve_anchors(item) for item in data]
        else:
            return data

    def _save_optimized_config(self):
        """
        Save the optimized configuration to a YAML file.

        This method creates a copy of the optimized configuration, resolves all anchors and aliases,
        and saves it to a new YAML file. The new file name is based on the original file name with '_opt' appended.
        """
        # Create a copy of the optimized config to modify
        config_to_save = self.optimized_config.copy()

        resolved_config = Optimizer.resolve_anchors(config_to_save)

        with open(self.optimized_config_path, "w") as f:
            yaml.safe_dump(resolved_config, f, default_flow_style=False)

        self.console.log(
            f"[green italic]Optimized config saved to {self.optimized_config_path}[/green italic]"
        )


if __name__ == "__main__":
    optimizer = Optimizer("workloads/medical/synth_resolve.yaml", model="gpt-4o")
    optimizer.optimize()
