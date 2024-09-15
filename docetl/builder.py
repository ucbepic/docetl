import copy
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from rich.console import Console
from rich.status import Status
from rich.traceback import install

install(show_locals=True)

from docetl.operations import get_operation
from docetl.operations.base import BaseOperation
from docetl.operations.utils import flush_cache
from docetl.optimizers.join_optimizer import JoinOptimizer
from docetl.optimizers.map_optimizer import MapOptimizer
from docetl.optimizers.reduce_optimizer import ReduceOptimizer
from docetl.optimizers.utils import LLMClient
from docetl.utils import load_config

SUPPORTED_OPS = ["map", "resolve", "reduce", "equijoin", "filter"]
NUM_OPTIMIZER_RETRIES = 1

SAMPLE_SIZE_MAP = {
    "reduce": 40,
    "map": 5,
    "resolve": 100,
    "equijoin": 100,
    "filter": 5,
    "split": 100,
    "gather": 100,
}


class DatasetOnDisk(dict):
    def __init__(self, dir: str, console: Console):
        self.dir = dir
        self.console = console

    def __setitem__(self, key, value):
        self._save_to_disk(key, value)

    def __getitem__(self, key):
        with open(f"{self.dir}/{key}", "r") as f:
            self.console.log(f"Loading dataset from disk... {key}")
            return json.load(f)

    def _save_to_disk(self, save_suffix: str, value: Any):
        with open(f"{self.dir}/{save_suffix}", "w") as f:
            json.dump(value, f)
        self.console.log(
            f"[green]Saved intermediate results to disk at {self.dir}/{save_suffix}[/green]"
        )

    def __len__(self):
        return len(os.listdir(self.dir))

    def __iter__(self):
        return iter(os.listdir(self.dir))

    def __contains__(self, item):
        return item in os.listdir(self.dir)

    def keys(self):
        return os.listdir(self.dir)

    def values(self):
        return [self[key] for key in self.keys()]

    def items(self):
        return [(key, self[key]) for key in self.keys()]


class Optimizer:
    def __init__(
        self,
        yaml_file: str,
        max_threads: Optional[int] = None,
        model: str = "gpt-4o",
        resume: bool = False,
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
            resume (bool): Whether to resume optimization from a previous run. Defaults to False.
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
        self.optimized_config = copy.deepcopy(self.config)
        self.llm_client = LLMClient(model)
        self.max_threads = max_threads or (os.cpu_count() or 1) * 4
        self.operations_cost = 0
        self.timeout = timeout
        self.selectivities = defaultdict(dict)
        self.samples_taken = defaultdict(dict)
        self.resume = resume

        home_dir = os.path.expanduser("~")
        yaml_file_suffix = yaml_file.split("/")[-1].split(".")[0]
        cache_dir = os.path.join(home_dir, f".docetl/cache/{yaml_file_suffix}")
        os.makedirs(cache_dir, exist_ok=True)
        self.datasets = DatasetOnDisk(dir=cache_dir, console=self.console)
        self.optimized_ops_path = f"{cache_dir}/optimized_ops"
        base_name = yaml_file.rsplit(".", 1)[0]
        self.optimized_config_path = f"{base_name}_opt.yaml"

        # Update sample size map
        self.sample_size_map = SAMPLE_SIZE_MAP
        if self.config.get("optimizer_config", {}).get("sample_sizes", {}):
            self.sample_size_map.update(self.config["optimizer_config"]["sample_sizes"])

        self.status = None
        self.step_op_to_optimized_ops = {}

        self.print_optimizer_config()

    def find_operation(self, op_name: str, config: Optional[Dict] = None) -> Dict:
        if not config:
            config = self.config
        for operation_config in config["operations"]:
            if operation_config["name"] == op_name:
                return operation_config
        raise ValueError(f"Operation '{op_name}' not found in configuration.")

    def syntax_check(self):
        """
        Perform a syntax check on all operations defined in the configuration.

        This method validates each operation by attempting to instantiate it.
        If any operation fails to instantiate, a ValueError is raised.

        Raises:
            ValueError: If any operation fails the syntax check.
        """
        for operation_config in self.config["operations"]:
            operation = operation_config["name"]
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
        self.console.rule("[bold cyan]Optimizer Configuration[/bold cyan]")
        self.console.log(f"[yellow]YAML File:[/yellow] {self.yaml_file_path}")
        self.console.log(f"[yellow]Sample Size:[/yellow] {self.sample_size_map}")
        self.console.log(f"[yellow]Max Threads:[/yellow] {self.max_threads}")
        self.console.log(f"[yellow]Model:[/yellow] {self.llm_client.model}")
        self.console.log(f"[yellow]Timeout:[/yellow] {self.timeout} seconds")

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
                if step_op in self.step_op_to_optimized_ops:
                    upstream_ops.extend(self.step_op_to_optimized_ops[step_op])
                else:
                    upstream_ops.append(step_op)
            else:
                break

        if len(upstream_ops) == 0:
            return self.sample_size_map.get(op_config.get("type"), float("inf"))

        # Otherwise, compute the sample size based on the upstream operations
        sample_size = self.sample_size_map.get(op_config.get("type"), 100)
        for op in reversed(upstream_ops):
            # Use the selectivity of the upstream operation to compute the sample size
            if op not in self.selectivities[step_name]:
                raise ValueError(
                    f"Selectivity for operation {op} not found in selectivities. Other ops are {self.selectivities[step_name]}"
                )

            sample_size = sample_size / self.selectivities[step_name].get(op)

        return int(math.ceil(sample_size))

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
                if isinstance(op, dict):
                    op = list(op.keys())[0]
                op_config = self.find_operation(op)
                op_type = op_config["type"]
                if op_type == "map":
                    has_map = True
                    map_op = op
                elif op_type == "reduce" and op_config.get("synthesize_resolve", True):
                    has_reduce = True
                    reduce_op = op
                elif op_type == "resolve":
                    has_resolve = True

            if has_map and has_reduce and not has_resolve:
                # Synthesize an empty resolver
                self.console.log(
                    "[yellow]Synthesizing empty resolver operation:[/yellow]"
                )
                self.console.log(
                    f"  • [cyan]Reduce operation:[/cyan] [bold]{reduce_op}[/bold]"
                )
                self.console.log(f"  • [cyan]Step:[/cyan] [bold]{step['name']}[/bold]")

                new_resolve_op = f"synthesized_resolve_{i}"
                reduce_key = self.find_operation(reduce_op).get("reduce_key")
                if isinstance(reduce_key, str):
                    reduce_key = [reduce_key]
                self.config["operations"].append(
                    {
                        "name": new_resolve_op,
                        "type": "resolve",
                        "empty": True,
                        "optimize": True,
                        "embedding_model": "text-embedding-3-small",
                        "resolution_model": self.config.get(
                            "default_model", "gpt-4o-mini"
                        ),
                        "comparison_model": self.config.get(
                            "default_model", "gpt-4o-mini"
                        ),
                        "_intermediates": {
                            "map_prompt": self.find_operation(map_op).get("prompt"),
                            "reduce_key": reduce_key,
                        },
                    }
                )

                # Insert the new resolve operation before the reduce operation
                reduce_index = next(
                    i
                    for i, op in enumerate(operations)
                    if self.find_operation(op).get("type") == "reduce"
                )
                operations.insert(reduce_index, new_resolve_op)

                has_resolve = True

            self.config["pipeline"]["steps"][i]["operations"] = operations

        # Update the pipeline configuration
        self.config["pipeline"]["steps"] = self.config["pipeline"]["steps"]

    def _add_map_prompts_to_reduce_operations(self):
        """
        Add relevant map prompts to reduce operations based on their reduce keys.

        This method iterates through all map operations to create a dictionary mapping
        output schema keys to map prompts. It then loops through reduce operations,
        adding the relevant map prompts based on the reduce keys and output schema.

        Side effects:
        - Modifies reduce operations in self.config["operations"] by adding map prompts.
        """
        # Create a dictionary mapping output schema keys to map prompts
        output_key_to_prompt = {}
        for op_config in self.config["operations"]:
            if op_config.get("type") == "map":
                output_schema = op_config.get("output", {}).get("schema", {})
                prompt = op_config.get("prompt", "")
                for key in output_schema.keys():
                    output_key_to_prompt[key] = prompt

        # Add relevant map prompts to reduce operations
        for op_config in self.config["operations"]:
            if op_config.get("type") == "reduce":
                reduce_keys = op_config.get("reduce_key", [])
                if isinstance(reduce_keys, str):
                    reduce_keys = [reduce_keys]

                relevant_prompts = []
                for key in reduce_keys:
                    if key in output_key_to_prompt:
                        relevant_prompts.append(output_key_to_prompt[key])

                if relevant_prompts:
                    op_config["_intermediates"] = op_config.get("_intermediates", {})
                    op_config["_intermediates"]["last_map_prompt"] = relevant_prompts[
                        -1
                    ]

    def _load_optimized_ops(self):
        """
        Load the optimized operations from disk.
        """
        if os.path.exists(self.optimized_ops_path):
            for filename in os.listdir(self.optimized_ops_path):
                if filename.endswith(".json"):
                    original_op_name = filename[:-5]  # Remove '.json' from the filename
                    with open(
                        os.path.join(self.optimized_ops_path, filename), "r"
                    ) as f:
                        optimized_ops = json.load(f)

                    # Update the config with the optimized operations
                    if original_op_name in [
                        op["name"] for op in self.config["operations"]
                    ]:
                        # Update the config with the optimized operations
                        # First, remove all operations that are already in the config with the same name
                        self.config["operations"] = [
                            op
                            for op in self.config["operations"]
                            if op["name"] != original_op_name
                        ]

                        for op in optimized_ops:
                            op["optimize"] = False
                            self.config["operations"].append(op)

                        # Update the step operations
                        for step in self.config["pipeline"]["steps"]:
                            if original_op_name in step["operations"]:
                                index = step["operations"].index(original_op_name)
                                step["operations"] = (
                                    step["operations"][:index]
                                    + [op["name"] for op in optimized_ops]
                                    + step["operations"][index + 1 :]
                                )

                    self.console.log(
                        f"Loaded optimized operations for {original_op_name}"
                    )

            self.console.log("[green]Finished loading optimized operations[/green]")

            # Print out the operations for each step
            self.console.log("[bold blue]Operations for each step:[/bold blue]")
            for step in self.config["pipeline"]["steps"]:
                step_name = step.get("name")
                operations = step.get("operations", [])
                self.console.log(f"[cyan]Step: {step_name}[/cyan]")
                for op in operations:
                    if isinstance(op, dict):
                        # Handle the case where the operation is a dictionary (e.g., for equijoin)
                        op_name = list(op.keys())[0]
                        op_details = op[op_name]
                        self.console.log(f"  - {op_name}: {op_details}")
                    else:
                        self.console.log(f"  - {op}")
                self.console.log("")  # Add a blank line between steps
        else:
            self.console.log("[yellow]No optimized operations found[/yellow]")

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
        self.console.rule("[bold cyan]Beginning Pipeline Optimization[/bold cyan]")

        self.syntax_check()

        self._insert_empty_resolve_operations()

        # If resume is True, load the optimized operations from disk
        if self.resume:
            self._load_optimized_ops()

        for step in self.config["pipeline"]["steps"]:
            step_name = step.get("name")
            if not step_name:
                raise ValueError(
                    "Step does not have a name. Each step must have a unique name."
                )

            optimized_step, step_operations, input_data = self._optimize_step(step)
            old_op_names = [
                op
                for op in step["operations"]
                if op not in optimized_step["operations"]
            ]

            # Remove all old_op_names from self.optimized_config["operations"]
            self.optimized_config["operations"] = [
                op
                for op in self.optimized_config["operations"]
                if op["name"] not in old_op_names
            ]

            for op in optimized_step["operations"]:
                changed_op = False
                for i, op_config in enumerate(self.optimized_config["operations"]):
                    if op_config["name"] == op:
                        self.optimized_config["operations"][i] = step_operations[op]
                        changed_op = True
                if not changed_op:
                    self.optimized_config["operations"].append(step_operations[op])

            self.optimized_config["pipeline"]["steps"] = [
                step
                for step in self.optimized_config["pipeline"]["steps"]
                if step["name"] != step_name
            ] + [optimized_step]

            self.step_op_to_optimized_ops[step_name] = optimized_step["operations"]

            step_hash = (
                hashlib.md5(
                    json.dumps(
                        {
                            "step": [
                                s
                                for s in self.optimized_config["pipeline"]["steps"]
                                if s["name"] == step_name
                            ][0],
                            "operations": [
                                self.find_operation(op, self.optimized_config)
                                for op in optimized_step["operations"]
                            ],
                        }
                    ).encode()
                ).hexdigest()
                + ".json"
            )
            # If the dataset already exists, skip the step
            if step_hash in self.datasets:
                continue

            flush_cache(self.console)

            if step_name in self.config.get("optimizer_config", {}).get(
                "run_full_step", []
            ):
                # Run the entire step
                input_data = self._run_partial_step(
                    step,
                    step_operations,
                    float("inf"),  # TODO: FIX THIS
                )
                self.datasets[step_hash] = copy.deepcopy(input_data)
            else:
                self.datasets[step_hash] = copy.deepcopy(input_data)

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
            if "name" not in op_object:
                op_object["name"] = op

            input_sample = self._run_operation(op_object, input_sample)
        return input_sample

    def _optimize_step(
        self, step: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
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
        replacement_operations = {}  # List from old op name to new ops

        for op_idx, operation in enumerate(step["operations"]):
            if isinstance(operation, dict):
                operation_name = list(operation.keys())[0]
                operation_config = operation[operation_name]
            else:
                operation_name = operation
                operation_config = {}

            op_object = self.find_operation(operation_name).copy()
            op_object.update(operation_config)
            op_object["name"] = operation_name

            # Run the pipeline
            step_ops = []
            for step_op in step.get("operations"):
                if step_op in replacement_operations:
                    step_ops.extend(replacement_operations[step_op])
                else:
                    step_ops.append(step_op)

            # TODO: incorporate this into the optimizer to not run the most downstream operations
            downstream_ops_exist = op_idx < len(step["operations"]) - 1

            sample_size = self.compute_sample_size(
                step.get("name"), step_ops, op_object
            )
            input_data = self._run_partial_step(
                step, optimized_operation_names, sample_size, optimized_operations
            )

            if (
                not op_object.get("optimize", False)  # Default don't optimize
                or op_object.get("type") not in SUPPORTED_OPS
            ):
                # If optimize is False or operation type is not supported, just use the operation without optimization
                output_data = self._run_operation(op_object, input_data)
                optimized_operations[operation_name] = op_object
                optimized_operation_names.append(operation_name)

                selectivity = len(output_data) / len(input_data)

                self.selectivities[step.get("name")][operation_name] = selectivity
                self.samples_taken[step.get("name")][operation_name] = sample_size
            else:
                # Use rich console status to indicate optimization of the operation
                with self.console.status(
                    f"[bold blue]Optimizing operation: {operation_name} (Type: {op_object['type']})[/bold blue]"
                ) as status:
                    self.status = status

                    # Print the number of elements in input_data
                    self.console.rule(
                        f"[yellow]Optimizing operation {operation_name} (Type: {op_object['type']})[/yellow]"
                    )
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
                    for retry in range(
                        self.config.get("optimizer_config", {}).get(
                            "num_retries", NUM_OPTIMIZER_RETRIES
                        )
                    ):
                        try:
                            if op_object.get("type") == "map":
                                optimized_ops = self._optimize_map(
                                    op_object, input_data
                                )
                            elif op_object.get("type") == "filter":
                                optimized_ops = self._optimize_map(
                                    op_object, input_data, is_filter=True
                                )
                            elif op_object.get("type") == "reduce":
                                optimized_ops = self._optimize_reduce(
                                    op_object, input_data, status
                                )
                            elif op_object.get("type") == "resolve":
                                optimized_ops = self._optimize_resolve(
                                    op_object, input_data
                                )
                            elif op_object.get("type") == "equijoin":
                                (
                                    optimized_ops,
                                    input_data,
                                    new_left_name,
                                    new_right_name,
                                ) = self._optimize_equijoin(
                                    op_object,
                                    operation["left"],
                                    operation["right"],
                                    input_data["left"],
                                    input_data["right"],
                                    status,
                                )
                            else:
                                raise ValueError(
                                    f"Unsupported operation type: {op_object['type']}"
                                )
                            break  # If successful, break out of the retry loop
                        except Exception as e:
                            if (
                                retry
                                == self.config.get("optimizer_config", {}).get(
                                    "num_retries", NUM_OPTIMIZER_RETRIES
                                )
                                - 1
                            ):
                                raise  # If this was the last retry, re-raise the exception
                            self.console.log(
                                f"Optimization attempt {retry + 1} failed. Retrying..."
                            )

                    if self.status:
                        self.status.update(
                            f"[bold blue]Running optimized operation to estimate selectivities: {operation_name}[/bold blue]"
                        )

                    for op in optimized_ops:
                        op_name = op["name"]
                        optimized_operations[op_name] = op
                        if op.get("type") == "equijoin":
                            optimized_operation_names.append(
                                {
                                    op_name: {
                                        "left": new_left_name,
                                        "right": new_right_name,
                                    }
                                }
                            )
                        else:
                            optimized_operation_names.append(op_name)

                        old_input_data_size = len(input_data)
                        input_data = self._run_operation(op, input_data)
                        new_input_data_size = len(input_data)
                        selectivity = new_input_data_size / old_input_data_size
                        self.selectivities[step.get("name")][op_name] = selectivity
                        self.samples_taken[step.get("name")][op_name] = sample_size

                    # Set replacement_operations
                    replacement_operations[op_object["name"]] = [
                        o["name"] for o in optimized_ops
                    ]

                    # Print new operator configs
                    self.console.log("[bold green]New op configurations:[/bold green]")
                    for op_name, op_config in optimized_operations.items():
                        if op_name in [o["name"] for o in optimized_ops]:
                            self.console.log(
                                f"[cyan]{op_name}:[/cyan] {json.dumps(op_config, indent=2)}"
                            )

                    # Save the optimized operations to disk
                    os.makedirs(self.optimized_ops_path, exist_ok=True)

                    for original_op, replacement_ops in replacement_operations.items():
                        optimized_ops_list = [
                            (
                                optimized_operations[op_name]
                                if isinstance(op_name, str)
                                else {
                                    list(op_name.keys())[0]: optimized_operations[
                                        list(op_name.keys())[0]
                                    ]
                                }
                            )
                            for op_name in replacement_ops
                        ]

                        # Save to disk
                        optimized_op_file = os.path.join(
                            self.optimized_ops_path, f"{original_op}.json"
                        )
                        with open(optimized_op_file, "w") as f:
                            json.dump(optimized_ops_list, f, indent=2)

                    self.console.log(
                        f"[green]Saved optimized operations to {self.optimized_ops_path}[/green]"
                    )
                    self.status = None
                    output_data = input_data

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

        if any(
            s["name"] == dataset_name
            for s in self.optimized_config["pipeline"]["steps"]
        ):
            step = [
                s
                for s in self.optimized_config["pipeline"]["steps"]
                if s["name"] == dataset_name
            ][0]
            name_hash = (
                hashlib.md5(
                    json.dumps(
                        {
                            "step": step,
                            "operations": [
                                self.find_operation(op) for op in step["operations"]
                            ],
                        }
                    ).encode()
                ).hexdigest()
                + ".json"
            )
        else:
            name_hash = None

        if name_hash and name_hash in self.datasets:
            data = self.datasets[name_hash]
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

        # Take the random 500 examples or all if less than 500
        initial_data = random.sample(data, min(500, len(data)))

        # Calculate counts for each example
        char_counts = [len(str(item)) for item in initial_data]
        total_counts = sum(char_counts)

        # Calculate weights based on word counts
        weights = [count / total_counts for count in char_counts]

        # Perform weighted random sampling
        return random.choices(
            initial_data, weights=weights, k=min(sample_size, len(initial_data))
        )

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
        self,
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
        status: Status,
    ) -> List[Dict[str, Any]]:
        """
        Optimize a reduce operation.

        This method creates a ReduceOptimizer instance and uses it to optimize the reduce operation.

        Args:
            op_config (Dict[str, Any]): The configuration of the reduce operation.
            input_data (List[Dict[str, Any]]): The input data for the reduce operation.
            status (Status): The status object to update with the progress of the optimization.

        Returns:
            List[Dict[str, Any]]: The optimized operation configuration.
        """
        reduce_optimizer = ReduceOptimizer(
            self.config,
            self.console,
            self.llm_client,
            self.max_threads,
            self._run_operation,
            status=status,
        )
        optimized_ops, _, cost = reduce_optimizer.optimize(op_config, input_data)
        self.operations_cost += cost
        return optimized_ops

    def _optimize_equijoin(
        self,
        op_config: Dict[str, Any],
        left_name: str,
        right_name: str,
        left_data: List[Dict[str, Any]],
        right_data: List[Dict[str, Any]],
        status: Status,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], str, str]:
        """
        Optimize an equijoin operation.

        This method creates a JoinOptimizer instance and uses it to optimize the equijoin operation.
        It updates the operation cost and runs the optimized operation.
        If the LLM suggests a map transformation, it will optimize the map operation as its own step, and then go back to optimize the equijoin operation.

        Args:
            op_config (Dict[str, Any]): The configuration of the equijoin operation.
            left_name (str): The name of the left dataset.
            right_name (str): The name of the right dataset.
            left_data (List[Dict[str, Any]]): The left dataset for the join.
            right_data (List[Dict[str, Any]]): The right dataset for the join.

        Returns:
            Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], str, str]: The optimized operation configuration, the new left and right datasets, and the new left and right names.
        """
        max_iterations = 2
        new_left_name = left_name
        new_right_name = right_name
        for _ in range(max_iterations):
            join_optimizer = JoinOptimizer(
                self.config,
                op_config,
                self.console,
                self.llm_client,
                self.max_threads,
                target_recall=self.config.get("optimizer_config", {})
                .get("equijoin", {})
                .get("target_recall", 0.95),
                estimated_selectivity=self.config.get("optimizer_config", {})
                .get("equijoin", {})
                .get("estimated_selectivity", None),
                status=status,
            )
            optimized_config, cost, agent_results = join_optimizer.optimize_equijoin(
                left_data, right_data
            )
            self.operations_cost += cost
            # Update the operation config with the optimized values
            op_config.update(optimized_config)

            if not agent_results.get("optimize_map", False):
                break  # Exit the loop if no more map optimizations are necessary

            # Update the status to indicate we're optimizing a map operation
            output_key = agent_results["output_key"]
            if self.status:
                self.status.update(
                    f"Optimizing map operation for {output_key} extraction to help with the equijoin"
                )
            map_prompt = agent_results["map_prompt"]
            dataset_to_transform = (
                left_data
                if agent_results["dataset_to_transform"] == "left"
                else right_data
            )

            # Create a new step for the map operation
            map_operation = {
                "name": f"synthesized_{output_key}_extraction",
                "type": "map",
                "prompt": map_prompt,
                "model": self.config.get("default_model", "gpt-4o-mini"),
                "output": {"schema": {output_key: "string"}},
                "optimize": False,
            }

            # Optimize the map operation
            if map_operation["optimize"]:
                dataset_to_transform_sample = random.sample(
                    dataset_to_transform, self.sample_size_map.get("map")
                )
                optimized_map_operations = self._optimize_map(
                    map_operation, dataset_to_transform_sample
                )
            else:
                optimized_map_operations = [map_operation]

            new_step = {
                "name": f"synthesized_{output_key}_extraction",
                "input": (
                    left_name
                    if agent_results["dataset_to_transform"] == "left"
                    else right_name
                ),
                "operations": [mo["name"] for mo in optimized_map_operations],
            }
            if agent_results["dataset_to_transform"] == "left":
                new_left_name = new_step["name"]
            else:
                new_right_name = new_step["name"]

            for optimized_map_op in optimized_map_operations:
                self.optimized_config["operations"].append(optimized_map_op)

            self.optimized_config["pipeline"]["steps"].append(new_step)

            # Now run the optimized map operation on the entire dataset_to_transform
            for op in optimized_map_operations:
                dataset_to_transform = self._run_operation(op, dataset_to_transform)

            # Update the appropriate dataset for the next iteration
            if agent_results["dataset_to_transform"] == "left":
                left_data = dataset_to_transform
            else:
                right_data = dataset_to_transform

            if self.status:
                self.status.update(
                    f"Optimizing equijoin operation with {output_key} extraction"
                )

        # Pop off "left" and "right" from the op_config
        op_config.pop("left")
        op_config.pop("right")
        return (
            [op_config],
            {"left": left_data, "right": right_data},
            new_left_name,
            new_right_name,
        )

    def _optimize_map(
        self,
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
        is_filter: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Optimize a map operation.

        This method creates a MapOptimizer instance and uses it to optimize the map operation.

        Args:
            op_config (Dict[str, Any]): The configuration of the map operation.
            input_data (List[Dict[str, Any]]): The input data for the map operation.
            is_filter (bool, optional): If True, the operation is a filter operation. Defaults to False.

        Returns:
            List[Dict[str, Any]]: The optimized operation configuration.
        """
        map_optimizer = MapOptimizer(
            self.config,
            self.console,
            self.llm_client,
            self.max_threads,
            self._run_operation,
            timeout=self.timeout,
            is_filter=is_filter,
        )
        optimized_ops, _, cost = map_optimizer.optimize(op_config, input_data)
        self.operations_cost += cost
        return optimized_ops

    def _optimize_resolve(
        self, op_config: Dict[str, Any], input_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Optimize a resolve operation.

        This method creates a JoinOptimizer instance and uses it to optimize the resolve operation.
        It updates the operation cost and runs the optimized operation.

        Args:
            op_config (Dict[str, Any]): The configuration of the resolve operation.
            input_data (List[Dict[str, Any]]): The input data for the resolve operation.

        Returns:
            List[Dict[str, Any]]: The optimized operation configuration.
        """
        optimized_config, cost = JoinOptimizer(
            self.config,
            op_config,
            self.console,
            self.llm_client,
            self.max_threads,
            target_recall=self.config.get("optimizer_config", {})
            .get("resolve", {})
            .get("target_recall", 0.95),
        ).optimize_resolve(input_data)

        if optimized_config.get("empty", False):
            # Remove this operation from the pipeline and just return input data
            return [], input_data

        self.operations_cost += cost

        # Update the operation config with the optimized values
        op_config.update(optimized_config)

        return [op_config]

    def _run_operation(
        self,
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
        return_instance: bool = False,
        is_build: bool = False,
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

        oc_kwargs = {
            "config": op_config,
            "default_model": self.config["default_model"],
            "max_threads": self.max_threads,
            "console": self.console,
            "status": self.status,
        }
        operation_instance = operation_class(**oc_kwargs)
        if op_config["type"] == "equijoin":
            left_data = input_data["left"]
            right_data = input_data["right"]
            output_data, cost = operation_instance.execute(left_data, right_data)
        elif op_config["type"] == "filter":
            output_data, cost = operation_instance.execute(input_data, is_build)
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

        # Remove _intermediates from each operation in resolved_config
        if "operations" in resolved_config:
            for op_config in resolved_config["operations"]:
                if "_intermediates" in op_config:
                    del op_config["_intermediates"]

        with open(self.optimized_config_path, "w") as f:
            yaml.safe_dump(resolved_config, f, default_flow_style=False, width=80)
            self.console.log(
                f"[green italic]💾 Optimized config saved to {self.optimized_config_path}[/green italic]"
            )


if __name__ == "__main__":
    optimizer = Optimizer("workloads/medical/map.yaml", model="gpt-4o-mini")
    optimizer.optimize()
