"""
The Optimizer module implements a pipeline optimization system that works with DocETL's pull-based execution model.
It analyzes operations marked for optimization and rewrites them into more efficient sub-pipelines while preserving
the lazy evaluation semantics of the container system.

The architecture follows these key principles:
- Integration with the container-based lazy evaluation model
- Specialized optimizers for different operation types (map, reduce, join)
- Sample-based optimization to handle large datasets efficiently
- Cost tracking and caching of intermediate results
"""

import copy
import hashlib
import os
import random
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import yaml
from rich.panel import Panel
from rich.traceback import install

from docetl.containers import OpContainer, StepBoundary
from docetl.operations.utils import flush_cache
from docetl.optimizers.join_optimizer import JoinOptimizer
from docetl.optimizers.map_optimizer import MapOptimizer
from docetl.optimizers.reduce_optimizer import ReduceOptimizer
from docetl.optimizers.utils import LLMClient
from docetl.utils import CapturedOutput

if TYPE_CHECKING:
    from docetl.runner import DSLRunner

install(show_locals=False)

SAMPLE_SIZE_MAP = {
    "reduce": 40,
    "map": 5,
    "resolve": 100,
    "equijoin": 100,
    "filter": 5,
    "split": 10,
    "gather": 10,
    "unnest": 10,
}


class Optimizer:
    """
    Orchestrates the optimization of a DocETL pipeline by analyzing and potentially rewriting
    operations marked for optimization. Works with the runner's pull-based execution model
    to maintain lazy evaluation while improving pipeline efficiency.
    """

    def __init__(
        self,
        runner: "DSLRunner",
        rewrite_agent_model: str = "gpt-4o",
        judge_agent_model: str = "gpt-4o-mini",
        litellm_kwargs: Dict[str, Any] = {},
        resume: bool = False,
        timeout: int = 60,
    ):
        """
        Initialize the optimizer with a runner instance and configuration.
        Sets up optimization parameters, caching, and cost tracking.

        Args:
            yaml_file (str): Path to the YAML configuration file.
            model (str): The name of the language model to use. Defaults to "gpt-4o".
            resume (bool): Whether to resume optimization from a previous run. Defaults to False.
            timeout (int): Timeout in seconds for operations. Defaults to 60.

        Attributes:
            config (Dict): Stores the loaded configuration from the YAML file.
            console (Console): Rich console for formatted output.
            max_threads (int): Maximum number of threads for parallel processing.
            base_name (str): Base name used for file paths.
            yaml_file_suffix (str): Suffix for YAML configuration files.
            runner (DSLRunner): The DSL runner instance.
            status: Status tracking for the runner.
            optimized_config (Dict): A copy of the original config to be optimized.
            llm_client (LLMClient): Client for interacting with the language model.
            timeout (int): Timeout for operations in seconds.
            resume (bool): Whether to resume from previous optimization.
            captured_output (CapturedOutput): Captures output during optimization.
            sample_cache (Dict): Maps operation names to tuples of (output_data, sample_size).
            optimized_ops_path (str): Path to store optimized operations.
            sample_size_map (Dict): Maps operation types to sample sizes.

        The method also calls print_optimizer_config() to display the initial configuration.
        """
        self.config = runner.config
        self.console = runner.console
        self.max_threads = runner.max_threads

        self.base_name = runner.base_name
        self.yaml_file_suffix = runner.yaml_file_suffix
        self.runner = runner
        self.status = runner.status

        self.optimized_config = copy.deepcopy(self.config)

        # Get the rate limits from the optimizer config
        rate_limits = self.config.get("optimizer_config", {}).get("rate_limits", {})

        self.llm_client = LLMClient(
            runner,
            rewrite_agent_model,
            judge_agent_model,
            rate_limits,
            **litellm_kwargs,
        )
        self.timeout = timeout
        self.resume = resume
        self.captured_output = CapturedOutput()

        # Add sample cache for build operations
        self.sample_cache = {}  # Maps operation names to (output_data, sample_size)

        home_dir = os.environ.get("DOCETL_HOME_DIR", os.path.expanduser("~"))
        cache_dir = os.path.join(home_dir, f".docetl/cache/{runner.yaml_file_suffix}")
        os.makedirs(cache_dir, exist_ok=True)

        # Hash the config to create a unique identifier
        config_hash = hashlib.sha256(str(self.config).encode()).hexdigest()
        self.optimized_ops_path = f"{cache_dir}/{config_hash}.yaml"

        # Update sample size map
        self.sample_size_map = SAMPLE_SIZE_MAP
        if self.config.get("optimizer_config", {}).get("sample_sizes", {}):
            self.sample_size_map.update(self.config["optimizer_config"]["sample_sizes"])

        if not self.runner._from_df_accessors:
            self.print_optimizer_config()

        self.default_model = self.config.get("default_model", "gpt-4o-mini")
        self.model_choices = self.config.get("optimizer_config", {}).get(
            "model_choices", [self.default_model]
        )

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
        self.console.log(
            Panel.fit(
                "[bold cyan]Optimizer Configuration[/bold cyan]\n"
                f"[yellow]Sample Size:[/yellow] {self.sample_size_map}\n"
                f"[yellow]Max Threads:[/yellow] {self.max_threads}\n"
                f"[yellow]Rewrite Agent Model:[/yellow] {self.llm_client.rewrite_agent_model}\n"
                f"[yellow]Judge Agent Model:[/yellow] {self.llm_client.judge_agent_model}\n"
                f"[yellow]Rate Limits:[/yellow] {self.config.get('optimizer_config', {}).get('rate_limits', {})}\n",
                title="Optimizer Configuration",
            )
        )

    def _insert_empty_resolve_operations(self):
        """
        Determines whether to insert resolve operations in the pipeline.

        For each reduce operation in the tree, checks if it has any map operation as a descendant
        without a resolve operation in between. If found, inserts an empty resolve operation
        right after the reduce operation.

        The method modifies the operation container tree in-place.

        Returns:
            None
        """
        if not self.runner.last_op_container:
            return

        def find_map_without_resolve(container, visited=None):
            """Helper to find first map descendant without a resolve operation in between."""
            if visited is None:
                visited = set()

            if container.name in visited:
                return None
            visited.add(container.name)

            if not container.children:
                return None

            for child in container.children:
                if child.config["type"] == "map":
                    return child
                if child.config["type"] == "resolve":
                    continue
                map_desc = find_map_without_resolve(child, visited)
                if map_desc:
                    return map_desc
            return None

        # Walk down the operation container tree
        containers_to_check = [self.runner.last_op_container]
        while containers_to_check:
            current = containers_to_check.pop(0)

            # Skip if this is a boundary or has no children
            if isinstance(current, StepBoundary) or not current.children:
                containers_to_check.extend(current.children)
                continue

            # Get the step name from the container's name
            step_name = current.name.split("/")[0]

            # Check if current container is a reduce operation
            if current.config["type"] == "reduce" and current.config.get(
                "synthesize_resolve", True
            ):
                reduce_key = current.config.get("reduce_key", "_all")
                if isinstance(reduce_key, str):
                    reduce_key = [reduce_key]

                if "_all" not in reduce_key:
                    # Find map descendant without resolve
                    map_desc = find_map_without_resolve(current)
                    if map_desc:
                        # Synthesize an empty resolver
                        self.console.log(
                            "[yellow]Synthesizing empty resolver operation:[/yellow]"
                        )
                        self.console.log(
                            f"  • [cyan]Reduce operation:[/cyan] [bold]{current.name}[/bold]"
                        )
                        self.console.log(
                            f"  • [cyan]Step:[/cyan] [bold]{step_name}[/bold]"
                        )

                        # Create new resolve operation config
                        new_resolve_name = (
                            f"synthesized_resolve_{len(self.config['operations'])}"
                        )
                        new_resolve_config = {
                            "name": new_resolve_name,
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
                                "map_prompt": map_desc.config.get("prompt"),
                                "reduce_key": reduce_key,
                            },
                        }

                        # Add to operations list
                        self.config["operations"].append(new_resolve_config)

                        # Create new resolve container
                        new_resolve_container = OpContainer(
                            f"{step_name}/{new_resolve_name}",
                            self.runner,
                            new_resolve_config,
                        )

                        # Insert the new container between reduce and its children
                        new_resolve_container.children = current.children
                        for child in new_resolve_container.children:
                            child.parent = new_resolve_container
                        current.children = [new_resolve_container]
                        new_resolve_container.parent = current

                        # Add to container map
                        self.runner.op_container_map[
                            f"{step_name}/{new_resolve_name}"
                        ] = new_resolve_container

                        # Add children to the queue
                        containers_to_check.extend(new_resolve_container.children)

    def _add_map_prompts_to_reduce_operations(self):
        """
        Add relevant map prompts to reduce operations based on their reduce keys.

        This method walks the operation container tree to find map operations and their
        output schemas, then associates those with reduce operations that use those keys.
        When a reduce operation is found, it looks through its descendants to find the
        relevant map operations and adds their prompts.

        The method modifies the operation container tree in-place.
        """
        if not self.runner.last_op_container:
            return

        def find_map_prompts_for_keys(container, keys, visited=None):
            """Helper to find map prompts for given keys in the container's descendants."""
            if visited is None:
                visited = set()

            if container.name in visited:
                return []
            visited.add(container.name)

            prompts = []
            if container.config["type"] == "map":
                output_schema = container.config.get("output", {}).get("schema", {})
                if any(key in output_schema for key in keys):
                    prompts.append(container.config.get("prompt", ""))

            for child in container.children:
                prompts.extend(find_map_prompts_for_keys(child, keys, visited))

            return prompts

        # Walk down the operation container tree
        containers_to_check = [self.runner.last_op_container]
        while containers_to_check:
            current = containers_to_check.pop(0)

            # Skip if this is a boundary or has no children
            if isinstance(current, StepBoundary) or not current.children:
                containers_to_check.extend(current.children)
                continue

            # If this is a reduce operation, find relevant map prompts
            if current.config["type"] == "reduce":
                reduce_keys = current.config.get("reduce_key", [])
                if isinstance(reduce_keys, str):
                    reduce_keys = [reduce_keys]

                # Find map prompts in descendants
                relevant_prompts = find_map_prompts_for_keys(current, reduce_keys)

                if relevant_prompts:
                    current.config["_intermediates"] = current.config.get(
                        "_intermediates", {}
                    )
                    current.config["_intermediates"]["last_map_prompt"] = (
                        relevant_prompts[-1]
                    )

            # Add children to the queue
            containers_to_check.extend(current.children)

    def should_optimize(
        self, step_name: str, op_name: str
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], float]:
        """
        Analyzes whether an operation should be optimized by running it on a sample of input data
        and evaluating potential optimizations. Returns the optimization suggestion and relevant data.
        """
        self.console.rule("[bold cyan]Beginning Pipeline Assessment[/bold cyan]")

        self._insert_empty_resolve_operations()

        node_of_interest = self.runner.op_container_map[f"{step_name}/{op_name}"]

        # Run the node_of_interest's children
        input_data = []
        for child in node_of_interest.children:
            input_data.append(
                child.next(
                    is_build=True,
                    sample_size_needed=SAMPLE_SIZE_MAP.get(child.config["type"]),
                )[0]
            )

        # Set the step
        self.captured_output.set_step(step_name)

        # Determine whether we should optimize the node_of_interest
        if (
            node_of_interest.config.get("type") == "map"
            or node_of_interest.config.get("type") == "filter"
        ):
            # Create instance of map optimizer
            map_optimizer = MapOptimizer(
                self.runner,
                self.runner._run_operation,
                is_filter=node_of_interest.config.get("type") == "filter",
            )
            should_optimize_output, input_data, output_data = (
                map_optimizer.should_optimize(node_of_interest.config, input_data[0])
            )
        elif node_of_interest.config.get("type") == "reduce":
            reduce_optimizer = ReduceOptimizer(
                self.runner,
                self.runner._run_operation,
            )
            should_optimize_output, input_data, output_data = (
                reduce_optimizer.should_optimize(node_of_interest.config, input_data[0])
            )
        elif node_of_interest.config.get("type") == "resolve":
            resolve_optimizer = JoinOptimizer(
                self.runner,
                node_of_interest.config,
                target_recall=self.config.get("optimizer_config", {})
                .get("resolve", {})
                .get("target_recall", 0.95),
            )
            _, should_optimize_output = resolve_optimizer.should_optimize(input_data[0])

            # if should_optimize_output is empty, then we should move to the reduce operation
            if should_optimize_output == "":
                return "", [], [], 0.0
        else:
            return "", [], [], 0.0

        # Return the string and operation cost
        return (
            should_optimize_output,
            input_data,
            output_data,
            self.runner.total_cost + self.llm_client.total_cost,
        )

    def optimize(self) -> float:
        """
        Optimizes the entire pipeline by walking the operation DAG and applying
        operation-specific optimizers where marked. Returns the total optimization cost.
        """

        self.console.rule("[bold cyan]Estimating Selectivities[/bold cyan]")
        # Estimate selectivities for all operations
        self.runner.last_op_container.estimate_selectivities()

        self.console.rule("[bold cyan]Beginning Pipeline Rewrites[/bold cyan]")

        # If self.resume is True and there's a checkpoint, load it
        if self.resume:
            if os.path.exists(self.optimized_ops_path):
                # Load the yaml and change the runner with it
                with open(self.optimized_ops_path, "r") as f:
                    partial_optimized_config = yaml.safe_load(f)
                    self.console.log(
                        "[yellow]Loading partially optimized pipeline from checkpoint...[/yellow]"
                    )
                    self.runner._build_operation_graph(partial_optimized_config)
            else:
                self.console.log(
                    "[yellow]No checkpoint found, starting optimization from scratch...[/yellow]"
                )

        else:
            self._insert_empty_resolve_operations()

        # Start with the last operation container and visit each child
        self.runner.last_op_container.optimize()

        flush_cache(self.console)

        # Compile the optimized operations
        candidate_plans = self.runner.last_op_container.compile_optimized_plans()

        # Sort candidate plans by score
        candidate_plans.sort(key=lambda x: x.score, reverse=True)

        # Print all query plans with their costs
        for i, cp in enumerate(candidate_plans):
            cost = cp.cost
            score = cp.score
            container = cp.container
            title = "Optimized Query Plan" if i == 0 else "Candidate Plan"
            self.console.rule(
                f"[bold cyan]{title} ([green]${cost:.8f}[/green], score: [blue]{score:.4f}[/blue])[/bold cyan]"
            )
            self.runner.print_query_plan(container)

        # Set the best plan to highest scoring plan
        self.runner.last_op_container = candidate_plans[0].container

        return self.llm_client.total_cost, candidate_plans

    def _optimize_equijoin(
        self,
        op_config: Dict[str, Any],
        left_name: str,
        right_name: str,
        left_data: List[Dict[str, Any]],
        right_data: List[Dict[str, Any]],
        run_operation: Callable[
            [Dict[str, Any], List[Dict[str, Any]]], List[Dict[str, Any]]
        ],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], str, str]:
        """
        Optimizes an equijoin operation by analyzing join conditions and potentially inserting
        map operations to improve join efficiency. Returns the optimized configuration and updated data.
        """
        max_iterations = 2
        new_left_name = left_name
        new_right_name = right_name
        new_steps = []
        for _ in range(max_iterations):
            join_optimizer = JoinOptimizer(
                self.runner,
                op_config,
                target_recall=self.runner.config.get("optimizer_config", {})
                .get("equijoin", {})
                .get("target_recall", 0.95),
                estimated_selectivity=self.runner.config.get("optimizer_config", {})
                .get("equijoin", {})
                .get("estimated_selectivity", None),
            )
            optimized_config, cost, agent_results = join_optimizer.optimize_equijoin(
                left_data, right_data
            )
            self.runner.total_cost += cost
            # Update the operation config with the optimized values
            op_config.update(optimized_config)

            if not agent_results.get("optimize_map", False):
                break  # Exit the loop if no more map optimizations are necessary

            # Update the status to indicate we're optimizing a map operation
            output_key = agent_results["output_key"]
            if self.runner.status:
                self.runner.status.update(
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
                "model": self.default_model,
                "output": {"schema": {output_key: "string"}},
                "optimize": False,
            }

            # Optimize the map operation
            if map_operation["optimize"]:
                dataset_to_transform_sample = (
                    random.sample(dataset_to_transform, self.sample_size_map.get("map"))
                    if self.config.get("optimizer_config", {}).get(
                        "random_sample", False
                    )
                    else dataset_to_transform[: self.sample_size_map.get("map")]
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

            new_steps.append((new_step["name"], new_step, optimized_map_operations))

            # Now run the optimized map operation on the entire dataset_to_transform
            for op in optimized_map_operations:
                dataset_to_transform = run_operation(op, dataset_to_transform)

            # Update the appropriate dataset for the next iteration
            if agent_results["dataset_to_transform"] == "left":
                left_data = dataset_to_transform
            else:
                right_data = dataset_to_transform

            if self.runner.status:
                self.runner.status.update(
                    f"Optimizing equijoin operation with {output_key} extraction"
                )

        return op_config, new_steps, new_left_name, new_right_name

    def checkpoint_optimized_ops(self) -> None:
        """
        Generates the clean config and saves it to the self.optimized_ops_path
        This is used to resume optimization from a previous run
        """
        clean_config = self.clean_optimized_config(self.runner.last_op_container)
        with open(self.optimized_ops_path, "w") as f:
            yaml.safe_dump(clean_config, f, default_flow_style=False, width=80)

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

    def clean_optimized_config(self, container: Optional[OpContainer] = None) -> Dict:
        """
        Creates a clean YAML configuration from the optimized operation containers,
        removing internal fields and organizing operations into proper pipeline steps.
        """
        if not container:
            return self.config

        # Create a clean copy of the config
        datasets = {}
        for dataset_name, dataset_config in self.config.get("datasets", {}).items():
            if dataset_config["type"] == "memory":
                dataset_config_copy = copy.deepcopy(dataset_config)
                dataset_config_copy["path"] = "in-memory data"
                datasets[dataset_name] = dataset_config_copy
            else:
                datasets[dataset_name] = dataset_config

        clean_config = {
            "datasets": datasets,
            "operations": [],
            "pipeline": self.runner.config.get(
                "pipeline", {}
            ).copy(),  # Copy entire pipeline config
        }

        # Reset steps to regenerate
        clean_config["pipeline"]["steps"] = []

        # Keep track of operations we've seen to avoid duplicates
        seen_operations = set()

        # Map to store step names to their input datasets
        step_inputs = {}

        def clean_operation(op_container: OpContainer) -> Dict:
            """Remove internal fields from operation config"""
            op_config = op_container.config
            clean_op = copy.deepcopy(op_config)

            clean_op.pop("_intermediates", None)

            # If op has already been optimized, remove the recursively_optimize and optimize fields
            if op_container.is_optimized:
                for field in ["recursively_optimize", "optimize"]:
                    clean_op.pop(field, None)

            return clean_op

        def process_container(container, current_step=None):
            """Process an operation container and its dependencies"""
            # Skip step boundaries
            if (
                isinstance(container, StepBoundary)
                or container.config.get("type") == "step_boundary"
            ):
                if container.children:
                    return process_container(container.children[0], current_step)
                return None, None

            # Get step name from container name
            step_name = container.name.split("/")[0]

            # If this is a new step, create it
            if not current_step or current_step["name"] != step_name:
                current_step = {"name": step_name, "operations": []}
                clean_config["pipeline"]["steps"].insert(0, current_step)

            # Record input dataset if this is a scan operation
            if container.config["type"] == "scan":
                dataset_name = container.config["dataset_name"]
                step_inputs[step_name] = dataset_name

                if container.children:
                    return process_container(container.children[0], current_step)
                return None, current_step

            # Handle equijoin operations
            if container.is_equijoin:
                # Add operation to list if not seen
                if container.name not in seen_operations:
                    op_config = clean_operation(container)
                    clean_config["operations"].append(op_config)
                    seen_operations.add(container.name)

                # Add to step operations with left and right inputs
                current_step["operations"].insert(
                    0,
                    {
                        container.config["name"]: {
                            "left": container.kwargs["left_name"],
                            "right": container.kwargs["right_name"],
                        }
                    },
                )

                # Process both children
                if container.children:
                    process_container(container.children[0], current_step)
                    process_container(container.children[1], current_step)
            else:
                # Add operation to list if not seen
                if container.name not in seen_operations:
                    op_config = clean_operation(container)
                    clean_config["operations"].append(op_config)
                    seen_operations.add(container.name)

                # Add to step operations
                current_step["operations"].insert(0, container.config["name"])

                # Process children
                if container.children:
                    for child in container.children:
                        process_container(child, current_step)

            return container, current_step

        # Start processing from the last container
        process_container(container)

        # Add inputs to steps based on the step_inputs map
        for step in clean_config["pipeline"]["steps"]:
            first_op = step["operations"][0]
            step_name = step["name"]

            if isinstance(first_op, dict):  # This is an equijoin
                continue  # Equijoin steps don't need an input field
            elif len(step["operations"]) > 0 and step_name in step_inputs:
                step["input"] = step_inputs[step_name]

        # Preserve all other config key-value pairs from original config
        for key, value in self.config.items():
            if key not in ["datasets", "operations", "pipeline"]:
                clean_config[key] = value

        return clean_config

    def save_optimized_config(self, optimized_config_path: str):
        """
        Saves the optimized configuration to a YAML file after resolving all references
        and cleaning up internal optimization artifacts.
        """
        resolved_config = self.clean_optimized_config(self.runner.last_op_container)

        with open(optimized_config_path, "w") as f:
            yaml.safe_dump(resolved_config, f, default_flow_style=False, width=80)
            self.console.log(
                f"[green italic]💾 Optimized config saved to {optimized_config_path}[/green italic]"
            )

    def save_candidate_plan(self, candidate_plan, plan_path: str):
        """
        Saves a candidate plan to a YAML file.
        """
        # Get the container from the candidate plan
        container = candidate_plan.container

        # Get the clean optimized config
        resolved_config = self.clean_optimized_config(container)

        # Save the resolved config
        with open(plan_path, "w") as f:
            yaml.safe_dump(resolved_config, f, default_flow_style=False, width=80)

            self.console.log(
                f"[green italic]💾 Candidate plan saved to {plan_path}[/green italic]"
            )
