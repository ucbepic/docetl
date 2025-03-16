"""
This module contains the container classes used by the DSLRunner for pipeline execution.
The containers implement a pull-based execution model where operations are lazily evaluated
only when their outputs are needed by parent nodes.
"""

import copy
import json
import math
import os
from typing import TYPE_CHECKING, Dict, List, Tuple

from rich.panel import Panel

from docetl.dataset import Dataset
from docetl.operations import get_operation
from docetl.operations.utils import flush_cache
from docetl.optimizers import JoinOptimizer, MapOptimizer, ReduceOptimizer
from docetl.optimizers.utils import CandidatePlan
from docetl.utils import smart_sample

if TYPE_CHECKING:
    from docetl.runner import DSLRunner

SUPPORTED_OPS = ["map", "resolve", "reduce", "equijoin", "filter"]
NUM_OPTIMIZER_RETRIES = 1
NUM_PLANS_TO_KEEP = 3


class OpContainer:
    """
    OpContainer implements a pull-based execution model for pipeline operations. Each container
    represents a node in the execution DAG and lazily evaluates its operation only when its
    output is requested by a parent node.

    Key features:
    - Lazy evaluation: Operations only execute when their output is needed
    - Transparent caching: Results can be cached and reused across pipeline runs
    - Cost tracking: Each operation's execution cost is tracked and aggregated

    The pull-based model means that execution flows backwards through the DAG - when the final
    node is asked for data, it recursively requests data from its children until reaching leaf
    nodes (typically scan operations that load initial datasets).
    """

    def __init__(self, name: str, runner: "DSLRunner", config: Dict, **kwargs):
        self.name = name
        self.config = config
        self.children = []
        self.parent = None
        self.is_equijoin = config.get("type") == "equijoin"
        self.runner = runner
        self.selectivity = kwargs.get("selectivity", None)
        if not self.selectivity:
            # If it's a map or resolve or gather operation, we know the selectivity is 1
            if self.config.get("type") in [
                "map",
                "parallel_map",
                "code_map",
                "resolve",
                "gather",
            ]:
                self.selectivity = 1
        self.is_optimized = False
        self.kwargs = kwargs
        self.shadow_configs = []

    def to_string(self) -> str:
        return json.dumps(self.config, indent=2)

    def add_child(self, child: "OpContainer") -> None:
        self.children.append(child)
        child.parent = self

    def optimize(self):
        """
        Optimize the operation and store optimized configurations in shadow_configs.
        This method does not modify the execution tree structure.

        We do the following:
        1. Optimize the children
        2. Run the children to get the input data for optimizing this operation
        3. Optimize this operation and store the optimized configurations in shadow_configs
        """
        # Return early if already optimized
        if self.is_optimized:
            return

        # optimize the children
        for child in self.children:
            child.optimize()

        # If operation is not eligible for optimization, return
        if (
            not self.config.get("optimize", False)
            or self.config["type"] not in SUPPORTED_OPS
        ):
            if (
                self.config.get("optimize", False)
                and self.config["type"] not in SUPPORTED_OPS
            ):
                self.runner.console.log(
                    f"[red]Operation {self.name} is not supported for optimization. Proceeding without optimizing it.[/red]"
                )
            self.is_optimized = True
            return

        # Figure out the sample size needed for this operation from the sample size map
        sample_size_needed = self.runner.optimizer.sample_size_map.get(
            self.config["type"]
        )

        # if type is equijoin, sample_size_needed may be a dictionary
        if self.config["type"] == "equijoin":
            if isinstance(sample_size_needed, dict):
                sample_size_needed = [
                    sample_size_needed["left"],
                    sample_size_needed["right"],
                ]
            else:
                sample_size_needed = [sample_size_needed, sample_size_needed]
        else:
            sample_size_needed = [sample_size_needed]

        assert len(sample_size_needed) >= len(
            self.children
        ), f"Sample size list must be a list of at least the same length as the number of children. Current sample size list: {sample_size_needed}. Current number of children: {len(self.children)}"

        # run the children to get the input data for optimizing this operation
        input_data = []
        for idx, child in enumerate(self.children):
            input_data.append(
                child.next(is_build=True, sample_size_needed=sample_size_needed[idx])[0]
            )

        # If this is a build operation, set the captured output
        self.runner.optimizer.captured_output.set_step(self.name.split("/")[0])

        # Print statistics for optimizing this operation
        sample_info = []
        if self.config["type"] == "equijoin":
            sample_info.extend(
                [
                    f"[yellow]Sample size (left): {len(input_data[0])}",
                    f"[yellow]Sample size (right): {len(input_data[1])}",
                ]
            )
        else:
            sample_info.append(f"[yellow]Sample size: {len(input_data[0])}")

        # Get optimizer config for this operation type if it exists
        optimizer_config = self.runner.config.get("optimizer_config", {}).get(
            self.config["type"], {}
        )

        panel_content = "\n".join(sample_info)
        if optimizer_config:
            panel_content += "\n\n[cyan]Optimizer Config:[/cyan]"
            for key, value in optimizer_config.items():
                panel_content += f"\n[cyan]{key}:[/cyan] {value}"

        self.runner.console.log(
            Panel.fit(
                panel_content,
                title=f"[yellow]Optimizing {self.name} (Type: {self.config['type']})",
            )
        )

        # Log the start of optimization - no status display
        self.runner.console.log(
            f"[bold blue]Running optimization for: {self.name} (Type: {self.config['type']})[/bold blue]"
        )

        # Run optimization without status display
        top_plan_objects = []

        # Run optimization
        for retry in range(
            self.runner.config.get("optimizer_config", {}).get(
                "num_retries", NUM_OPTIMIZER_RETRIES
            )
        ):
            try:
                if self.config.get("type") in ["map", "filter"]:
                    map_optimizer = MapOptimizer(
                        self.runner,
                        self.runner._run_operation,
                        is_filter=self.config["type"] == "filter",
                        num_desired_plans=self.runner.config.get(
                            "keep_num_plans", NUM_PLANS_TO_KEEP
                        ),
                    )
                    top_plan_objects, cost = map_optimizer.optimize(
                        self.config,
                        input_data[0],
                        plan_types=self.runner.config.get("optimizer_config", {})
                        .get("map", {})
                        .get(
                            "plan_types",
                            ["chunk", "proj_synthesis", "glean"],
                        ),
                    )
                    self.runner.total_cost += cost

                    # Store all plan objects in shadow_configs instead of just operations from top plan
                    self.shadow_configs = top_plan_objects

                elif self.config.get("type") == "reduce":
                    reduce_optimizer = ReduceOptimizer(
                        self.runner,
                        self.runner._run_operation,
                        num_desired_plans=self.runner.config.get(
                            "keep_num_plans", NUM_PLANS_TO_KEEP
                        ),
                    )
                    top_plan_objects, _, cost = reduce_optimizer.optimize(
                        self.config, input_data[0]
                    )
                    self.runner.total_cost += cost

                    # Store all plan objects in shadow_configs
                    self.shadow_configs = top_plan_objects

                elif self.config.get("type") == "resolve":
                    optimized_config, cost = JoinOptimizer(
                        self.runner,
                        self.config,
                        target_recall=self.runner.config.get("optimizer_config", {})
                        .get("resolve", {})
                        .get("target_recall", 0.95),
                        estimated_selectivity=self.runner.config.get(
                            "optimizer_config", {}
                        )
                        .get("resolve", {})
                        .get("estimated_selectivity", None),
                    ).optimize_resolve(input_data[0])
                    self.runner.total_cost += cost
                    # Store optimized config in shadow_configs
                    if not optimized_config.get("empty", False):
                        op_config = self.config.copy()
                        op_config.update(optimized_config)
                        self.shadow_configs = [op_config]
                elif self.config.get("type") == "equijoin":
                    op_config, new_steps, new_left_name, new_right_name = (
                        self.runner.optimizer._optimize_equijoin(
                            self.config,
                            self.kwargs["left_name"],
                            self.kwargs["right_name"],
                            input_data[0],
                            input_data[1],
                            self.runner._run_operation,
                        )
                    )
                    # Store optimized equijoin configuration and steps
                    self.shadow_configs = [
                        {
                            "op_config": op_config,
                            "new_steps": new_steps,
                            "new_left_name": new_left_name,
                            "new_right_name": new_right_name,
                        }
                    ]
                else:
                    raise ValueError(
                        f"Unsupported operation type: {self.config['type']}"
                    )

                # Log successful optimization
                if self.shadow_configs:
                    self.runner.console.log(
                        f"[green]✓ Optimization successful for {self.name}[/green]"
                    )
                else:
                    self.runner.console.log(
                        f"[yellow]⚠ No optimization plans generated for {self.name}[/yellow]"
                    )

                break  # If successful, break out of the retry loop
            except Exception as e:
                if (
                    retry
                    == self.runner.config.get("optimizer_config", {}).get(
                        "num_retries", NUM_OPTIMIZER_RETRIES
                    )
                    - 1
                ):
                    self.runner.console.log(
                        f"[red]✗ Optimization failed for {self.name}: {str(e)}[/red]"
                    )
                    raise  # If this was the last retry, re-raise the exception
                self.runner.console.log(
                    f"[yellow]⚠ Optimization attempt {retry + 1} failed with error: {e}. Retrying...[/yellow]"
                )

        # Mark as optimized
        self.is_optimized = True

        # Checkpoint the optimized operations
        self.runner.optimizer.checkpoint_optimized_ops()

    def compile_optimized_plans(self):
        """
        Generate combinations of optimized plans from all nodes in the execution tree.

        This method creates a product of different plan combinations and returns
        a list of execution trees, each with its associated cost and score.

        We apply optimizations from bottom to top (leaf nodes first), so that all child
        optimizations are applied before parent optimizations modify the tree structure.

        Returns:
            List[CandidatePlan]: A list of candidate plans with their containers, costs, and scores
        """
        # First, recursively get all plan combinations from children
        child_plan_combinations = []
        for child in self.children:
            child_plans = child.compile_optimized_plans()
            if child_plans:
                child_plan_combinations.append(child_plans)
            else:
                # If a child has no plans, create a new container with the same config
                new_child = OpContainer(
                    child.name,
                    self.runner,
                    copy.deepcopy(child.config),
                    **copy.deepcopy(child.kwargs) if child.kwargs else {},
                )
                new_child.selectivity = child.selectivity
                child_plan_combinations.append(
                    [CandidatePlan(container=new_child, cost=0.0, score=0.0)]
                )

        # If no children, return just this node
        if not child_plan_combinations:
            # Create a new container with the same config
            new_container = OpContainer(
                self.name,
                self.runner,
                copy.deepcopy(self.config),
                **copy.deepcopy(self.kwargs) if self.kwargs else {},
            )
            new_container.selectivity = self.selectivity
            return [CandidatePlan(container=new_container, cost=0.0, score=0.0)]

        # Log that we're compiling optimized plans
        self.runner.console.log(
            f"[yellow]Compiling optimized plans for {self.name}; with {len(self.shadow_configs)} shadow configs[/yellow]"
        )

        result_plans = []

        # If no shadow configs but we have child plans, create a container for each child plan combination
        if not self.shadow_configs:
            # For each child plan combination, create a new container and connect the children
            for child_plan_combo in self._generate_child_plan_combinations(
                child_plan_combinations
            ):
                # Create a new container with the current config
                new_container = OpContainer(
                    self.name,
                    self.runner,
                    copy.deepcopy(self.config),
                    **copy.deepcopy(self.kwargs) if self.kwargs else {},
                )
                new_container.selectivity = self.selectivity

                # Connect to child plans
                total_cost = 0.0
                total_score = 0.0
                for child_plan in child_plan_combo:
                    # Create a deep copy of the child container
                    child_copy = self._deep_copy_container(child_plan.container)
                    new_container.add_child(child_copy)
                    total_cost += child_plan.cost
                    total_score += child_plan.score

                result_plans.append(
                    CandidatePlan(
                        container=new_container, cost=total_cost, score=total_score
                    )
                )

            return result_plans

        # Special handling for equijoin which has a different shadow_configs structure
        if (
            self.config["type"] == "equijoin"
            and isinstance(self.shadow_configs[0], dict)
            and "op_config" in self.shadow_configs[0]
        ):
            # For equijoin, we just create one optimized plan with the optimized config
            shadow_config = self.shadow_configs[0]
            op_config = copy.deepcopy(shadow_config["op_config"])
            new_steps = shadow_config["new_steps"]
            new_left_name = shadow_config["new_left_name"]
            new_right_name = shadow_config["new_right_name"]

            # Create a new OpContainer for the optimized equijoin
            new_kwargs = copy.deepcopy(self.kwargs) if self.kwargs else {}
            new_container = OpContainer(self.name, self.runner, op_config, **new_kwargs)
            new_container.selectivity = self.selectivity

            # Set new children (scan operations for the optimized tables)
            curr_step_name = self.name.split("/")[0]
            left_child = OpContainer(
                f"{curr_step_name}/scan_{new_left_name}",
                self.runner,
                {
                    "type": "scan",
                    "name": f"scan_{new_left_name}",
                    "dataset_name": new_left_name,
                },
            )
            right_child = OpContainer(
                f"{curr_step_name}/scan_{new_right_name}",
                self.runner,
                {
                    "type": "scan",
                    "name": f"scan_{new_right_name}",
                    "dataset_name": new_right_name,
                },
            )

            new_container.add_child(left_child)
            new_container.add_child(right_child)

            # Find which child dataset changed (left or right)
            left_changed = new_left_name != self.kwargs.get("left_name")
            if left_changed:
                local_last_op_container = left_child
            else:
                local_last_op_container = right_child

            # Update the kwargs left and right names
            new_container.kwargs["left_name"] = new_left_name
            new_container.kwargs["right_name"] = new_right_name

            # Add the new steps and operations to the query plan
            for step_name, step_obj, operations in reversed(new_steps):
                # Create the step boundary
                step_boundary = StepBoundary(
                    f"{step_name}/boundary",
                    self.runner,
                    {
                        "type": "step_boundary",
                        "name": f"{step_name}/boundary",
                    },
                )

                # Point the local_last_op_container to this step boundary
                local_last_op_container.add_child(step_boundary)
                local_last_op_container = step_boundary

                # Create the operations chain
                for op in operations:
                    op_config_copy = copy.deepcopy(op)
                    op_container = OpContainer(
                        f"{step_name}/{op_config_copy['name']}",
                        self.runner,
                        op_config_copy,
                    )
                    local_last_op_container.add_child(op_container)
                    local_last_op_container = op_container

                # Add the scan operation
                scan_op = OpContainer(
                    f"{step_name}/scan_{step_obj['input']}",
                    self.runner,
                    {
                        "type": "scan",
                        "name": f"scan_{step_obj['input']}",
                        "dataset_name": step_obj["input"],
                    },
                )
                local_last_op_container.add_child(scan_op)
                local_last_op_container = scan_op

            # For equijoins, we estimate a base cost of 0 since we don't have a direct cost estimate
            # And default score to 0 as well
            result_plans.append(
                CandidatePlan(container=new_container, cost=0.0, score=0.0)
            )
            self.runner.console.log(
                f"[green]Created 1 optimized plan for equijoin {self.name}[/green]"
            )

        # Special handling for resolve which just has a different config
        elif (
            self.config["type"] == "resolve"
            and self.shadow_configs
            and isinstance(self.shadow_configs[0], dict)
        ):
            # For resolve, create one optimized plan with the updated config
            for child_plan_combo in self._generate_child_plan_combinations(
                child_plan_combinations
            ):
                optimized_config = copy.deepcopy(self.shadow_configs[0])

                # Create a new container with the optimized config
                new_kwargs = copy.deepcopy(self.kwargs) if self.kwargs else {}
                new_container = OpContainer(
                    self.name, self.runner, optimized_config, **new_kwargs
                )
                new_container.selectivity = self.selectivity

                # Connect to child plans
                total_cost = 0.0  # Base cost for resolve
                total_score = 0.0
                for child_plan in child_plan_combo:
                    # Create a deep copy of the child container to avoid shared instances
                    child_copy = self._deep_copy_container(child_plan.container)
                    new_container.add_child(child_copy)
                    total_cost += child_plan.cost
                    total_score += child_plan.score

                result_plans.append(
                    CandidatePlan(
                        container=new_container, cost=total_cost, score=total_score
                    )
                )

            self.runner.console.log(
                f"[green]Created {len(result_plans)} optimized plans for resolve {self.name}[/green]"
            )

        # Standard case for map, filter, reduce operations
        else:
            # For each plan in shadow_configs and each combination of child plans,
            # create a new optimized container and connect it to the children
            plan_count = 0

            # For map/filter/reduce, shadow_configs contains PlanResult objects
            for plan_result in self.shadow_configs:  # Limit number of plans
                for child_plan_combo in self._generate_child_plan_combinations(
                    child_plan_combinations
                ):
                    # Get the operations from the plan_result
                    optimized_ops = (
                        plan_result.ops if hasattr(plan_result, "ops") else []
                    )

                    # Skip empty plans
                    if not optimized_ops:
                        continue

                    # Create a chain of new optimized containers
                    last_optimized_container = None
                    first_optimized_container = None

                    # Get current step name
                    curr_step_name = self.name.split("/")[0]

                    # Create containers for each operation in the optimized plan
                    for idx, op in enumerate(list(reversed(optimized_ops))):
                        # Create a copy of the operation config
                        op_config = copy.deepcopy(op)
                        op_container = OpContainer(
                            f"{curr_step_name}/{op_config['name']}",
                            self.runner,
                            op_config,
                        )

                        if idx == 0:
                            # First operation in the optimized chain
                            first_optimized_container = op_container
                            last_optimized_container = op_container
                        else:
                            # Add to the chain
                            last_optimized_container.add_child(op_container)
                            last_optimized_container = op_container

                    # Connect to child plans
                    if last_optimized_container:
                        # Extract plan cost and score directly from the plan_result object
                        plan_cost = (
                            plan_result.cost if hasattr(plan_result, "cost") else 0.0
                        )
                        plan_score = (
                            plan_result.score if hasattr(plan_result, "score") else 0.0
                        )
                        total_cost = plan_cost
                        total_score = plan_score

                        for child_plan in child_plan_combo:
                            # Create a deep copy of the child container to avoid shared instances
                            child_copy = self._deep_copy_container(child_plan.container)
                            last_optimized_container.add_child(child_copy)
                            total_cost += child_plan.cost
                            total_score += child_plan.score

                        result_plans.append(
                            CandidatePlan(
                                container=first_optimized_container,
                                cost=total_cost,
                                score=total_score,
                            )
                        )
                        plan_count += 1

            if plan_count == 0:
                # If no valid plans were created, use the original container
                for child_plan_combo in self._generate_child_plan_combinations(
                    child_plan_combinations
                ):
                    # Create a new container with the original config
                    new_kwargs = copy.deepcopy(self.kwargs) if self.kwargs else {}
                    new_container = OpContainer(
                        self.name, self.runner, copy.deepcopy(self.config), **new_kwargs
                    )
                    new_container.selectivity = self.selectivity

                    # Connect to child plans
                    total_cost = 0.0
                    total_score = 0.0
                    for child_plan in child_plan_combo:
                        # Create a deep copy of the child container to avoid shared instances
                        child_copy = self._deep_copy_container(child_plan.container)
                        new_container.add_child(child_copy)
                        total_cost += child_plan.cost
                        total_score += child_plan.score

                    result_plans.append(
                        CandidatePlan(
                            container=new_container, cost=total_cost, score=total_score
                        )
                    )
                    break  # Just create one original plan

            self.runner.console.log(
                f"[green]Created {len(result_plans)} optimized plans for {self.name}[/green]"
            )

        return result_plans

    def _deep_copy_container(self, container):
        """
        Create a deep copy of a container and its child tree.
        This avoids sharing container instances between different optimization plans.

        Args:
            container (OpContainer): The container to copy

        Returns:
            OpContainer: A new container with copies of all children
        """
        # Create a new container with copied config and kwargs, preserving the class type
        if isinstance(container, StepBoundary):
            new_container = StepBoundary(
                container.name,
                self.runner,
                copy.deepcopy(container.config),
                **copy.deepcopy(container.kwargs) if container.kwargs else {},
            )
        else:
            new_container = OpContainer(
                container.name,
                self.runner,
                copy.deepcopy(container.config),
                **copy.deepcopy(container.kwargs) if container.kwargs else {},
            )

        new_container.selectivity = container.selectivity

        # Recursively copy all children
        for child in container.children:
            child_copy = self._deep_copy_container(child)
            new_container.add_child(child_copy)

        return new_container

    def _generate_child_plan_combinations(self, child_plan_combinations):
        """
        Generate combinations of child plans.

        Args:
            child_plan_combinations (List[List[CandidatePlan]]): List of child plan lists

        Yields:
            List[CandidatePlan]: Combinations of child plans
        """
        if not child_plan_combinations:
            yield []
            return

        combination_count = 0

        # Generate cartesian product of child plans
        import itertools

        for combo in itertools.product(*child_plan_combinations):
            yield combo
            combination_count += 1

    def next(
        self, is_build: bool = False, sample_size_needed: int = None
    ) -> Tuple[List[Dict], float, str]:
        """
        Execute this operation and return its results. This is the core method implementing
        the pull-based execution model.

        The execution follows these steps:
        1. Check for cached results in checkpoints
        2. If not cached, recursively request input data from child nodes
        3. Apply any configured sampling
        4. Execute the operation on the input data
        5. Cache results if checkpointing is enabled

        Returns:
            Tuple[List[Dict], float, str]: A tuple containing:
                - The operation's output data
                - Total cost of this operation and its children
                - Execution logs as a formatted string
        """
        # Track cost and logs for this operation and its children
        input_data = None
        cost = 0.0
        this_op_cost = 0.0
        curr_logs = ""
        input_len = None

        # If this is a build operation, check the sample cache first
        if is_build:
            cache_key = self.name
            if cache_key in self.runner.optimizer.sample_cache:
                cached_data, cached_sample_size = self.runner.optimizer.sample_cache[
                    cache_key
                ]
                # If we have enough samples cached, use them
                if not sample_size_needed or cached_sample_size >= sample_size_needed:
                    curr_logs += f"[green]✓[/green] Using cached {self.name} (sample size: {cached_sample_size})\n"
                    # Sample the cached data if needed
                    if sample_size_needed:
                        cached_data = smart_sample(cached_data, sample_size_needed)

                    return cached_data, 0, curr_logs

        # Try to load from checkpoint if available
        if not is_build:
            attempted_input_data = self.runner._load_from_checkpoint_if_exists(
                self.name.split("/")[0], self.name.split("/")[-1]
            )
            if attempted_input_data is not None:
                curr_logs += f"[green]✓[/green] Using cached {self.name}\n"
                return attempted_input_data, 0, curr_logs

        # If there's a selectivity estimate, we need to take a sample of size sample_size_needed / selectivity
        if self.selectivity and sample_size_needed:
            input_sample_size_needed = int(
                math.ceil(sample_size_needed / self.selectivity)
            )
        else:
            input_sample_size_needed = sample_size_needed

        # Clear any existing checkpoint before running
        if self.runner.intermediate_dir:
            checkpoint_path = os.path.join(
                self.runner.intermediate_dir,
                self.name.split("/")[0],
                f"{self.name.split('/')[-1]}.json",
            )
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

        # Handle equijoin operations which have two input streams
        if self.is_equijoin:
            assert (
                len(self.children) == 2
            ), "Equijoin should have left and right children"
            left_data, left_cost, left_logs = self.children[0].next(
                is_build, input_sample_size_needed
            )
            right_data, right_cost, right_logs = self.children[1].next(
                is_build, input_sample_size_needed
            )
            cost += left_cost + right_cost
            curr_logs += left_logs + right_logs
            input_len = max(len(left_data), len(right_data))
            input_data = {"left_data": left_data, "right_data": right_data}
        # Handle standard operations with single input
        elif len(self.children) > 0:
            input_data, input_cost, input_logs = self.children[0].next(
                is_build, input_sample_size_needed
            )
            cost += input_cost
            curr_logs += input_logs
            input_len = len(input_data)

        # Apply sampling if configured
        if input_data and "sample" in self.config and not is_build:
            input_data = input_data[: self.config["sample"]]

        # Execute the operation
        cost_before_execution = self.runner.total_cost

        # Skip using status display in build mode to avoid LiveError with nested status displays
        if is_build:
            # Execute operation without status display when in build mode
            output_data = self.runner._run_operation(
                self.config, input_data, is_build=is_build
            )

            # Track costs and log execution
            this_op_cost = self.runner.total_cost - cost_before_execution
            cost += this_op_cost

            build_indicator = "[yellow](build)[/yellow] "
            curr_logs += f"[green]✓[/green] {build_indicator}{self.name} (Cost: [green]${this_op_cost:.2f}[/green])\n"
            self.runner.console.log(
                f"[green]✓[/green] {build_indicator}{self.name} (Cost: [green]${this_op_cost:.2f}[/green])"
            )
        else:
            # Use status display for non-build operations
            with self.runner.console.status(f"Running {self.name}") as status:
                self.runner.status = status

                # Execute operation with appropriate inputs
                output_data = self.runner._run_operation(
                    self.config, input_data, is_build=is_build
                )

                # Track costs and log execution
                this_op_cost = self.runner.total_cost - cost_before_execution
                cost += this_op_cost

                curr_logs += f"[green]✓[/green] {self.name} (Cost: [green]${this_op_cost:.2f}[/green])\n"
                self.runner.console.log(
                    f"[green]✓[/green] {self.name} (Cost: [green]${this_op_cost:.2f}[/green])"
                )

        # Save selectivity estimate
        output_size = len(output_data)
        self.selectivity = output_size / input_len if input_len else 1

        # Cache the results if this is a build operation
        if is_build:
            self.runner.optimizer.sample_cache[self.name] = (
                output_data,
                len(output_data),
            )

        # Truncate output data to the sample size needed
        if sample_size_needed:
            output_data = smart_sample(output_data, sample_size_needed)

        # Save checkpoint if enabled
        if (
            not is_build
            and self.runner.intermediate_dir
            and self.name.split("/")[1]
            in self.runner.step_op_hashes[self.name.split("/")[0]]
        ):
            self.runner._save_checkpoint(
                self.name.split("/")[0], self.name.split("/")[-1], output_data
            )

        return output_data, cost, curr_logs

    def syntax_check(self) -> str:
        operation = self.config["name"]
        operation_type = self.config["type"]

        operation_class = get_operation(operation_type)
        obj = operation_class(
            self.runner,
            self.config,
            self.runner.default_model,
            self.runner.max_threads,
            self.runner.console,
            self.runner.status,
        )

        # Do syntax check
        obj.syntax_check()

        return f"[green]✓[/green] Operation '{operation}' ({operation_type})"

    def estimate_selectivities(self):
        """
        Estimates and stores selectivities for operations in the execution plan without optimizing them.
        This method runs each operation on sample data to determine its selectivity (output size / input size ratio).
        Sample sizes are determined from the optimizer's sample_size_map.
        """
        # Skip if selectivity is already known
        if self.selectivity is not None and self.config["type"] not in [
            "equijoin",
            "resolve",
        ]:
            return

        # Figure out the sample size needed for this operation from the sample size map
        sample_size_needed = self.runner.optimizer.sample_size_map.get(
            self.config["type"]
        )

        # For equijoin operations, sample_size_needed may be a dictionary
        if self.config["type"] == "equijoin":
            if isinstance(sample_size_needed, dict):
                sample_size_needed = [
                    sample_size_needed["left"],
                    sample_size_needed["right"],
                ]
            else:
                sample_size_needed = [sample_size_needed, sample_size_needed]
        else:
            sample_size_needed = [sample_size_needed]

        self.runner.console.log(
            f"[cyan]▶ Starting selectivity estimation:[/cyan] {self.name} [dim]({self.config['type']})[/dim] | Sample size: [yellow]{sample_size_needed}[/yellow] | Children: [yellow]{len(self.children)}[/yellow]"
        )

        # For leaf nodes (operations with no children)
        if len(self.children) == 0:
            # Selectivity is 1 for scan operations
            self.selectivity = 1
            self.runner.console.log(
                f"[green]✓ Estimated selectivity:[/green] {self.name} [dim]({self.config['type']})[/dim] | Selectivity: [bold green]1.0000[/bold green] [dim](scan operation)[/dim]"
            )
            return

        # First estimate selectivity for children recursively
        for idx, child in enumerate(self.children):
            child.estimate_selectivities()

        # Special handling for resolve operations
        if self.config["type"] == "resolve":
            # Get input data for optimization directly from children rather than using next()
            input_data, _, _ = self.children[0].next(
                is_build=True, sample_size_needed=sample_size_needed[0]
            )
            input_len = len(input_data)

            # Use JoinOptimizer to get optimized config
            join_optimizer = JoinOptimizer(
                self.runner,
                self.config,
                target_recall=self.runner.config.get("optimizer_config", {})
                .get("resolve", {})
                .get("target_recall", 0.95),
                estimated_selectivity=self.runner.config.get("optimizer_config", {})
                .get("resolve", {})
                .get("estimated_selectivity", None),
            )

            self.runner.console.log(
                f"[blue]  ↳ Using JoinOptimizer for resolve operation:[/blue] {self.name}"
            )

            # Get optimized config
            optimized_config, cost = join_optimizer.optimize_resolve(input_data)
            self.runner.total_cost += cost

            # Update the current config with optimized version
            self.config.update(optimized_config)

            # Run the operation directly using _run_operation instead of next()
            output_data = self.runner._run_operation(
                self.config, input_data, is_build=True
            )

            output_size = len(output_data)

            # Calculate selectivity
            self.selectivity = output_size / input_len if input_len else 1

            self.runner.console.log(
                f"[green]✓ Estimated selectivity:[/green] {self.name} [dim]({self.config['type']})[/dim] | Selectivity: [bold green]{self.selectivity:.4f}[/bold green] | Input: [yellow]{input_len}[/yellow] | Output: [yellow]{output_size}[/yellow]"
            )

        # Special handling for equijoin operations
        elif self.config["type"] == "equijoin":
            # Get input data for optimization
            left_data, _, _ = self.children[0].next(
                is_build=True, sample_size_needed=sample_size_needed[0]
            )
            right_data, _, _ = self.children[1].next(
                is_build=True, sample_size_needed=sample_size_needed[1]
            )
            input_len = max(len(left_data), len(right_data))

            self.runner.console.log(
                f"[blue]  ↳ Optimizing equijoin operation:[/blue] {self.name}"
            )

            # Use the optimizer to get optimized equijoin config
            op_config, new_steps, new_left_name, new_right_name = (
                self.runner.optimizer._optimize_equijoin(
                    self.config,
                    self.kwargs.get(
                        "left_name", self.children[0].config.get("dataset_name")
                    ),
                    self.kwargs.get(
                        "right_name", self.children[1].config.get("dataset_name")
                    ),
                    left_data,
                    right_data,
                    self.runner._run_operation,
                )
            )

            # Update the current config with optimized version
            self.config = op_config

            # Update child configurations
            curr_step_name = self.name.split("/")[0]
            self.children[0].config = {
                "type": "scan",
                "name": f"scan_{new_left_name}",
                "dataset_name": new_left_name,
            }
            self.children[0].name = f"{curr_step_name}/scan_{new_left_name}"

            self.children[1].config = {
                "type": "scan",
                "name": f"scan_{new_right_name}",
                "dataset_name": new_right_name,
            }
            self.children[1].name = f"{curr_step_name}/scan_{new_right_name}"

            # Update the op map
            self.runner.op_container_map[f"{curr_step_name}/scan_{new_left_name}"] = (
                self.children[0]
            )
            self.runner.op_container_map[f"{curr_step_name}/scan_{new_right_name}"] = (
                self.children[1]
            )

            # Update kwargs
            if hasattr(self, "kwargs"):
                self.kwargs["left_name"] = new_left_name
                self.kwargs["right_name"] = new_right_name
            else:
                self.kwargs = {"left_name": new_left_name, "right_name": new_right_name}

            # Run the operation directly using _run_operation instead of next()
            input_data = {"left_data": left_data, "right_data": right_data}
            output_data = self.runner._run_operation(
                self.config, input_data, is_build=True
            )

            output_size = len(output_data)

            # Calculate selectivity
            self.selectivity = output_size / input_len if input_len else 1

            self.runner.console.log(
                f"[green]✓ Estimated selectivity:[/green] {self.name} [dim]({self.config['type']})[/dim] | Selectivity: [bold green]{self.selectivity:.4f}[/bold green] | Input: [yellow]{input_len}[/yellow] | Output: [yellow]{output_size}[/yellow]"
            )

        # For standard operations with single input
        else:
            # Get input data size before running this operation
            input_data, _, _ = self.children[0].next(
                is_build=True, sample_size_needed=sample_size_needed[0]
            )
            input_len = len(input_data)

            # Run the operation directly using _run_operation instead of next()
            output_data = self.runner._run_operation(
                self.config, input_data, is_build=True
            )

            # Calculate and store selectivity
            output_size = len(output_data)
            self.selectivity = output_size / input_len if input_len else 1

            # Log the estimated selectivity
            self.runner.console.log(
                f"[green]✓ Estimated selectivity:[/green] {self.name} [dim]({self.config['type']})[/dim] | Selectivity: [bold green]{self.selectivity:.4f}[/bold green] | Input: [yellow]{input_len}[/yellow] | Output: [yellow]{output_size}[/yellow]"
            )


class StepBoundary(OpContainer):
    def next(
        self, is_build: bool = False, sample_size_needed: int = None
    ) -> Tuple[List[Dict], float, str]:

        output_data, step_cost, step_logs = self.children[0].next(
            is_build, sample_size_needed
        )

        # Print step logs only if not building
        self.runner.datasets[self.name.split("/")[0]] = Dataset(
            self, "memory", output_data
        )
        if not is_build:
            flush_cache(self.runner.console)
            self.runner.console.log(
                Panel.fit(
                    step_logs
                    + f"Step [cyan]{self.name}[/cyan] completed. Cost: [green]${step_cost:.2f}[/green]",
                    title=f"[bold blue]Step Execution: {self.name}[/bold blue]",
                )
            )

        return output_data, 0, ""

    def estimate_selectivities(self):
        """
        StepBoundary version of estimate_selectivities that acts as a no-op pass-through.
        StepBoundaries always have a selectivity of 1.0 as they don't transform the data.
        """
        # Skip if selectivity is already known
        if self.selectivity is not None:
            return

        # StepBoundary always has selectivity of 1 (passthrough)
        self.selectivity = 1.0

        self.runner.console.log(
            f"[cyan]▶ Starting selectivity estimation:[/cyan] {self.name} [dim](step_boundary)[/dim] | Children: [yellow]{len(self.children)}[/yellow]"
        )

        # If we have a child, estimate its selectivity
        if len(self.children) > 0:
            # Get the child operation
            child = self.children[0]
            child_type = child.config["type"]

            # Calculate appropriate sample size based on child's operation type
            sample_size_needed = self.runner.optimizer.sample_size_map.get(child_type)

            # Log information about the step boundary child
            self.runner.console.log(
                f"[blue]  ↳ Delegating to child:[/blue] {child.name} [dim]({child_type})[/dim] | Sample size: [yellow]{sample_size_needed}[/yellow]"
            )

            # Recursively estimate selectivity for the child
            child.estimate_selectivities()

        # Log the boundary's selectivity (always 1.0)
        self.runner.console.log(
            f"[green]✓ Estimated selectivity:[/green] {self.name} [dim](step_boundary)[/dim] | Selectivity: [bold green]{self.selectivity:.4f}[/bold green] [dim](pass-through)[/dim]"
        )

    def syntax_check(self) -> str:
        return ""
