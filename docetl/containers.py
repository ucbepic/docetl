"""
This module contains the container classes used by the DSLRunner for pipeline execution.
The containers implement a pull-based execution model where operations are lazily evaluated
only when their outputs are needed by parent nodes.
"""

import json
import math
import os
from typing import TYPE_CHECKING

from rich.panel import Panel

from docetl.dataset import Dataset
from docetl.operations import get_operation
from docetl.operations.utils import flush_cache
from docetl.optimizers import JoinOptimizer, MapOptimizer, ReduceOptimizer
from docetl.utils import smart_sample

if TYPE_CHECKING:
    from docetl.runner import DSLRunner

SUPPORTED_OPS = ["map", "resolve", "reduce", "equijoin", "filter"]
NUM_OPTIMIZER_RETRIES = 1


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

    def __init__(self, name: str, runner: "DSLRunner", config: dict, **kwargs):
        self._name = name
        parts = name.split("/", 1)
        self.step_name = parts[0]
        self.op_name = parts[1] if len(parts) > 1 else parts[0]
        self.config = config
        self.children = []
        self.parent = None
        self.is_equijoin = config.get("type") == "equijoin"
        self.runner = runner
        self.selectivity = kwargs.get("selectivity", None)
        if not self.selectivity:
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

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value
        parts = value.split("/", 1)
        self.step_name = parts[0]
        self.op_name = parts[1] if len(parts) > 1 else parts[0]

    def to_string(self) -> str:
        return json.dumps(self.config, indent=2)

    def add_child(self, child: "OpContainer") -> None:
        self.children.append(child)
        child.parent = self

    def optimize(self):
        """
        Optimize the next operation, to get a sample of size sample_size.
        Along the way, we will replace this op container with the optimized op container.

        We do the following:
        1. Optimize the children
        2. Run the children to get the input data for optimizing this operation
        3. Optimize this operation and replace it with the optimized op containers
        """
        # Return early if already optimized
        if self.is_optimized:
            return

        # optimize the children
        for child in self.children:
            child.optimize()

        # Figure out the sample size needed for this operation from the sample size map
        # It may be None if the operation is not in the sample size map, which means we will get all the data
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

        # Optimize this operation if it's eligible for optimization
        new_head_pointer = self
        if self.config.get("optimize", False):
            if self.config["type"] not in SUPPORTED_OPS:
                self.runner.console.log(
                    f"[red]Operation {self.name} is not supported for optimization. Proceeding without optimizing it.[/red]"
                )
            else:
                # If this is a build operation, set the captured output
                self.runner.optimizer.captured_output.set_step(self.step_name)

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

                # Use rich console status to indicate optimization of the operation
                with self.runner.console.status(
                    f"[bold blue]Optimizing operation: {self.name} (Type: {self.config['type']})[/bold blue]"
                ) as status:
                    self.runner.status = status
                    optimized_ops = []

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
                                )
                                optimized_ops, _, cost = map_optimizer.optimize(
                                    self.config,
                                    input_data[0],
                                    plan_types=self.runner.config.get(
                                        "optimizer_config", {}
                                    )
                                    .get("map", {})
                                    .get(
                                        "plan_types",
                                        ["chunk", "proj_synthesis", "glean"],
                                    ),
                                )
                                self.runner.total_cost += cost
                            elif self.config.get("type") == "reduce":
                                reduce_optimizer = ReduceOptimizer(
                                    self.runner,
                                    self.runner._run_operation,
                                )
                                optimized_ops, _, cost = reduce_optimizer.optimize(
                                    self.config, input_data[0]
                                )
                                self.runner.total_cost += cost
                            elif self.config.get("type") == "resolve":
                                optimized_config, cost = JoinOptimizer(
                                    self.runner,
                                    self.config,
                                    target_recall=self.runner.config.get(
                                        "optimizer_config", {}
                                    )
                                    .get("resolve", {})
                                    .get("target_recall", 0.95),
                                    estimated_selectivity=self.runner.config.get(
                                        "optimizer_config", {}
                                    )
                                    .get("resolve", {})
                                    .get("estimated_selectivity", None),
                                ).optimize_resolve(input_data[0])
                                op_config = self.config.copy()
                                op_config.update(optimized_config)
                                optimized_ops = (
                                    [op_config]
                                    if not optimized_config.get("empty", False)
                                    else []
                                )
                                self.runner.total_cost += cost

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
                                # Set this current config to be op_config
                                self.config = op_config

                                # Replace old op map
                                self.runner.op_container_map = {
                                    k: v
                                    for k, v in self.runner.op_container_map.items()
                                    if k
                                    not in [
                                        self.children[0].name,
                                        self.children[1].name,
                                    ]
                                }

                                # Set the children to be scans of the new left and right names
                                curr_step_name = self.step_name
                                self.children[0].config = {
                                    "type": "scan",
                                    "name": f"scan_{new_left_name}",
                                    "dataset_name": new_left_name,
                                }
                                self.children[0].name = (
                                    f"{curr_step_name}/scan_{new_left_name}"
                                )
                                self.children[1].config = {
                                    "type": "scan",
                                    "name": f"scan_{new_right_name}",
                                    "dataset_name": new_right_name,
                                }
                                self.children[1].name = (
                                    f"{curr_step_name}/scan_{new_right_name}"
                                )

                                # Replace in the op map
                                self.runner.op_container_map[
                                    f"{curr_step_name}/scan_{new_left_name}"
                                ] = self.children[0]
                                self.runner.op_container_map[
                                    f"{curr_step_name}/scan_{new_right_name}"
                                ] = self.children[1]

                                # Find the child dataset name that changed (left or right)
                                left_changed = new_left_name != self.kwargs["left_name"]
                                if left_changed:
                                    # Set the left to be the local last op container
                                    local_last_op_container = self.children[0]
                                else:
                                    # Set the right to be the local last op container
                                    local_last_op_container = self.children[1]

                                # Change the kwargs left and right names
                                self.kwargs["left_name"] = new_left_name
                                self.kwargs["right_name"] = new_right_name

                                # Insert new containers before local_last_op_container's children and local_last_op_container
                                old_children = local_last_op_container.children
                                local_last_op_container.children = []

                                # Add the new steps and operations to the query plan
                                for step_name, step_obj, operations in reversed(
                                    new_steps
                                ):
                                    # Create the step boundary op container
                                    step_boundary_container = StepBoundary(
                                        f"{step_name}/boundary",
                                        self.runner,
                                        {
                                            "type": "step_boundary",
                                            "name": f"{step_name}/boundary",
                                        },
                                    )
                                    self.runner.op_container_map[
                                        f"{step_name}/boundary"
                                    ] = step_boundary_container
                                    # Point the equijoin op container to this step boundary
                                    local_last_op_container.add_child(
                                        step_boundary_container
                                    )

                                    local_last_op_container = step_boundary_container

                                    # Create new op containers for each operation
                                    for op in operations:
                                        op_container = OpContainer(
                                            f"{step_name}/{op['name']}", self.runner, op
                                        )
                                        self.runner.op_container_map[
                                            f"{step_name}/{op['name']}"
                                        ] = op_container
                                        local_last_op_container.add_child(op_container)
                                        local_last_op_container = op_container

                                    # Add a scan operation based on the input for the step op
                                    scan_op_container = OpContainer(
                                        f"{step_name}/scan_{step_obj['input']}",
                                        self.runner,
                                        {
                                            "type": "scan",
                                            "name": f"scan_{step_obj['input']}",
                                            "dataset_name": step_obj["input"],
                                        },
                                    )
                                    self.runner.op_container_map[
                                        f"{step_name}/scan_{step_obj['input']}"
                                    ] = scan_op_container
                                    local_last_op_container.add_child(scan_op_container)
                                    local_last_op_container = scan_op_container

                                # Set the local_last_op_container's children to the old children
                                for child in old_children:
                                    local_last_op_container.add_child(child)

                            else:
                                raise ValueError(
                                    f"Unsupported operation type: {self.config['type']}"
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
                                raise  # If this was the last retry, re-raise the exception
                            self.runner.console.log(
                                f"Optimization attempt {retry + 1} failed with error: {e}. Retrying..."
                            )

                    if len(optimized_ops) > 0:
                        # Replace this op container with the optimized op containers
                        # Since this is not an equijoin, we have only one child
                        old_children = self.children
                        self.children = []
                        local_last_op_container = self.parent
                        local_last_op_container.children = []
                        curr_step_name = self.step_name

                        for idx, op in enumerate(list(reversed(optimized_ops))):
                            op_container = OpContainer(
                                f"{curr_step_name}/{op['name']}", self.runner, op
                            )
                            if idx == 0:
                                new_head_pointer = op_container

                            self.runner.op_container_map[
                                f"{curr_step_name}/{op['name']}"
                            ] = op_container
                            local_last_op_container.add_child(op_container)
                            local_last_op_container = op_container

                        for child in old_children:
                            local_last_op_container.add_child(child)

        # Figure out the sample size needed for this operation from the sample size map
        # It may be None if the operation is not in the sample size map, which means we will get all the data
        sample_size_needed = self.runner.optimizer.sample_size_map.get(
            new_head_pointer.config["type"]
        )
        # if it's an equijoin, sample_size_needed may be a dictionary
        if new_head_pointer.config["type"] == "equijoin":
            if isinstance(sample_size_needed, dict):
                sample_size_needed = min(
                    sample_size_needed["left"], sample_size_needed["right"]
                )

        # walk down the new head pointer and set the selectivities
        queue = [new_head_pointer] if new_head_pointer.parent else []
        while queue:
            curr_op = queue.pop(0)
            if not curr_op.selectivity:
                # Run the operation to set the selectivity
                if len(curr_op.children) == 0:
                    # Selectivity is 1 because it's a scan
                    curr_op.selectivity = 1
                else:
                    # Just run the operation because next will set the selectivity
                    curr_op.next(is_build=True, sample_size_needed=sample_size_needed)

            # Set the curr op to be optimized
            curr_op.is_optimized = True

            queue.extend(curr_op.children)

        # Checkpoint the optimized operations
        self.runner.optimizer.checkpoint_optimized_ops()

    def _pull_children(
        self, is_build: bool = False, sample_size_needed: int = None
    ) -> tuple:
        """Pull input data from child nodes. Returns (input_data, input_len, cost, logs)."""
        if self.is_equijoin:
            assert len(self.children) == 2, "Equijoin should have left and right children"
            left_data, left_cost, left_logs = self.children[0].next(is_build, sample_size_needed)
            right_data, right_cost, right_logs = self.children[1].next(is_build, sample_size_needed)
            return (
                {"left_data": left_data, "right_data": right_data},
                max(len(left_data), len(right_data)),
                left_cost + right_cost,
                left_logs + right_logs,
            )
        elif len(self.children) > 0:
            data, cost, logs = self.children[0].next(is_build, sample_size_needed)
            return data, len(data), cost, logs
        return None, None, 0.0, ""

    def _token_totals(self):
        p = sum(u["prompt_tokens"] for u in self.runner.total_token_usage.values())
        c = sum(u["completion_tokens"] for u in self.runner.total_token_usage.values())
        return p, c

    def _notify_cached(self, data: list[dict]) -> None:
        """Surface a cached operation in the interactive progress view."""
        tracker = getattr(self.runner, "progress_tracker", None)
        if tracker is not None and self.config.get("type") not in ("scan", "step_boundary"):
            tracker.op_start(
                self.name,
                self.config.get("type", "?"),
                self.config.get("model", self.runner.default_model),
                len(data),
            )
            tracker.op_done(self.name, cost=0.0, prompt_tokens=0, completion_tokens=0, outputs=data)

    def next(
        self, is_build: bool = False, sample_size_needed: int = None
    ) -> tuple[list[dict], float, str]:
        """
        Execute this operation and return its results via pull-based evaluation.

        Returns:
            (output_data, total_cost, execution_logs)
        """
        # Build-phase: check the optimizer's sample cache (v1 optimizer path)
        if is_build:
            cache_key = self.name
            if cache_key in self.runner.optimizer.sample_cache:
                cached_data, cached_sample_size = self.runner.optimizer.sample_cache[cache_key]
                if not sample_size_needed or cached_sample_size >= sample_size_needed:
                    if sample_size_needed:
                        cached_data = smart_sample(cached_data, sample_size_needed)
                    return cached_data, 0, f"[green]✓[/green] Using cached {self.name} (sample size: {cached_sample_size})\n"

        # Production path: try loading from checkpoint
        if not is_build and not self.config.get("bypass_cache", False):
            cached = self.runner._load_from_checkpoint_if_exists(self.step_name, self.op_name)
            if cached is not None:
                self._notify_cached(cached)
                return cached, 0, f"[green]✓[/green] Using cached {self.name}\n"

        # Compute how many input items we need (selectivity-adjusted for build phase)
        if self.selectivity and sample_size_needed:
            input_sample_size = int(math.ceil(sample_size_needed / self.selectivity))
        else:
            input_sample_size = sample_size_needed

        # Clear stale checkpoint
        if self.runner.intermediate_dir:
            checkpoint_path = os.path.join(self.runner.intermediate_dir, self.step_name, f"{self.op_name}.json")
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

        # Pull inputs from children
        input_data, input_len, cost, curr_logs = self._pull_children(is_build, input_sample_size)

        # Apply sampling if configured
        if input_data and "sample" in self.config and not is_build:
            input_data = input_data[: self.config["sample"]]

        # Start progress tracking
        tracker = getattr(self.runner, "progress_tracker", None)
        track = (
            tracker is not None
            and not is_build
            and self.config.get("type") not in ("scan", "step_boundary")
        )
        if track:
            tokens_before = self._token_totals()
            tracker.op_start(
                self.name,
                self.config.get("type", "?"),
                self.config.get("model", self.runner.default_model),
                input_len,
            )

        # Execute the operation
        with self.runner.console.status(f"Running {self.name}") as status:
            self.runner.status = status
            cost_before = self.runner.total_cost

            output_data = self.runner._run_operation(self.config, input_data, is_build=is_build)

            op_cost = self.runner.total_cost - cost_before
            cost += op_cost

        # Finish progress tracking
        if track:
            tokens_after = self._token_totals()
            tracker.op_done(
                self.name,
                cost=op_cost,
                prompt_tokens=tokens_after[0] - tokens_before[0],
                completion_tokens=tokens_after[1] - tokens_before[1],
                outputs=output_data,
            )

        build_indicator = "[yellow](build)[/yellow] " if is_build else ""
        curr_logs += f"[green]✓[/green] {build_indicator}{self.name} (Cost: [green]${op_cost:.2f}[/green])\n"
        self.runner.console.log(
            f"[green]✓[/green] {build_indicator}{self.name} (Cost: [green]${op_cost:.2f}[/green])"
        )

        # Update selectivity estimate
        self.selectivity = len(output_data) / input_len if input_len else 1

        # Build-phase: cache for the v1 optimizer
        if is_build:
            self.runner.optimizer.sample_cache[self.name] = (output_data, len(output_data))

        if sample_size_needed:
            output_data = smart_sample(output_data, sample_size_needed)

        # Save checkpoint
        if (
            not is_build
            and self.runner.intermediate_dir
            and self.op_name in self.runner.step_op_hashes.get(self.step_name, {})
        ):
            self.runner._save_checkpoint(self.step_name, self.op_name, output_data)

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


class StepBoundary(OpContainer):
    def next(
        self, is_build: bool = False, sample_size_needed: int = None
    ) -> tuple[list[dict], float, str]:

        output_data, step_cost, step_logs = self.children[0].next(
            is_build, sample_size_needed
        )

        # Print step logs only if not building
        self.runner.datasets[self.step_name] = Dataset(
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

    def syntax_check(self) -> str:
        return ""
