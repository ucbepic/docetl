"""Pull-based execution containers for pipeline operations."""

import json
import math
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
        if self.is_optimized:
            return

        for child in self.children:
            child.optimize()

        sample_size_needed = self.runner.optimizer.sample_size_map.get(
            self.config["type"]
        )
        if self.config["type"] == "equijoin":
            if isinstance(sample_size_needed, dict):
                sample_size_needed = [sample_size_needed["left"], sample_size_needed["right"]]
            else:
                sample_size_needed = [sample_size_needed, sample_size_needed]
        else:
            sample_size_needed = [sample_size_needed]

        input_data = [
            child.next(is_build=True, sample_size_needed=sample_size_needed[idx])[0]
            for idx, child in enumerate(self.children)
        ]

        new_head_pointer = self
        if self.config.get("optimize", False):
            if self.config["type"] not in SUPPORTED_OPS:
                self.runner.console.log(
                    f"[red]Operation {self.name} is not supported for optimization.[/red]"
                )
            else:
                new_head_pointer = self._run_optimizer(input_data)

        self._propagate_selectivities(new_head_pointer)
        self.runner.optimizer.checkpoint_optimized_ops()

    def _run_optimizer(self, input_data: list) -> "OpContainer":
        self.runner.optimizer.captured_output.set_step(self.step_name)
        self._log_optimization_panel(input_data)

        opt_cfg = self.runner.config.get("optimizer_config", {})
        num_retries = opt_cfg.get("num_retries", NUM_OPTIMIZER_RETRIES)

        with self.runner.console.status(
            f"[bold blue]Optimizing operation: {self.name} (Type: {self.config['type']})[/bold blue]"
        ) as status:
            self.runner.status = status
            optimized_ops = []

            for retry in range(num_retries):
                try:
                    op_type = self.config["type"]
                    if op_type in ("map", "filter"):
                        optimized_ops = self._optimize_map_filter(input_data)
                    elif op_type == "reduce":
                        optimized_ops = self._optimize_reduce(input_data)
                    elif op_type == "resolve":
                        optimized_ops = self._optimize_resolve(input_data)
                    elif op_type == "equijoin":
                        self._optimize_equijoin(input_data)
                        return self
                    else:
                        raise ValueError(f"Unsupported operation type: {op_type}")
                    break
                except Exception as e:
                    if retry == num_retries - 1:
                        raise
                    self.runner.console.log(
                        f"Optimization attempt {retry + 1} failed with error: {e}. Retrying..."
                    )

            if optimized_ops:
                return self._replace_with_optimized(optimized_ops)
        return self

    def _log_optimization_panel(self, input_data: list) -> None:
        if self.config["type"] == "equijoin":
            sample_info = [
                f"[yellow]Sample size (left): {len(input_data[0])}",
                f"[yellow]Sample size (right): {len(input_data[1])}",
            ]
        else:
            sample_info = [f"[yellow]Sample size: {len(input_data[0])}"]

        opt_type_cfg = self.runner.config.get("optimizer_config", {}).get(
            self.config["type"], {}
        )
        panel_content = "\n".join(sample_info)
        if opt_type_cfg:
            panel_content += "\n\n[cyan]Optimizer Config:[/cyan]"
            for key, value in opt_type_cfg.items():
                panel_content += f"\n[cyan]{key}:[/cyan] {value}"

        self.runner.console.log(
            Panel.fit(panel_content, title=f"[yellow]Optimizing {self.name} (Type: {self.config['type']})")
        )

    def _optimize_map_filter(self, input_data: list) -> list[dict]:
        opt_cfg = self.runner.config.get("optimizer_config", {})
        optimizer = MapOptimizer(
            self.runner,
            self.runner._run_operation,
            is_filter=self.config["type"] == "filter",
        )
        optimized_ops, _, cost = optimizer.optimize(
            self.config,
            input_data[0],
            plan_types=opt_cfg.get("map", {}).get(
                "plan_types", ["chunk", "proj_synthesis", "glean"]
            ),
        )
        self.runner.total_cost += cost
        return optimized_ops

    def _optimize_reduce(self, input_data: list) -> list[dict]:
        optimizer = ReduceOptimizer(self.runner, self.runner._run_operation)
        optimized_ops, _, cost = optimizer.optimize(self.config, input_data[0])
        self.runner.total_cost += cost
        return optimized_ops

    def _optimize_resolve(self, input_data: list) -> list[dict]:
        opt_cfg = self.runner.config.get("optimizer_config", {})
        resolve_cfg = opt_cfg.get("resolve", {})
        optimized_config, cost = JoinOptimizer(
            self.runner,
            self.config,
            target_recall=resolve_cfg.get("target_recall", 0.95),
            estimated_selectivity=resolve_cfg.get("estimated_selectivity", None),
        ).optimize_resolve(input_data[0])
        op_config = self.config.copy()
        op_config.update(optimized_config)
        self.runner.total_cost += cost
        if optimized_config.get("empty", False):
            return []
        return [op_config]

    def _optimize_equijoin(self, input_data: list) -> None:
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
        self.config = op_config

        self.runner.op_container_map = {
            k: v for k, v in self.runner.op_container_map.items()
            if k not in [self.children[0].name, self.children[1].name]
        }

        step = self.step_name
        for idx, ds_name in enumerate([new_left_name, new_right_name]):
            self.children[idx].config = {
                "type": "scan", "name": f"scan_{ds_name}", "dataset_name": ds_name,
            }
            self.children[idx].name = f"{step}/scan_{ds_name}"
            self.runner.op_container_map[f"{step}/scan_{ds_name}"] = self.children[idx]

        left_changed = new_left_name != self.kwargs["left_name"]
        insertion_point = self.children[0] if left_changed else self.children[1]
        self.kwargs["left_name"] = new_left_name
        self.kwargs["right_name"] = new_right_name

        old_children = insertion_point.children
        insertion_point.children = []

        for step_name, step_obj, operations in reversed(new_steps):
            boundary = StepBoundary(
                f"{step_name}/boundary", self.runner,
                {"type": "step_boundary", "name": f"{step_name}/boundary"},
            )
            self.runner.op_container_map[f"{step_name}/boundary"] = boundary
            insertion_point.add_child(boundary)
            insertion_point = boundary

            for op in operations:
                container = OpContainer(f"{step_name}/{op['name']}", self.runner, op)
                self.runner.op_container_map[f"{step_name}/{op['name']}"] = container
                insertion_point.add_child(container)
                insertion_point = container

            scan = OpContainer(
                f"{step_name}/scan_{step_obj['input']}", self.runner,
                {"type": "scan", "name": f"scan_{step_obj['input']}", "dataset_name": step_obj["input"]},
            )
            self.runner.op_container_map[f"{step_name}/scan_{step_obj['input']}"] = scan
            insertion_point.add_child(scan)
            insertion_point = scan

        for child in old_children:
            insertion_point.add_child(child)

    def _replace_with_optimized(self, optimized_ops: list[dict]) -> "OpContainer":
        old_children = self.children
        self.children = []
        parent = self.parent
        parent.children = []
        step = self.step_name
        new_head = None

        for idx, op in enumerate(reversed(optimized_ops)):
            container = OpContainer(f"{step}/{op['name']}", self.runner, op)
            if idx == 0:
                new_head = container
            self.runner.op_container_map[f"{step}/{op['name']}"] = container
            parent.add_child(container)
            parent = container

        for child in old_children:
            parent.add_child(child)

        return new_head

    def _propagate_selectivities(self, head: "OpContainer") -> None:
        sample_size = self.runner.optimizer.sample_size_map.get(head.config["type"])
        if head.config["type"] == "equijoin" and isinstance(sample_size, dict):
            sample_size = min(sample_size["left"], sample_size["right"])

        queue = [head] if head.parent else []
        while queue:
            curr = queue.pop(0)
            if not curr.selectivity:
                if not curr.children:
                    curr.selectivity = 1
                else:
                    curr.next(is_build=True, sample_size_needed=sample_size)
            curr.is_optimized = True
            queue.extend(curr.children)

    def _pull_children(
        self, is_build: bool = False, sample_size_needed: int = None
    ) -> tuple:
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
        tracker = getattr(self.runner, "progress_tracker", None)
        if tracker is not None and self.config.get("type") not in ("scan", "step_boundary"):
            tracker.op_start(
                self.name,
                self.config.get("type", "?"),
                self.config.get("model", self.runner.default_model),
                len(data),
            )
            tracker.op_done(self.name, cost=0.0, prompt_tokens=0, completion_tokens=0, outputs=data)

    def _check_caches(
        self, is_build: bool, sample_size_needed: int | None
    ) -> tuple[list[dict], float, str] | None:
        if is_build:
            cache_key = self.name
            if cache_key in self.runner.optimizer.sample_cache:
                cached_data, cached_sample_size = self.runner.optimizer.sample_cache[cache_key]
                if not sample_size_needed or cached_sample_size >= sample_size_needed:
                    if sample_size_needed:
                        cached_data = smart_sample(cached_data, sample_size_needed)
                    return cached_data, 0, f"[green]✓[/green] Using cached {self.name} (sample size: {cached_sample_size})\n"

        if not is_build and not self.config.get("bypass_cache", False) and self.runner.checkpoints:
            cached = self.runner.checkpoints.load(self.step_name, self.op_name)
            if cached is not None:
                self.runner.console.log(
                    f"[green]✓[/green] [italic]Loaded checkpoint for operation "
                    f"'{self.op_name}' in step '{self.step_name}'[/italic]"
                )
                self._notify_cached(cached)
                return cached, 0, f"[green]✓[/green] Using cached {self.name}\n"

        return None

    def _execute_and_track(
        self, input_data, input_len: int, is_build: bool
    ) -> tuple[list[dict], float]:
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

        with self.runner.console.status(f"Running {self.name}") as status:
            self.runner.status = status
            cost_before = self.runner.total_cost
            output_data = self.runner._run_operation(self.config, input_data, is_build=is_build)
            op_cost = self.runner.total_cost - cost_before

        if track:
            tokens_after = self._token_totals()
            tracker.op_done(
                self.name,
                cost=op_cost,
                prompt_tokens=tokens_after[0] - tokens_before[0],
                completion_tokens=tokens_after[1] - tokens_before[1],
                outputs=output_data,
            )

        return output_data, op_cost

    def next(
        self, is_build: bool = False, sample_size_needed: int = None
    ) -> tuple[list[dict], float, str]:
        cached = self._check_caches(is_build, sample_size_needed)
        if cached is not None:
            return cached

        if self.selectivity and sample_size_needed:
            input_sample_size = int(math.ceil(sample_size_needed / self.selectivity))
        else:
            input_sample_size = sample_size_needed

        if self.runner.checkpoints:
            self.runner.checkpoints.clear_stale(self.step_name, self.op_name)

        input_data, input_len, cost, curr_logs = self._pull_children(is_build, input_sample_size)

        if input_data and "sample" in self.config and not is_build:
            input_data = input_data[: self.config["sample"]]

        output_data, op_cost = self._execute_and_track(input_data, input_len, is_build)
        cost += op_cost

        build_indicator = "[yellow](build)[/yellow] " if is_build else ""
        curr_logs += f"[green]✓[/green] {build_indicator}{self.name} (Cost: [green]${op_cost:.2f}[/green])\n"
        self.runner.console.log(
            f"[green]✓[/green] {build_indicator}{self.name} (Cost: [green]${op_cost:.2f}[/green])"
        )

        self.selectivity = len(output_data) / input_len if input_len else 1

        if is_build:
            self.runner.optimizer.sample_cache[self.name] = (output_data, len(output_data))

        if sample_size_needed:
            output_data = smart_sample(output_data, sample_size_needed)

        if (
            not is_build
            and self.runner.checkpoints
            and self.runner.checkpoints.has_hash(self.step_name, self.op_name)
        ):
            path = self.runner.checkpoints.save(self.step_name, self.op_name, output_data)
            self.runner.console.log(
                f"[green]✓ [italic]Intermediate saved for operation '{self.op_name}' "
                f"in step '{self.step_name}' at {path}[/italic][/green]"
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

        obj.syntax_check()

        return f"[green]✓[/green] Operation '{operation}' ({operation_type})"


class StepBoundary(OpContainer):
    def next(
        self, is_build: bool = False, sample_size_needed: int = None
    ) -> tuple[list[dict], float, str]:

        output_data, step_cost, step_logs = self.children[0].next(
            is_build, sample_size_needed
        )

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
