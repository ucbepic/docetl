"""V1 pipeline optimizer: sample-based rewrites for map/reduce/join operations."""

import copy
import hashlib
import os
from typing import TYPE_CHECKING, Any, Callable

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

    def __init__(
        self,
        runner: "DSLRunner",
        rewrite_agent_model: str = "gpt-5.1",
        judge_agent_model: str = "gpt-4o-mini",
        litellm_kwargs: dict[str, Any] = {},
        resume: bool = False,
        timeout: int = 60,
    ):
        self.config = runner.config
        self.console = runner.console
        self.max_threads = runner.max_threads

        self.base_name = runner.base_name
        self.yaml_file_suffix = runner.yaml_file_suffix
        self.runner = runner
        self.status = runner.status

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

        self.sample_cache = {}

        home_dir = os.environ.get("DOCETL_HOME_DIR", os.path.expanduser("~"))
        cache_dir = os.path.join(home_dir, f".docetl/cache/{runner.yaml_file_suffix}")
        os.makedirs(cache_dir, exist_ok=True)

        config_hash = hashlib.sha256(str(self.config).encode()).hexdigest()
        self.optimized_ops_path = f"{cache_dir}/{config_hash}.yaml"

        self.sample_size_map = SAMPLE_SIZE_MAP
        if self.config.get("optimizer_config", {}).get("sample_sizes", {}):
            self.sample_size_map.update(self.config["optimizer_config"]["sample_sizes"])

        if not self.runner._from_df_accessors:
            self.print_optimizer_config()

    def print_optimizer_config(self):
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
        if not self.runner.last_op_container:
            return

        containers_to_check = [self.runner.last_op_container]
        while containers_to_check:
            current = containers_to_check.pop(0)

            if isinstance(current, StepBoundary) or not current.children:
                containers_to_check.extend(current.children)
                continue

            if current.config["type"] == "reduce" and current.config.get(
                "synthesize_resolve", True
            ):
                new_container = self._maybe_synthesize_resolve(current)
                if new_container:
                    containers_to_check.extend(new_container.children)
                    continue

            containers_to_check.extend(current.children)

    def _find_map_without_resolve(self, container, visited=None):
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
            result = self._find_map_without_resolve(child, visited)
            if result:
                return result
        return None

    def _maybe_synthesize_resolve(self, reduce_container) -> OpContainer | None:
        reduce_key = reduce_container.config.get("reduce_key", "_all")
        if isinstance(reduce_key, str):
            reduce_key = [reduce_key]
        if "_all" in reduce_key:
            return None

        map_desc = self._find_map_without_resolve(reduce_container)
        if not map_desc:
            return None

        step_name = reduce_container.step_name
        self.console.log("[yellow]Synthesizing empty resolver operation:[/yellow]")
        self.console.log(f"  • [cyan]Reduce operation:[/cyan] [bold]{reduce_container.name}[/bold]")
        self.console.log(f"  • [cyan]Step:[/cyan] [bold]{step_name}[/bold]")

        resolve_name = f"synthesized_resolve_{len(self.config['operations'])}"
        resolve_config = {
            "name": resolve_name,
            "type": "resolve",
            "empty": True,
            "optimize": True,
            "embedding_model": "text-embedding-3-small",
            "resolution_model": self.config.get("default_model", "gpt-4o-mini"),
            "comparison_model": self.config.get("default_model", "gpt-4o-mini"),
            "_intermediates": {
                "map_prompt": map_desc.config.get("prompt"),
                "reduce_key": reduce_key,
            },
        }
        self.config["operations"].append(resolve_config)

        resolve_container = OpContainer(
            f"{step_name}/{resolve_name}", self.runner, resolve_config,
        )
        resolve_container.children = reduce_container.children
        for child in resolve_container.children:
            child.parent = resolve_container
        reduce_container.children = [resolve_container]
        resolve_container.parent = reduce_container
        self.runner.op_container_map[f"{step_name}/{resolve_name}"] = resolve_container
        return resolve_container

    def should_optimize(
        self, step_name: str, op_name: str
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], float]:
        self.console.rule("[bold cyan]Beginning Pipeline Assessment[/bold cyan]")

        self._insert_empty_resolve_operations()

        node_of_interest = self.runner.op_container_map[f"{step_name}/{op_name}"]

        input_data = []
        for child in node_of_interest.children:
            input_data.append(
                child.next(
                    is_build=True,
                    sample_size_needed=SAMPLE_SIZE_MAP.get(child.config["type"]),
                )[0]
            )

        self.captured_output.set_step(step_name)

        if (
            node_of_interest.config.get("type") == "map"
            or node_of_interest.config.get("type") == "filter"
        ):
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

            if should_optimize_output == "":
                return "", [], [], 0.0
        else:
            return "", [], [], 0.0

        return (
            should_optimize_output,
            input_data,
            output_data,
            self.runner.total_cost + self.llm_client.total_cost,
        )

    def optimize(self) -> float:
        self.console.rule("[bold cyan]Beginning Pipeline Rewrites[/bold cyan]")

        if self.resume:
            if os.path.exists(self.optimized_ops_path):
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

        self.runner.last_op_container.optimize()
        flush_cache(self.console)
        self.console.rule("[bold cyan]Optimized Query Plan[/bold cyan]")
        self.runner.print_query_plan()

        return self.llm_client.total_cost

    def _optimize_equijoin(
        self,
        op_config: dict[str, Any],
        left_name: str,
        right_name: str,
        left_data: list[dict[str, Any]],
        right_data: list[dict[str, Any]],
        run_operation: Callable[
            [dict[str, Any], list[dict[str, Any]]], list[dict[str, Any]]
        ],
    ) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], str, str]:
        new_left_name = left_name
        new_right_name = right_name
        new_steps = []

        equijoin_cfg = self.runner.config.get("optimizer_config", {}).get("equijoin", {})
        for _ in range(2):
            join_optimizer = JoinOptimizer(
                self.runner, op_config,
                target_recall=equijoin_cfg.get("target_recall", 0.95),
                estimated_selectivity=equijoin_cfg.get("estimated_selectivity", None),
            )
            optimized_config, cost, agent_results = join_optimizer.optimize_equijoin(
                left_data, right_data
            )
            self.runner.total_cost += cost
            op_config.update(optimized_config)

            if not agent_results.get("optimize_map", False):
                break

            step, ops = self._synthesize_extraction_step(
                agent_results, left_name, right_name, left_data, right_data, run_operation,
            )
            is_left = agent_results["dataset_to_transform"] == "left"
            if is_left:
                new_left_name = step["name"]
                left_data = self._run_synthesized_ops(ops, left_data, run_operation)
            else:
                new_right_name = step["name"]
                right_data = self._run_synthesized_ops(ops, right_data, run_operation)

            new_steps.append((step["name"], step, ops))

            if self.runner.status:
                self.runner.status.update(
                    f"Optimizing equijoin operation with {agent_results['output_key']} extraction"
                )

        return op_config, new_steps, new_left_name, new_right_name

    def _synthesize_extraction_step(
        self, agent_results, left_name, right_name, left_data, right_data, run_operation,
    ) -> tuple[dict, list[dict]]:
        output_key = agent_results["output_key"]
        if self.runner.status:
            self.runner.status.update(
                f"Optimizing map operation for {output_key} extraction to help with the equijoin"
            )

        map_operation = {
            "name": f"synthesized_{output_key}_extraction",
            "type": "map",
            "prompt": agent_results["map_prompt"],
            "model": self.config.get("default_model", "gpt-4o-mini"),
            "output": {"schema": {output_key: "string"}},
            "optimize": False,
        }
        optimized_ops = [map_operation]

        is_left = agent_results["dataset_to_transform"] == "left"
        step = {
            "name": f"synthesized_{output_key}_extraction",
            "input": left_name if is_left else right_name,
            "operations": [op["name"] for op in optimized_ops],
        }
        return step, optimized_ops

    @staticmethod
    def _run_synthesized_ops(ops, data, run_operation):
        for op in ops:
            data = run_operation(op, data)
        return data

    def checkpoint_optimized_ops(self) -> None:
        clean_config = self.clean_optimized_config()
        with open(self.optimized_ops_path, "w") as f:
            yaml.safe_dump(clean_config, f, default_flow_style=False, width=80)

    def clean_optimized_config(self) -> dict:
        if not self.runner.last_op_container:
            return self.config

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
            "pipeline": self.runner.config.get("pipeline", {}).copy(),
        }
        clean_config["pipeline"]["steps"] = []
        seen_operations = set()

        def clean_operation(op_container: OpContainer) -> dict:
            clean_op = copy.deepcopy(op_container.config)
            clean_op.pop("_intermediates", None)
            if op_container.is_optimized:
                for field in ["recursively_optimize", "optimize"]:
                    clean_op.pop(field, None)
            return clean_op

        def process_container(container, current_step=None):
            if isinstance(container, StepBoundary):
                if container.children:
                    return process_container(container.children[0], current_step)
                return None, None

            step_name = container.step_name
            if not current_step or current_step["name"] != step_name:
                current_step = {"name": step_name, "operations": []}
                clean_config["pipeline"]["steps"].insert(0, current_step)

            if container.config["type"] == "scan":
                if container.children:
                    return process_container(container.children[0], current_step)
                return None, current_step

            if container.name not in seen_operations:
                clean_config["operations"].append(clean_operation(container))
                seen_operations.add(container.name)

            if container.is_equijoin:
                current_step["operations"].insert(
                    0,
                    {
                        container.config["name"]: {
                            "left": container.kwargs["left_name"],
                            "right": container.kwargs["right_name"],
                        }
                    },
                )
                if container.children:
                    process_container(container.children[0], current_step)
                    process_container(container.children[1], current_step)
            else:
                current_step["operations"].insert(0, container.config["name"])
                if container.children:
                    for child in container.children:
                        process_container(child, current_step)

            return container, current_step

        process_container(self.runner.last_op_container)

        for step in clean_config["pipeline"]["steps"]:
            first_op = step["operations"][0]
            if isinstance(first_op, dict):
                continue
            elif len(step["operations"]) > 0:
                op_container = self.runner.op_container_map.get(
                    f"{step['name']}/{first_op}"
                )
                if op_container and op_container.children:
                    child = op_container.children[0]
                    while (
                        child
                        and child.config["type"] == "step_boundary"
                        and child.children
                    ):
                        child = child.children[0]
                    if child and child.config["type"] == "scan":
                        step["input"] = child.config["dataset_name"]

        for key, value in self.config.items():
            if key not in ["datasets", "operations", "pipeline"]:
                clean_config[key] = value

        return clean_config

    def save_optimized_config(self, optimized_config_path: str):
        resolved_config = self.clean_optimized_config()

        with open(optimized_config_path, "w") as f:
            yaml.safe_dump(resolved_config, f, default_flow_style=False, width=80)
            self.console.log(
                f"[green italic]💾 Optimized config saved to {optimized_config_path}[/green italic]"
            )
