"""Pipeline execution engine with pull-based DAG evaluation."""

from __future__ import annotations

import datetime
import functools
import json
import os
import shutil
import time
from collections import defaultdict
from typing import Any

import pyrate_limiter
from dotenv import load_dotenv
from pyrate_limiter import BucketFullException, LimiterDelayException
from pydantic import BaseModel
from rich.panel import Panel

from docetl.console import get_console
from docetl.containers import OpContainer, StepBoundary
from docetl.dataset import Dataset, create_parsing_tool_map
from docetl.display import format_execution_summary, format_query_plan
from docetl.graph_builder import build_operation_graph, compute_operation_hashes
from docetl.operations import get_operation, get_operations
from docetl.operations.base import BaseOperation
from docetl.operations.utils import APIWrapper
from docetl.optimizer import Optimizer
from docetl.ratelimiter import create_bucket_factory
from docetl.utils import classproperty, decrypt, load_config

from . import schemas

# Avoid circular import — Pipeline is only needed for isinstance checks
# and from_dict calls, so import lazily or use TYPE_CHECKING.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docetl.api import Pipeline as PipelineType

load_dotenv()


def _create_router(console, fallback_models: list, router_type: str) -> Any | None:
    if not fallback_models:
        return None
    try:
        from litellm import Router
    except ImportError:
        console.log(
            f"[yellow]Warning: LiteLLM Router not available. Fallback {router_type} models will be ignored.[/yellow]"
        )
        return None

    model_list = []
    fallback_model_names = []
    for fallback_config in fallback_models:
        if isinstance(fallback_config, dict):
            model_name = fallback_config.get("model_name")
            litellm_params = fallback_config.get("litellm_params", {})
        elif isinstance(fallback_config, str):
            model_name = fallback_config
            litellm_params = {}
        else:
            continue
        if not model_name:
            continue
        litellm_params_with_model = litellm_params.copy()
        litellm_params_with_model["model"] = model_name
        model_list.append({"model_name": model_name, "litellm_params": litellm_params_with_model})
        fallback_model_names.append(model_name)

    if not model_list:
        return None
    try:
        router_kwargs = {"model_list": model_list}
        if len(fallback_model_names) > 1:
            fallbacks = []
            for i, model_name in enumerate(fallback_model_names):
                if i < len(fallback_model_names) - 1:
                    fallbacks.append({model_name: fallback_model_names[i + 1:]})
            router_kwargs["fallbacks"] = fallbacks
        router = Router(**router_kwargs)
        console.log(
            f"[green]Created LiteLLM {router_type} Router with {len(model_list)} fallback model(s) in order: {', '.join(fallback_model_names)}[/green]"
        )
        return router
    except Exception as e:
        console.log(
            f"[yellow]Warning: Failed to create LiteLLM {router_type} Router: {e}. Fallback models will be ignored.[/yellow]"
        )
        return None


class DSLRunner:

    @classproperty
    def schema(cls):
        # Accessing the schema loads all operations, so only do this
        # when we actually need it...
        # Yes, this means DSLRunner.schema isn't really accessible to
        # static type checkers. But it /is/ available for dynamic
        # checking, and for generating json schema.

        OpType = functools.reduce(
            lambda a, b: a | b, [op.schema for op in get_operations().values()]
        )
        # More pythonic implementation of the above, but only works in python 3.11:
        # OpType = Union[*[op.schema for op in get_operations().values()]]

        class Pipeline(BaseModel):
            config: dict[str, Any] | None
            parsing_tools: list[schemas.ParsingTool] | None
            datasets: dict[str, schemas.Dataset] | None = None
            retrievers: dict[str, Any] | None
            operations: list[OpType]
            pipeline: schemas.PipelineSpec
            # When true (and stdout is a TTY), launch the full-screen interactive
            # progress view for this run (requires the optional ``tui`` extra).
            interactive_ui: bool = False

        return Pipeline

    @classproperty
    def json_schema(cls):
        return cls.schema.model_json_schema()

    @classmethod
    def from_yaml(cls, yaml_file: str, **kwargs):
        if not yaml_file.endswith(".yaml") and not yaml_file.endswith(".yml"):
            raise ValueError(
                "Invalid file type. Please provide a YAML file ending with '.yaml' or '.yml'."
            )
        base_name = yaml_file.rsplit(".", 1)[0]
        suffix = yaml_file.split("/")[-1].split(".")[0]
        config = load_config(yaml_file)
        return cls(config, base_name=base_name, yaml_file_suffix=suffix, **kwargs)

    def __init__(self, config: "dict | PipelineType", max_threads: int | None = None, **kwargs):
        from docetl.api import Pipeline as PipelineCls

        if isinstance(config, PipelineCls):
            self.pipeline: PipelineCls = config
            config_dict = config._to_dict()
        else:
            config_dict = config
            self.pipeline = PipelineCls.from_dict(config_dict)

        self._raw_ops_list = config_dict.get("operations", [])
        self.config = config_dict
        self.base_name = kwargs.pop("base_name", None)
        self.yaml_file_suffix = kwargs.pop("yaml_file_suffix", None) or (
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.default_model = self.config.get("default_model", "gpt-4o-mini")
        self.console = kwargs.pop("console", None) or get_console()
        self.max_threads = max_threads or (os.cpu_count() or 1) * 4
        self.status = None

        self._setup_api_keys()
        self._setup_rate_limiter()
        self._setup_routers()
        self.api = APIWrapper(self)

        self.total_cost = 0
        self.total_token_usage = defaultdict(
            lambda: {"prompt_tokens": 0, "completion_tokens": 0}
        )
        self.progress_tracker = None
        self._tui_active = False

        self._initialize_state()
        self._setup_parsing_tools()
        self._setup_retrievers()
        build_operation_graph(self)
        compute_operation_hashes(self)
        self._setup_checkpoints()

        self._from_df_accessors = kwargs.get("from_df_accessors", False)
        if not self._from_df_accessors:
            self.syntax_check()

    def _setup_api_keys(self) -> None:
        encrypted_llm_api_keys = self.config.get("llm_api_keys", {})
        if encrypted_llm_api_keys:
            self.llm_api_keys = {
                key: decrypt(value, os.environ.get("DOCETL_ENCRYPTION_KEY", ""))
                for key, value in encrypted_llm_api_keys.items()
            }
        else:
            self.llm_api_keys = {}
        self._original_env = os.environ.copy()
        for key, value in self.llm_api_keys.items():
            os.environ[key] = value

    def _setup_rate_limiter(self) -> None:
        bucket_factory = create_bucket_factory(self.config.get("rate_limits", {}))
        self.rate_limiter = pyrate_limiter.Limiter(bucket_factory, max_delay=200)
        self.is_cancelled = False

    def _setup_routers(self) -> None:
        self.fallback_models_config = self.config.get("fallback_models", [])
        self.fallback_embedding_models_config = self.config.get("fallback_embedding_models", [])
        self.router = _create_router(self.console, self.fallback_models_config, "completion")
        self.embedding_router = _create_router(self.console, self.fallback_embedding_models_config, "embedding")
        self._router_cache: dict[str, Any] = {}

    def _setup_checkpoints(self) -> None:
        if self.intermediate_dir:
            from docetl.checkpoint import CheckpointStore
            self.checkpoints: "CheckpointStore | None" = CheckpointStore(
                self.intermediate_dir,
                self.step_op_hashes,
                bypass=self.config.get("bypass_cache", False),
            )
        else:
            self.checkpoints = None

    def _initialize_state(self) -> None:
        self.datasets = {}
        self.intermediate_dir = self.pipeline.output.intermediate_dir

    def reset_env(self):
        os.environ = self._original_env

    def blocking_acquire(self, key: str, weight: int, wait_time=0.5):
        while True:
            try:
                self.rate_limiter.try_acquire(key, weight=weight)
                return
            except LimiterDelayException as e:
                time_to_wait = e.meta_info["actual_delay"] / 1000
                self.console.log(
                    f"Rate limits met for {key}; sleeping for {max(time_to_wait, wait_time):.2f} seconds"
                )
                time.sleep(max(time_to_wait, wait_time))
            except BucketFullException as e:
                time_to_wait = e.meta_info["remaining_time"]
                self.console.log(
                    f"Rate limits met for {key}; sleeping for {max(time_to_wait, wait_time):.2f} seconds"
                )
                time.sleep(max(time_to_wait, wait_time))

    def _setup_parsing_tools(self) -> None:
        self.parsing_tool_map = create_parsing_tool_map(
            self.config.get("parsing_tools", None)
        )

    def _setup_retrievers(self) -> None:
        from docetl.retrievers.lancedb import LanceDBRetriever

        self.retrievers: dict[str, Any] = {}
        retrievers_cfg = self.config.get("retrievers", {}) or {}
        for name, rconf in retrievers_cfg.items():
            if not isinstance(rconf, dict):
                raise ValueError(f"Invalid retriever '{name}' configuration")
            if rconf.get("type") != "lancedb":
                raise ValueError(
                    f"Unsupported retriever type '{rconf.get('type')}' for '{name}'. Only 'lancedb' is supported."
                )
            required = ["dataset", "index_dir", "index_types"]
            for key in required:
                if key not in rconf:
                    raise ValueError(
                        f"Retriever '{name}' missing required key '{key}'."
                    )
            # Defaults
            rconf.setdefault("query", {"top_k": 5})
            rconf.setdefault("build_index", "if_missing")

            self.retrievers[name] = LanceDBRetriever(self, name, rconf)

    def get_output_path(self, require=False):
        output_path = self.pipeline.output.path or None
        if output_path:
            if not (
                output_path.lower().endswith(".json")
                or output_path.lower().endswith(".csv")
            ):
                raise ValueError(
                    f"Output path '{output_path}' is not a JSON or CSV file. Please provide a path ending with '.json' or '.csv'."
                )
        elif require:
            raise ValueError(
                "No output path specified in the configuration. Please provide an output path ending with '.json' or '.csv' in the configuration to use the save() method."
            )

        return output_path

    def syntax_check(self):
        self.console.log("[yellow]Checking operations...[/yellow]")

        # Just validate that it's a json file if specified
        self.get_output_path()
        current = self.last_op_container

        try:
            # Walk the last op container to check syntax
            op_containers = []
            if self.last_op_container:
                op_containers = [self.last_op_container]

            while op_containers:
                current = op_containers.pop(0)
                syntax_result = current.syntax_check()
                self.console.log(syntax_result, end="")
                # Add all children to the queue
                op_containers.extend(current.children)
        except Exception as e:
            raise ValueError(
                f"Syntax check failed for operation '{current.name}': {str(e)}"
            )

        self.console.log("[green]✓ All operations passed syntax check[/green]")

    def print_query_plan(self, show_boundaries=False):
        if not self.last_op_container:
            self.console.log("\n[bold]Pipeline Steps:[/bold]")
            self.console.log(Panel("No operations in pipeline", title="Query Plan", width=100))
            self.console.log()
            return

        step_colors, plan_text = format_query_plan(
            self.last_op_container, self.op_container_map, show_boundaries
        )
        self.console.log("\n[bold]Pipeline Steps:[/bold]")
        for step_name, color in step_colors.items():
            self.console.log(f"[{color}]■[/{color}] {step_name}")
        self.console.log(Panel(plan_text, title="Query Plan", width=100))
        self.console.log()

    def find_operation(self, op_name: str) -> dict:
        try:
            return self._op_map[op_name]
        except KeyError:
            raise ValueError(f"Operation '{op_name}' not found in configuration.")

    def list_pipeline_operations(self) -> list[tuple[str, str, str, str | None]]:
        ops_by_name = self.pipeline.ops_by_name
        ops: list[tuple[str, str, str, str | None]] = []
        for step in self.pipeline.steps:
            for entry in step.operations:
                op_name = entry if isinstance(entry, str) else list(entry.keys())[0]
                typed_op = ops_by_name.get(op_name)
                op_type = typed_op.type if typed_op else "?"
                if op_type in ("scan", "step_boundary"):
                    continue
                model = getattr(typed_op, "model", None) or self.default_model
                ops.append((step.name, f"{step.name}/{op_name}", op_type, model))
        return ops

    def _should_use_tui(self) -> bool:
        import sys

        if self._tui_active:
            return False
        if not self.config.get("interactive_ui", False):
            return False
        if not (sys.stdout.isatty() and sys.stdin.isatty()):
            self.console.log(
                "[yellow]interactive_ui requested but stdout is not a TTY; "
                "falling back to standard output.[/yellow]"
            )
            return False
        return True

    def load_run_save(self) -> float:
        # Route to the interactive TUI if requested and supported. The TUI runs
        # this same method again on a worker thread with ``_tui_active`` set, so
        # the actual execution path below is shared.
        if self._should_use_tui():
            try:
                from docetl.tui.app import run_with_tui
            except ImportError:
                self.console.log(
                    "[yellow]interactive_ui is enabled but the 'textual' package "
                    "is not installed. Install it with `pip install docetl[tui]` "
                    "to use the interactive progress view. Falling back to "
                    "standard output.[/yellow]"
                )
            else:
                return run_with_tui(self)

        output_path = self.get_output_path(require=True)

        # Print the query plan
        self.print_query_plan()

        start_time = time.time()

        if self.last_op_container:
            self.load()
            self.console.rule("[bold]Pipeline Execution[/bold]")
            output, _, _ = self.last_op_container.next()
            self.save(output)

        execution_time = time.time() - start_time

        summary = format_execution_summary(
            self.total_cost,
            execution_time,
            self.total_token_usage,
            self.intermediate_dir,
            output_path,
        )
        self.console.log(Panel(summary, title="Execution Summary"))

        if self.progress_tracker is not None:
            self.progress_tracker.pipeline_done()

        return self.total_cost

    def load(self) -> None:
        self.console.rule("[bold]Loading Datasets[/bold]")
        self.datasets = {}

        for name, ds in self.pipeline.datasets.items():
            parsing = ds.parsing or []
            self.datasets[name] = Dataset(
                self, ds.type, ds.path, source="local",
                parsing=parsing,
                user_defined_parsing_tool_map=self.parsing_tool_map,
            )
            label = ds.path if ds.type == "file" else "in-memory data"
            self.console.log(f"[green]✓[/green] Loaded dataset '{name}' from {label}")

        self.datasets["__empty__"] = Dataset(self, "memory", [{}])
        self.console.log()

    def save(self, data: list[dict]) -> None:
        self.get_output_path(require=True)

        out = self.pipeline.output
        if out.type == "file":
            if os.path.dirname(out.path):
                os.makedirs(os.path.dirname(out.path), exist_ok=True)
            if out.path.lower().endswith(".json"):
                with open(out.path, "w") as file:
                    json.dump(data, file, indent=2)
            else:  # CSV
                import csv

                with open(out.path, "w", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=data[0].keys())
                    limited_data = [
                        {k: d.get(k, None) for k in data[0].keys()} for d in data
                    ]
                    writer.writeheader()
                    writer.writerows(limited_data)
            self.console.log(
                f"[green]✓[/green] Saved to [dim]{out.path}[/dim]\n"
            )
        else:
            raise ValueError(
                f"Unsupported output type: {out.type}. Supported types: file"
            )

    def _load_from_checkpoint_if_exists(
        self, step_name: str, operation_name: str
    ) -> list[dict] | None:
        if not self.checkpoints:
            return None

        data = self.checkpoints.load(step_name, operation_name)
        if data is not None:
            self.datasets[f"{step_name}_{operation_name}"] = Dataset(
                self, "memory", data
            )
            self.console.log(
                f"[green]✓[/green] [italic]Loaded checkpoint for operation "
                f"'{operation_name}' in step '{step_name}'[/italic]"
            )
        return data

    def clear_intermediate(self) -> None:
        if self.checkpoints:
            self.checkpoints.clear_all()
            return
        raise ValueError("Intermediate directory not set. Cannot clear intermediate.")

    def _save_checkpoint(
        self, step_name: str, operation_name: str, data: list[dict]
    ) -> None:
        if not self.checkpoints:
            return
        path = self.checkpoints.save(step_name, operation_name, data)
        self.console.log(
            f"[green]✓ [italic]Intermediate saved for operation '{operation_name}' "
            f"in step '{step_name}' at {path}[/italic][/green]"
        )

    def should_optimize(
        self, step_name: str, op_name: str, **kwargs
    ) -> tuple[str, float, list[dict[str, Any]], list[dict[str, Any]]]:
        self.load()

        opt_cfg = self.pipeline.optimizer_config or {}
        kwargs["litellm_kwargs"] = opt_cfg.get("litellm_kwargs", {})
        kwargs["rewrite_agent_model"] = opt_cfg.get("rewrite_agent_model", "gpt-5.1")
        kwargs["judge_agent_model"] = opt_cfg.get("judge_agent_model", "gpt-4o-mini")

        builder = Optimizer(self, **kwargs)
        self.optimizer = builder
        result = builder.should_optimize(step_name, op_name)
        return result

    def optimize(
        self,
        save: bool = False,
        return_pipeline: bool = True,
        **kwargs,
    ) -> tuple[dict | "DSLRunner", float]:

        if not self.last_op_container:
            raise ValueError("No operations in pipeline. Cannot optimize.")

        self.load()

        opt_cfg = self.pipeline.optimizer_config or {}
        kwargs["litellm_kwargs"] = opt_cfg.get("litellm_kwargs", {})
        kwargs["rewrite_agent_model"] = opt_cfg.get("rewrite_agent_model", "gpt-5.1")
        kwargs["judge_agent_model"] = opt_cfg.get("judge_agent_model", "gpt-4o-mini")

        save_path = kwargs.get("save_path", None)
        # Pop the save_path from kwargs
        kwargs.pop("save_path", None)

        builder = Optimizer(
            self,
            **kwargs,
        )
        self.optimizer = builder
        llm_api_cost = builder.optimize()
        operations_cost = self.total_cost
        self.total_cost += llm_api_cost

        # Log the cost of optimization
        self.console.log(
            f"[green italic]💰 Total cost: ${self.total_cost:.4f}[/green italic]"
        )
        self.console.log(
            f"[green italic]  ├─ Operation execution cost: ${operations_cost:.4f}[/green italic]"
        )
        self.console.log(
            f"[green italic]  └─ Optimization cost: ${llm_api_cost:.4f}[/green italic]"
        )

        if save:
            # If output path is provided, save the optimized config to that path
            if kwargs.get("save_path"):
                save_path = kwargs["save_path"]
                if not os.path.isabs(save_path):
                    save_path = os.path.join(os.getcwd(), save_path)
                builder.save_optimized_config(save_path)
                self.optimized_config_path = save_path
            else:
                builder.save_optimized_config(f"{self.base_name}_opt.yaml")
                self.optimized_config_path = f"{self.base_name}_opt.yaml"

        if return_pipeline:
            return (
                DSLRunner(builder.clean_optimized_config(), self.max_threads),
                self.total_cost,
            )

        return builder.clean_optimized_config(), self.total_cost

    def _run_operation(
        self,
        op_config: dict[str, Any],
        input_data: list[dict[str, Any]] | dict[str, Any],
        return_instance: bool = False,
        is_build: bool = False,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], BaseOperation, float]:
        operation_class = get_operation(op_config["type"])

        oc_kwargs = {
            "runner": self,
            "config": op_config,
            "default_model": self.default_model,
            "max_threads": self.max_threads,
            "console": self.console,
            "status": self.status,
        }
        operation_instance = operation_class(**oc_kwargs)
        if op_config["type"] == "equijoin":
            output_data, cost = operation_instance.execute(
                input_data["left_data"], input_data["right_data"]
            )
        elif op_config["type"] == "filter":
            output_data, cost = operation_instance.execute(input_data, is_build)
        else:
            output_data, cost = operation_instance.execute(input_data)

        self.total_cost += cost

        if return_instance:
            return output_data, operation_instance
        else:
            return output_data

    def _flush_partial_results(
        self, operation_name: str, batch_index: int, data: list[dict]
    ) -> None:
        if self.checkpoints:
            path = self.checkpoints.flush_batch(operation_name, batch_index, data)
        elif self.intermediate_dir:
            batch_dir = os.path.join(self.intermediate_dir, f"{operation_name}_batches")
            os.makedirs(batch_dir, exist_ok=True)
            path = os.path.join(batch_dir, f"batch_{batch_index}.json")
            with open(path, "w") as f:
                json.dump(data, f)
        else:
            return
        self.console.log(
            f"[green]✓[/green] [italic]Partial checkpoint saved for '{operation_name}', "
            f"batch {batch_index} at '{path}'[/italic]"
        )
