"""
The DSLRunner module implements a declarative pipeline execution engine with a pull-based
evaluation model. Key architectural decisions include:

Design Patterns:
- Pull-based DAG: Operations are lazily evaluated only when their outputs are needed,
  enabling efficient resource utilization and caching
- Dependency Injection: Operations receive their dependencies through a standardized interface,
  making the system modular and testable
- Builder Pattern: Pipeline construction is separated from execution, allowing validation
  and optimization before runtime

Core Features:
- Transparent Caching: Automatic checkpointing and reuse of intermediate results
- Cost Tracking: Built-in tracking of operation costs for optimization
- Schema Validation: Type checking and schema validation at both build and runtime
- Extensible Operations: New operations can be added by implementing the operation interface

The architecture prioritizes:
1. Separation of Concerns: Building, validation, and execution are distinct phases
2. Flexibility: Support for both streaming and batch processing patterns
3. Observability: Rich logging and cost tracking throughout execution
4. Performance: Lazy evaluation and caching optimize resource usage
"""

from __future__ import annotations

import datetime
import functools
import hashlib
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
from rich.markup import escape
from rich.panel import Panel

from docetl.console import get_console
from docetl.containers import OpContainer, StepBoundary
from docetl.dataset import Dataset, create_parsing_tool_map
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


class DSLRunner:
    """
    DSLRunner orchestrates pipeline execution by building and traversing a DAG of OpContainers.
    The runner uses a two-phase approach:

    1. Build Phase:
       - Parses YAML config into a DAG of OpContainers
       - Each operation becomes a node connected to its dependencies
       - Special handling for equijoins which have two parent nodes
       - Validates operation syntax and schema compatibility

    2. Execution Phase:
       - Starts from the final operation and pulls data through the DAG
       - Handles caching/checkpointing of intermediate results
       - Tracks costs and execution metrics
       - Manages dataset loading and result persistence

    The separation between build and execution phases allows for:
    - Pipeline validation before any execution
    - Cost estimation and optimization
    - Partial pipeline execution for testing
    """

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
        """
        Initialize the DSLRunner with a config dict or a typed ``Pipeline`` object.

        Args:
            config: A raw YAML-style config dict **or** a ``docetl.api.Pipeline``
                instance.  When a ``Pipeline`` is passed it is used directly;
                when a dict is passed it is converted via ``Pipeline.from_dict``.
                Either way, ``self.pipeline`` holds the typed representation and
                ``self.config`` holds the raw dict (for backward compat with
                ConfigWrapper and code that still reads it).
            max_threads (int, optional): Maximum number of threads to use. Defaults to None.
        """
        from docetl.api import Pipeline as PipelineCls

        if isinstance(config, PipelineCls):
            self.pipeline: PipelineCls = config
            config_dict = config._to_dict()
        else:
            config_dict = config
            self.pipeline = PipelineCls.from_dict(config_dict)

        # Keep the raw operations list for _op_map so that checkpoint hashes
        # match regardless of whether Pipeline was passed or dict was passed.
        self._raw_ops_list = config_dict.get("operations", [])

        # --- Config & runtime setup (formerly ConfigWrapper) ---
        self.config = config_dict
        self.base_name = kwargs.pop("base_name", None)
        self.yaml_file_suffix = kwargs.pop("yaml_file_suffix", None) or (
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.default_model = self.config.get("default_model", "gpt-4o-mini")
        console = kwargs.pop("console", None)
        if console:
            self.console = console
        else:
            self.console = get_console()
        self.max_threads = max_threads or (os.cpu_count() or 1) * 4
        self.status = None

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

        bucket_factory = create_bucket_factory(self.config.get("rate_limits", {}))
        self.rate_limiter = pyrate_limiter.Limiter(bucket_factory, max_delay=200)
        self.is_cancelled = False

        self.fallback_models_config = self.config.get("fallback_models", [])
        self.fallback_embedding_models_config = self.config.get("fallback_embedding_models", [])
        self.router = self._create_router(self.fallback_models_config, "completion")
        self.embedding_router = self._create_router(self.fallback_embedding_models_config, "embedding")
        self._router_cache: dict[str, Any] = {}
        self.api = APIWrapper(self)
        # --- End config setup ---

        self.total_cost = 0
        self.total_token_usage = defaultdict(
            lambda: {"prompt_tokens": 0, "completion_tokens": 0}
        )
        # Interactive progress TUI state. ``progress_tracker`` is only set while
        # an interactive run is active; the TUI is enabled by the top-level
        # ``interactive_ui`` flag in the config.
        self.progress_tracker = None
        self._tui_active = False
        self._initialize_state()
        self._setup_parsing_tools()
        self._setup_retrievers()
        self._build_operation_graph()
        self._compute_operation_hashes()

        # Run initial validation
        self._from_df_accessors = kwargs.get("from_df_accessors", False)
        if not self._from_df_accessors:
            self.syntax_check()

    def _initialize_state(self) -> None:
        """Initialize basic runner state and datasets"""
        self.datasets = {}
        self.intermediate_dir = self.pipeline.output.intermediate_dir

    def _create_router(self, fallback_models: list, router_type: str) -> Any | None:
        if not fallback_models:
            return None
        try:
            from litellm import Router
        except ImportError:
            self.console.log(
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
            self.console.log(
                f"[green]Created LiteLLM {router_type} Router with {len(model_list)} fallback model(s) in order: {', '.join(fallback_model_names)}[/green]"
            )
            return router
        except Exception as e:
            self.console.log(
                f"[yellow]Warning: Failed to create LiteLLM {router_type} Router: {e}. Fallback models will be ignored.[/yellow]"
            )
            return None

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
        """Set up parsing tools from configuration"""
        self.parsing_tool_map = create_parsing_tool_map(
            self.config.get("parsing_tools", None)
        )

    def _setup_retrievers(self) -> None:
        """Instantiate retrievers from configuration (lazy index creation)."""
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

    def _build_operation_graph(self) -> None:
        """Build the DAG of operations from ``self.pipeline``."""
        self.op_container_map = {}
        self.last_op_container = None
        self._op_map = {op["name"]: op for op in self._raw_ops_list}

        for step in self.pipeline.steps:
            step_dict = {k: v for k, v in step.dict().items() if v is not None}
            self._validate_step(step_dict)

            if step.input:
                self._add_scan_operation(step_dict)
            elif step.operations and isinstance(step.operations[0], dict):
                self._add_equijoin_operation(step_dict)
            else:
                self._add_empty_scan_operation(step_dict)

            self._add_step_operations(step_dict)
            self._add_step_boundary(step_dict)

    def _validate_step(self, step: dict) -> None:
        """Validate step configuration"""
        assert "name" in step.keys(), f"Step {step} does not have a name"
        assert "operations" in step.keys(), f"Step {step} does not have `operations`"

    def _add_scan_operation(self, step: dict) -> None:
        """Add a scan operation for input datasets"""
        scan_op_container = OpContainer(
            f"{step['name']}/scan_{step['input']}",
            self,
            {
                "type": "scan",
                "dataset_name": step["input"],
                "name": f"scan_{step['input']}",
            },
        )
        self.op_container_map[f"{step['name']}/scan_{step['input']}"] = (
            scan_op_container
        )
        if self.last_op_container:
            scan_op_container.add_child(self.last_op_container)
        self.last_op_container = scan_op_container

    def _add_empty_scan_operation(self, step: dict) -> None:
        """Add a scan operation for a synthetic empty dataset [{}]"""
        dataset_name = "__empty__"
        scan_op_container = OpContainer(
            f"{step['name']}/scan_{dataset_name}",
            self,
            {
                "type": "scan",
                "dataset_name": dataset_name,
                "name": f"scan_{dataset_name}",
            },
        )
        self.op_container_map[f"{step['name']}/scan_{dataset_name}"] = scan_op_container
        if self.last_op_container:
            scan_op_container.add_child(self.last_op_container)
        self.last_op_container = scan_op_container

    def _add_equijoin_operation(self, step: dict) -> None:
        """Add an equijoin operation with its scan operations"""
        equijoin_operation_name = list(step["operations"][0].keys())[0]
        left_dataset_name = list(step["operations"][0].values())[0]["left"]
        right_dataset_name = list(step["operations"][0].values())[0]["right"]

        left_scan_op_container = OpContainer(
            f"{step['name']}/scan_{left_dataset_name}",
            self,
            {
                "type": "scan",
                "dataset_name": left_dataset_name,
                "name": f"scan_{left_dataset_name}",
            },
        )
        if self.last_op_container:
            left_scan_op_container.add_child(self.last_op_container)
        right_scan_op_container = OpContainer(
            f"{step['name']}/scan_{right_dataset_name}",
            self,
            {
                "type": "scan",
                "dataset_name": right_dataset_name,
                "name": f"scan_{right_dataset_name}",
            },
        )
        if self.last_op_container:
            right_scan_op_container.add_child(self.last_op_container)
        equijoin_op_container = OpContainer(
            f"{step['name']}/{equijoin_operation_name}",
            self,
            self.find_operation(equijoin_operation_name),
            left_name=left_dataset_name,
            right_name=right_dataset_name,
        )

        equijoin_op_container.add_child(left_scan_op_container)
        equijoin_op_container.add_child(right_scan_op_container)

        self.last_op_container = equijoin_op_container
        self.op_container_map[f"{step['name']}/{equijoin_operation_name}"] = (
            equijoin_op_container
        )
        self.op_container_map[f"{step['name']}/scan_{left_dataset_name}"] = (
            left_scan_op_container
        )
        self.op_container_map[f"{step['name']}/scan_{right_dataset_name}"] = (
            right_scan_op_container
        )

    def _add_step_operations(self, step: dict) -> None:
        """Add operations for a step"""
        # Skip first op only for equijoin (first op is a dict, not a string)
        is_equijoin = (
            step.get("input") is None
            and step["operations"]
            and isinstance(step["operations"][0], dict)
        )
        op_start_idx = 1 if is_equijoin else 0

        for operation_name in step["operations"][op_start_idx:]:
            if not isinstance(operation_name, str):
                raise ValueError(
                    f"Operation {operation_name} in step {step['name']} should be a string. "
                    "If you intend for it to be an equijoin, don't specify an input in the step."
                )

            op_container = OpContainer(
                f"{step['name']}/{operation_name}",
                self,
                self.find_operation(operation_name),
            )
            op_container.add_child(self.last_op_container)
            self.last_op_container = op_container
            self.op_container_map[f"{step['name']}/{operation_name}"] = op_container

    def _add_step_boundary(self, step: dict) -> None:
        """Add a step boundary node"""
        step_boundary = StepBoundary(
            f"{step['name']}/boundary",
            self,
            {"type": "step_boundary", "name": f"{step['name']}/boundary"},
        )
        step_boundary.add_child(self.last_op_container)
        self.op_container_map[f"{step['name']}/boundary"] = step_boundary
        self.last_op_container = step_boundary

    def _compute_operation_hashes(self) -> None:
        """Compute hashes for operations to enable caching."""
        self.step_op_hashes = defaultdict(dict)

        for step in self.pipeline.steps:
            for idx, entry in enumerate(step.operations):
                op_name = entry if isinstance(entry, str) else list(entry.keys())[0]

                all_ops_until_and_including_current = (
                    [self._op_map[prev] for prev in step.operations[:idx]
                     if isinstance(prev, str)]
                    + [self._op_map[op_name]]
                    + [self.pipeline.other_config.get("system_prompt", {})]
                )

                for op_cfg in all_ops_until_and_including_current:
                    if isinstance(op_cfg, dict) and "model" not in op_cfg:
                        op_cfg["model"] = self.default_model

                all_ops_str = json.dumps(all_ops_until_and_including_current)
                self.step_op_hashes[step.name][op_name] = hashlib.sha256(
                    all_ops_str.encode()
                ).hexdigest()

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
        """
        Perform a syntax check on all operations defined in the configuration.
        """
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
        """
        Print a visual representation of the entire query plan using indentation and arrows.
        Operations are color-coded by step to show the pipeline structure while maintaining
        dependencies between steps.
        """
        if not self.last_op_container:
            self.console.log("\n[bold]Pipeline Steps:[/bold]")
            self.console.log(
                Panel("No operations in pipeline", title="Query Plan", width=100)
            )
            self.console.log()
            return

        def _print_op(
            op: OpContainer, indent: int = 0, step_colors: dict[str, str] | None = None
        ) -> str:
            # Handle boundary operations based on show_boundaries flag
            if isinstance(op, StepBoundary):
                if show_boundaries:
                    output = []
                    indent_str = "  " * indent
                    step_name = op.step_name
                    color = step_colors.get(step_name, "white")
                    output.append(
                        f"{indent_str}[{color}][bold]{op.name}[/bold][/{color}]"
                    )
                    output.append(f"{indent_str}Type: step_boundary")
                    if op.children:
                        output.append(f"{indent_str}[yellow]▼[/yellow]")
                        for child in op.children:
                            output.append(_print_op(child, indent + 1, step_colors))
                    return "\n".join(output)
                elif op.children:
                    return _print_op(op.children[0], indent, step_colors)
                return ""

            # Build the string for the current operation with indentation
            indent_str = "  " * indent
            output = []

            # Color code the operation name based on its step
            step_name = op.step_name
            color = step_colors.get(step_name, "white")
            output.append(f"{indent_str}[{color}][bold]{op.name}[/bold][/{color}]")
            output.append(f"{indent_str}Type: {op.config['type']}")

            # Add schema if available
            if "output" in op.config and "schema" in op.config["output"]:
                output.append(f"{indent_str}Output Schema:")
                for field, field_type in op.config["output"]["schema"].items():
                    escaped_type = escape(str(field_type))
                    output.append(
                        f"{indent_str}  {field}: [bright_white]{escaped_type}[/bright_white]"
                    )

            # Add children
            if op.children:
                if op.is_equijoin:
                    output.append(f"{indent_str}[yellow]▼ LEFT[/yellow]")
                    output.append(_print_op(op.children[0], indent + 1, step_colors))
                    output.append(f"{indent_str}[yellow]▼ RIGHT[/yellow]")
                    output.append(_print_op(op.children[1], indent + 1, step_colors))
                else:
                    output.append(f"{indent_str}[yellow]▼[/yellow]")
                    for child in op.children:
                        output.append(_print_op(child, indent + 1, step_colors))

            return "\n".join(output)

        # Get all step boundaries and extract unique step names
        step_boundaries = [
            op
            for name, op in self.op_container_map.items()
            if isinstance(op, StepBoundary)
        ]
        step_boundaries.sort(key=lambda x: x.name)

        # Create a color map for steps - using distinct colors
        colors = ["cyan", "magenta", "green", "yellow", "blue", "red"]
        step_names = [b.step_name for b in step_boundaries]
        step_colors = {
            name: colors[i % len(colors)] for i, name in enumerate(step_names)
        }

        # Print the legend
        self.console.log("\n[bold]Pipeline Steps:[/bold]")
        for step_name, color in step_colors.items():
            self.console.log(f"[{color}]■[/{color}] {step_name}")

        # Print the full query plan starting from the last step boundary
        query_plan = _print_op(self.last_op_container, step_colors=step_colors)
        self.console.log(Panel(query_plan, title="Query Plan", width=100))
        self.console.log()

    def find_operation(self, op_name: str) -> dict:
        try:
            return self._op_map[op_name]
        except KeyError:
            raise ValueError(f"Operation '{op_name}' not found in configuration.")

    def list_pipeline_operations(self) -> list[tuple[str, str, str, str | None]]:
        """Return ``(step, full_name, op_type, model)`` for each real operation
        in pipeline order, excluding scans and step boundaries.

        Used to pre-populate the interactive progress view so all operations are
        visible (as ``queued``) before execution begins.
        """
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
        """Decide whether to launch the interactive TUI for this run.

        Enabled by the top-level ``interactive_ui`` flag in the config (alongside
        ``default_model``). The TUI requires an interactive terminal; otherwise
        we fall back to plain logging.
        """
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
        """
        Execute the entire pipeline defined in the configuration.
        """
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

        # Print execution summary
        token_usage_lines = ""
        if self.total_token_usage:
            token_usage_lines = "\n[bold]Token Usage:[/bold]\n"
            total_prompt = 0
            total_completion = 0
            total_cached = 0
            for model, usage in sorted(self.total_token_usage.items()):
                prompt = usage["prompt_tokens"]
                completion = usage["completion_tokens"]
                cached = usage.get("cached_tokens", 0)
                total_prompt += prompt
                total_completion += completion
                total_cached += cached
                line = f"  {model}: [cyan]{prompt:,}[/cyan] input"
                if cached:
                    line += f" ([dim]{cached:,} cached[/dim])"
                line += f", [cyan]{completion:,}[/cyan] output"
                token_usage_lines += line + "\n"
            if len(self.total_token_usage) > 1:
                total_line = f"  [bold]Total: [cyan]{total_prompt:,}[/cyan] input"
                if total_cached:
                    total_line += f" ([dim]{total_cached:,} cached[/dim])"
                total_line += f", [cyan]{total_completion:,}[/cyan] output[/bold]"
                token_usage_lines += total_line + "\n"

        summary = (
            f"Cost: [green]${self.total_cost:.2f}[/green]\n"
            f"Time: {execution_time:.2f}s\n"
            + token_usage_lines
            + (
                f"Cache: [dim]{self.intermediate_dir}[/dim]\n"
                if self.intermediate_dir
                else ""
            )
            + f"Output: [dim]{output_path}[/dim]"
        )
        self.console.log(Panel(summary, title="Execution Summary"))

        if self.progress_tracker is not None:
            self.progress_tracker.pipeline_done()

        return self.total_cost

    def load(self) -> None:
        """
        Load all datasets defined in the configuration.
        """
        datasets = {}
        self.console.rule("[bold]Loading Datasets[/bold]")

        for name, ds in self.pipeline.datasets.items():
            ds_type = ds.type if hasattr(ds, "type") else ds.get("type")
            ds_path = ds.path if hasattr(ds, "path") else ds.get("path")
            ds_parsing = (ds.parsing if hasattr(ds, "parsing") else ds.get("parsing")) or []

            if ds_type == "file":
                datasets[name] = Dataset(
                    self, "file", ds_path, source="local",
                    parsing=ds_parsing,
                    user_defined_parsing_tool_map=self.parsing_tool_map,
                )
                self.console.log(
                    f"[green]✓[/green] Loaded dataset '{name}' from {ds_path}"
                )
            elif ds_type == "memory":
                datasets[name] = Dataset(
                    self, "memory", ds_path, source="local",
                    parsing=ds_parsing,
                    user_defined_parsing_tool_map=self.parsing_tool_map,
                )
                self.console.log(
                    f"[green]✓[/green] Loaded dataset '{name}' from in-memory data"
                )
            else:
                raise ValueError(f"Unsupported dataset type: {ds_type}")

        self.datasets = {
            name: (
                dataset
                if isinstance(dataset, Dataset)
                else Dataset(self, "memory", dataset)
            )
            for name, dataset in datasets.items()
        }
        # Always register a synthetic empty dataset for pipelines without input data
        self.datasets["__empty__"] = Dataset(self, "memory", [{}])
        self.console.log()

    def save(self, data: list[dict]) -> None:
        """
        Save the final output of the pipeline.
        """
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
        if self.intermediate_dir is None or self.config.get("bypass_cache", False):
            return None

        intermediate_config_path = os.path.join(
            self.intermediate_dir, ".docetl_intermediate_config.json"
        )

        if not os.path.exists(intermediate_config_path):
            return None

        # Make sure the step and op name is in the checkpoint config path
        if (
            step_name not in self.step_op_hashes
            or operation_name not in self.step_op_hashes[step_name]
        ):
            return None

        # See if the checkpoint config is the same as the current step op hash
        with open(intermediate_config_path, "r") as f:
            intermediate_config = json.load(f)

        if (
            intermediate_config.get(step_name, {}).get(operation_name, "")
            != self.step_op_hashes[step_name][operation_name]
        ):
            return None

        checkpoint_path = os.path.join(
            self.intermediate_dir, step_name, f"{operation_name}.json"
        )
        # check if checkpoint exists
        if os.path.exists(checkpoint_path):
            if f"{step_name}_{operation_name}" not in self.datasets:
                self.datasets[f"{step_name}_{operation_name}"] = Dataset(
                    self, "file", checkpoint_path, "local"
                )

                self.console.log(
                    f"[green]✓[/green] [italic]Loaded checkpoint for operation '{operation_name}' in step '{step_name}' from {checkpoint_path}[/italic]"
                )

                return self.datasets[f"{step_name}_{operation_name}"].load()
        return None

    def clear_intermediate(self) -> None:
        """
        Clear the intermediate directory.
        """
        # Remove the intermediate directory
        if self.intermediate_dir:
            shutil.rmtree(self.intermediate_dir)
            return

        raise ValueError("Intermediate directory not set. Cannot clear intermediate.")

    def _save_checkpoint(
        self, step_name: str, operation_name: str, data: list[dict]
    ) -> None:
        """
        Save a checkpoint of the current data after an operation.

        This method creates a JSON file containing the current state of the data
        after an operation has been executed. The checkpoint is saved in a directory
        structure that reflects the step and operation names.

        Args:
            step_name (str): The name of the current step in the pipeline.
            operation_name (str): The name of the operation that was just executed.
            data (list[dict]): The current state of the data to be checkpointed.

        Note:
            The checkpoint is saved only if a checkpoint directory has been specified
            when initializing the DSLRunner.
        """
        checkpoint_path = os.path.join(
            self.intermediate_dir, step_name, f"{operation_name}.json"
        )
        if os.path.dirname(checkpoint_path):
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        with open(checkpoint_path, "w") as f:
            json.dump(data, f)

        # Update the intermediate config file with the hash for this step/operation
        # so that future runs can validate and reuse this checkpoint.
        if self.intermediate_dir:
            intermediate_config_path = os.path.join(
                self.intermediate_dir, ".docetl_intermediate_config.json"
            )

            # Initialize or load existing intermediate configuration
            if os.path.exists(intermediate_config_path):
                try:
                    with open(intermediate_config_path, "r") as cfg_file:
                        intermediate_config: dict[str, dict[str, str]] = json.load(
                            cfg_file
                        )
                except json.JSONDecodeError:
                    # If the file is corrupted, start fresh to avoid crashes
                    intermediate_config = {}
            else:
                intermediate_config = {}

            # Ensure nested dict structure exists
            step_dict = intermediate_config.setdefault(step_name, {})

            # Write (or overwrite) the hash for the current operation
            step_dict[operation_name] = self.step_op_hashes[step_name][operation_name]

            # Persist the updated configuration
            with open(intermediate_config_path, "w") as cfg_file:
                json.dump(intermediate_config, cfg_file, indent=2)

        self.console.log(
            f"[green]✓ [italic]Intermediate saved for operation '{operation_name}' in step '{step_name}' at {checkpoint_path}[/italic][/green]"
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
        """
        Run a single operation based on its configuration.

        This method creates an instance of the appropriate operation class and executes it.
        It also updates the total operation cost.

        Args:
            op_config (dict[str, Any]): The configuration of the operation to run.
            input_data (list[dict[str, Any]]): The input data for the operation.
            return_instance (bool, optional): If True, return the operation instance along with the output data.

        Returns:
            list[dict[str, Any]] | tuple[list[dict[str, Any]], BaseOperation, float]:
            If return_instance is False, returns the output data.
            If return_instance is True, returns a tuple of the output data, the operation instance, and the cost.
        """
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
        """
        Save partial (batch-level) results from an operation to a directory named
        '<operation_name>_batches' inside the intermediate directory.

        Args:
            operation_name (str): The name of the operation, e.g. 'extract_medications'.
            batch_index (int): Zero-based index of the batch.
            data (list[dict]): Batch results to write to disk.
        """
        if not self.intermediate_dir:
            return

        op_batches_dir = os.path.join(
            self.intermediate_dir, f"{operation_name}_batches"
        )
        os.makedirs(op_batches_dir, exist_ok=True)

        # File name: 'batch_0.json', 'batch_1.json', etc.
        checkpoint_path = os.path.join(op_batches_dir, f"batch_{batch_index}.json")

        with open(checkpoint_path, "w") as f:
            json.dump(data, f)

        self.console.log(
            f"[green]✓[/green] [italic]Partial checkpoint saved for '{operation_name}', "
            f"batch {batch_index} at '{checkpoint_path}'[/italic]"
        )
