"""
This module defines the core data structures and classes for the DocETL pipeline.

It includes Pydantic models for various operation types, pipeline steps, and the main Pipeline class.
The module provides a high-level API for defining, optimizing, and running document processing pipelines.

Classes:
    Dataset: Represents a dataset with a type, path, and optional parsing tools.
    BaseOp: Base class for all operation types.
    MapOp: Represents a map operation in the pipeline.
    ResolveOp: Represents a resolve operation for entity resolution.
    ReduceOp: Represents a reduce operation in the pipeline.
    ParallelMapOp: Represents a parallel map operation.
    FilterOp: Represents a filter operation in the pipeline.
    EquijoinOp: Represents an equijoin operation for joining datasets.
    SplitOp: Represents a split operation for dividing data.
    GatherOp: Represents a gather operation for collecting data.
    UnnestOp: Represents an unnest operation for flattening nested structures.
    PipelineStep: Represents a step in the pipeline with input and operations.
    PipelineOutput: Defines the output configuration for the pipeline.
    Pipeline: Main class for defining and running a complete document processing pipeline.

The Pipeline class provides methods for optimizing and running the defined pipeline,
as well as utility methods for converting between dictionary and object representations.

Usage:
    from docetl.api import Pipeline, Dataset, MapOp, ReduceOp

    pipeline = Pipeline(
        datasets={
            "input": Dataset(
                type="file",
                path="input.json",
                parsing=[{"name": "txt_to_string", "input_key": "text", "output_key": "content"}]
            )
        },
        operations=[
            MapOp(name="process", type="map", prompt="Process the document"),
            ReduceOp(name="summarize", type="reduce", reduce_key="content")
        ],
        steps=[
            PipelineStep(name="process_step", input="input", operations=["process"]),
            PipelineStep(name="summarize_step", input="process_step", operations=["summarize"])
        ],
        output=PipelineOutput(type="file", path="output.json")
    )

    optimized_pipeline = pipeline.optimize()
    result = optimized_pipeline.run()
"""

import inspect
import os
from typing import Any, Callable

import yaml
from rich import print

from docetl.runner import DSLRunner
from docetl.schemas import (
    ClusterOp,
    CodeFilterOp,
    CodeMapOp,
    CodeReduceOp,
    Dataset,
    EquijoinOp,
    ExtractOp,
    FilterOp,
    GatherOp,
    MapOp,
    OpType,
    ParallelMapOp,
    ParsingTool,
    PipelineOutput,
    PipelineStep,
    ReduceOp,
    ResolveOp,
    SampleOp,
    SplitOp,
    UnnestOp,
)


class Pipeline:
    """
    Represents a complete document processing pipeline.

    Attributes:
        name (str): The name of the pipeline.
        datasets (dict[str, Dataset]): A dictionary of datasets used in the pipeline,
                                       where keys are dataset names and values are Dataset objects.
        operations (list[OpType]): A list of operations to be performed in the pipeline.
        steps (list[PipelineStep]): A list of steps that make up the pipeline.
        output (PipelineOutput): The output configuration for the pipeline.
        parsing_tools (list[ParsingTool]): A list of parsing tools used in the pipeline.
                                           Defaults to an empty list.
        default_model (str | None): The default language model to use for operations
                                       that require one. Defaults to None.

    Example:
        ```python
        def custom_parser(text: str) -> list[str]:
            # this will convert the text in the column to uppercase
            # You should return a list of strings, where each string is a separate document
            return [text.upper()]

        pipeline = Pipeline(
            name="document_processing_pipeline",
            datasets={
                "input_data": Dataset(type="file", path="/path/to/input.json", parsing=[{"name": "custom_parser", "input_key": "content", "output_key": "uppercase_content"}]),
            },
            parsing_tools=[custom_parser],
            operations=[
                MapOp(
                    name="process",
                    type="map",
                    prompt="Determine what type of document this is: {{ input.uppercase_content }}",
                    output={"schema": {"document_type": "string"}}
                ),
                ReduceOp(
                    name="summarize",
                    type="reduce",
                    reduce_key="document_type",
                    prompt="Summarize the processed contents: {% for item in inputs %}{{ item.uppercase_content }} {% endfor %}",
                    output={"schema": {"summary": "string"}}
                )
            ],
            steps=[
                PipelineStep(name="process_step", input="input_data", operations=["process"]),
                PipelineStep(name="summarize_step", input="process_step", operations=["summarize"])
            ],
            output=PipelineOutput(type="file", path="/path/to/output.json"),
            default_model="gpt-4o-mini"
        )
        ```

    This example shows a complete pipeline configuration with datasets, operations,
    steps, and output settings.
    """

    # Maps operation type strings to their Pydantic schema classes.
    _OP_TYPE_REGISTRY: dict[str, type] = {
        "map": MapOp,
        "resolve": ResolveOp,
        "reduce": ReduceOp,
        "parallel_map": ParallelMapOp,
        "filter": FilterOp,
        "equijoin": EquijoinOp,
        "split": SplitOp,
        "gather": GatherOp,
        "unnest": UnnestOp,
        "cluster": ClusterOp,
        "sample": SampleOp,
        "code_map": CodeMapOp,
        "code_reduce": CodeReduceOp,
        "code_filter": CodeFilterOp,
        "extract": ExtractOp,
    }

    def __init__(
        self,
        name: str,
        datasets: dict[str, Dataset],
        operations: list[OpType],
        steps: list[PipelineStep],
        output: PipelineOutput,
        parsing_tools: list[ParsingTool | Callable] = [],
        default_model: str | None = None,
        rate_limits: dict[str, int] | None = None,
        optimizer_config: dict[str, Any] = {},
        **kwargs,
    ):
        self.name = name
        self.datasets = datasets
        self.operations = operations
        self.steps = steps
        self.output = output
        self.parsing_tools = [
            (
                tool
                if isinstance(tool, ParsingTool)
                else ParsingTool(
                    name=tool.__name__, function_code=inspect.getsource(tool)
                )
            )
            for tool in parsing_tools
        ]
        self.default_model = default_model
        self.rate_limits = rate_limits
        self.optimizer_config = optimizer_config

        # Add other kwargs to self.other_config
        self.other_config = kwargs

        self._load_env()

    # ------------------------------------------------------------------
    # Typed accessors
    # ------------------------------------------------------------------

    @property
    def ops_by_name(self) -> dict[str, OpType]:
        """Return a dict mapping operation name → typed operation object."""
        return {op.name: op for op in self.operations}

    def get_step_for_op(self, op_name: str) -> PipelineStep:
        """Return the step that contains *op_name*."""
        for step in self.steps:
            for entry in step.operations:
                name = entry if isinstance(entry, str) else list(entry.keys())[0]
                if name == op_name:
                    return step
        raise KeyError(f"Operation {op_name!r} not found in any step")

    # ------------------------------------------------------------------
    # Construction from raw dicts (YAML configs)
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, config: dict[str, Any], name: str | None = None) -> "Pipeline":
        """Build a ``Pipeline`` from a raw YAML-style config dict.

        This is the canonical way to go from untyped dicts to typed objects.
        Unknown operation types are kept as raw dicts so the round-trip is
        lossless even for plugin operation types.
        """
        datasets = {}
        for ds_name, ds_cfg in config.get("datasets", {}).items():
            datasets[ds_name] = Dataset(**ds_cfg)

        operations: list[OpType] = []
        for op_cfg in config.get("operations", []):
            op_type = op_cfg.get("type")
            schema_cls = cls._OP_TYPE_REGISTRY.get(op_type)
            if schema_cls is not None:
                filtered = {k: v for k, v in op_cfg.items() if v is not None}
                operations.append(schema_cls(**filtered))
            else:
                # Unknown / plugin operation type — store as-is via MapOp
                # with extra fields preserved through Pydantic's extra="allow"
                operations.append(MapOp(**{k: v for k, v in op_cfg.items() if v is not None}))

        steps = []
        for step_cfg in config.get("pipeline", {}).get("steps", []):
            steps.append(PipelineStep(**{k: v for k, v in step_cfg.items() if v is not None}))

        output = PipelineOutput(**config.get("pipeline", {}).get("output", {}))

        parsing_tools = []
        for tool_cfg in config.get("parsing_tools", []) or []:
            if isinstance(tool_cfg, ParsingTool):
                parsing_tools.append(tool_cfg)
            elif isinstance(tool_cfg, dict):
                parsing_tools.append(ParsingTool(**tool_cfg))

        # Collect remaining top-level keys as other_config
        known_keys = {
            "datasets", "operations", "pipeline", "default_model",
            "parsing_tools", "rate_limits", "optimizer_config",
        }
        other = {k: v for k, v in config.items() if k not in known_keys}

        return cls(
            name=name or "pipeline",
            datasets=datasets,
            operations=operations,
            steps=steps,
            output=output,
            parsing_tools=parsing_tools,
            default_model=config.get("default_model"),
            rate_limits=config.get("rate_limits"),
            optimizer_config=config.get("optimizer_config", {}),
            **other,
        )

    def _load_env(self):
        import os

        from dotenv import load_dotenv

        # Get the current working directory
        cwd = os.getcwd()

        # Load .env file from the current working directory if it exists
        env_file = os.path.join(cwd, ".env")
        if os.path.exists(env_file):
            load_dotenv(env_file)

    def optimize(
        self,
        method: str = "moar",
        # MOAR parameters (used when method="moar")
        eval_fn: Any = None,
        metric_key: str | None = None,
        models: list[str] | None = None,
        agent_model: str | None = None,
        max_iterations: int = 20,
        save_dir: str | None = None,
        exploration_weight: float = 1.414,
        dataset_path: str | None = None,
        # V1 parameters (used when method="v1")
        max_threads: int | None = None,
        resume: bool = False,
        save_path: str | None = None,
    ) -> "MOARResult | Pipeline":
        """
        Optimize the pipeline.

        Args:
            method: ``"moar"`` (default) for multi-objective agentic rewrite
                optimization, or ``"v1"`` for the legacy single-pass optimizer.

        **MOAR parameters** (``method="moar"``):

        Args:
            eval_fn: A callable that scores pipeline output. Accepts
                ``(results_path) -> dict`` (1-arg) or
                ``(dataset_path, results_path) -> dict`` (2-arg).
                Also accepts a file path to a ``@register_eval``-decorated
                function. **Required** for MOAR.
            metric_key: Key to extract from the eval function's return dict.
                **Required** for MOAR.
            models: Model names to search over. ``None`` = auto-detect from
                environment API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.).
            agent_model: Model for the rewrite agent. ``None`` = auto-select.
            max_iterations: MCTS iterations (default 20).
            save_dir: Output directory. ``None`` = temp directory.
            exploration_weight: UCB exploration constant (default sqrt(2)).
            dataset_path: Explicit dataset path override.

        **V1 parameters** (``method="v1"``):

        Args:
            max_threads: Maximum threads for optimization.
            resume: Resume from a previous optimization state.
            save_path: Path to save the optimized pipeline.

        Returns:
            ``MOARResult`` when ``method="moar"``, ``Pipeline`` when ``method="v1"``.

        Examples::

            # MOAR optimization (recommended)
            def my_eval(results_path):
                import json
                with open(results_path) as f:
                    results = json.load(f)
                return {"score": sum(1 for r in results if r["correct"])}

            result = pipeline.optimize(eval_fn=my_eval, metric_key="score")

            # Each frontier point is a runnable pipeline
            best = result.best()
            best.run()

            # Inspect all explored plans as a DataFrame
            df = result.to_df()

            # V1 optimization (legacy)
            optimized = pipeline.optimize(method="v1")
            optimized.run()
        """
        if method == "moar":
            if eval_fn is None:
                raise ValueError(
                    "eval_fn is required for MOAR optimization. "
                    "Pass a callable, e.g.: "
                    "eval_fn=lambda results_path: {'score': compute_score(results_path)}"
                )
            if metric_key is None:
                raise ValueError(
                    "metric_key is required for MOAR optimization. "
                    "This is the key in your eval function's return dict to optimize."
                )

            from docetl.moar.optimizer import MOAROptimizer

            optimizer = MOAROptimizer(
                pipeline=self,
                eval_fn=eval_fn,
                metric_key=metric_key,
                models=models,
                agent_model=agent_model,
                max_iterations=max_iterations,
                save_dir=save_dir,
                exploration_weight=exploration_weight,
                dataset_path=dataset_path,
            )
            return optimizer.optimize()

        elif method == "v1":
            config = self._to_dict()
            runner = DSLRunner(
                config,
                base_name=os.path.join(os.getcwd(), self.name),
                yaml_file_suffix=self.name,
                max_threads=max_threads,
            )
            optimized_config, _ = runner.optimize(
                resume=resume,
                return_pipeline=False,
                save_path=save_path,
            )

            updated_pipeline = Pipeline(
                name=self.name,
                datasets=self.datasets,
                operations=self.operations,
                steps=self.steps,
                output=self.output,
                default_model=self.default_model,
                parsing_tools=self.parsing_tools,
                optimizer_config=self.optimizer_config,
            )
            updated_pipeline._update_from_dict(optimized_config)
            return updated_pipeline

        else:
            raise ValueError(
                f"Unknown optimization method {method!r}. Use 'moar' or 'v1'."
            )

    def run(self, max_threads: int | None = None) -> float:
        """
        Run the pipeline using the DSLRunner.

        Args:
            max_threads (int | None): Maximum number of threads to use for execution.

        Returns:
            float: The total cost of running the pipeline.
        """
        runner = DSLRunner(
            self,
            base_name=os.path.join(os.getcwd(), self.name),
            yaml_file_suffix=self.name,
            max_threads=max_threads,
        )
        result = runner.load_run_save()
        return result

    def run_with_stats(self, max_threads: int | None = None) -> dict[str, Any]:
        """
        Run the pipeline and return detailed execution statistics.

        Args:
            max_threads (int | None): Maximum number of threads to use for execution.

        Returns:
            dict[str, Any]: A dictionary containing:
                - cost (float): The total cost of running the pipeline.
                - token_usage (dict[str, dict[str, int]]): Token usage broken down
                  by model, each with "prompt_tokens" and "completion_tokens".

        Example:
            ```python
            stats = pipeline.run_with_stats()
            print(f"Cost: ${stats['cost']:.2f}")
            for model, usage in stats['token_usage'].items():
                print(f"{model}: {usage['prompt_tokens']} in, {usage['completion_tokens']} out")
            ```
        """
        runner = DSLRunner(
            self,
            base_name=os.path.join(os.getcwd(), self.name),
            yaml_file_suffix=self.name,
            max_threads=max_threads,
        )
        runner.load_run_save()
        return {
            "cost": runner.total_cost,
            "token_usage": dict(runner.total_token_usage),
        }

    def to_yaml(self, path: str) -> None:
        """
        Convert the Pipeline object to a YAML string and save it to a file.

        Args:
            path (str): Path to save the YAML file.

        Returns:
            None
        """
        config = self._to_dict()
        with open(path, "w") as f:
            yaml.safe_dump(config, f)

        print(f"[green]Pipeline saved to {path}[/green]")

    def _to_dict(self) -> dict[str, Any]:
        """
        Convert the Pipeline object to a dictionary representation.

        Returns:
            dict[str, Any]: Dictionary representation of the Pipeline.
        """
        d = {
            "datasets": {
                name: dataset.dict() for name, dataset in self.datasets.items()
            },
            "operations": [
                {k: v for k, v in op.dict().items() if v is not None}
                for op in self.operations
            ],
            "pipeline": {
                "steps": [
                    {k: v for k, v in step.dict().items() if v is not None}
                    for step in self.steps
                ],
                "output": self.output.dict(),
            },
            "default_model": self.default_model,
            "parsing_tools": (
                [tool.dict() for tool in self.parsing_tools]
                if self.parsing_tools
                else None
            ),
            "optimizer_config": self.optimizer_config,
            **self.other_config,
        }
        if self.rate_limits:
            d["rate_limits"] = self.rate_limits
        return d

    def _update_from_dict(self, config: dict[str, Any]):
        """
        Update this Pipeline's fields from a raw config dict.

        Delegates to ``from_dict`` so the type-dispatch logic lives in one place.
        """
        other = Pipeline.from_dict(config, name=self.name)
        self.datasets = other.datasets
        self.operations = other.operations
        self.steps = other.steps
        self.output = other.output
        self.default_model = other.default_model
        self.parsing_tools = other.parsing_tools
        self.optimizer_config = other.optimizer_config
        self.other_config = other.other_config


# Export the main classes and functions for easy import
__all__ = [
    "Pipeline",
    "Dataset",
    "MapOp",
    "ResolveOp",
    "ReduceOp",
    "ParallelMapOp",
    "FilterOp",
    "EquijoinOp",
    "SplitOp",
    "GatherOp",
    "UnnestOp",
    "CodeMapOp",
    "CodeReduceOp",
    "CodeFilterOp",
    "ExtractOp",
    "PipelineStep",
    "PipelineOutput",
    "ParsingTool",
]
