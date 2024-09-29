"""
This module defines the core data structures and classes for the DocETL pipeline.

It includes dataclasses for various operation types, pipeline steps, and the main Pipeline class.
The module provides a high-level API for defining, optimizing, and running document processing pipelines.

Classes:
    Dataset: Represents a dataset with a type and path.
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
        datasets={"input": Dataset(type="file", path="input.json")},
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

from dataclasses import dataclass
import os
from typing import List, Optional, Dict, Any, Union

import yaml

from docetl.builder import Optimizer
from docetl.runner import DSLRunner

from rich import print


@dataclass
class Dataset:
    type: str
    path: str


@dataclass
class BaseOp:
    name: str
    type: str


@dataclass
class MapOp(BaseOp):
    output: Optional[Dict[str, Any]] = None
    prompt: Optional[str] = None
    model: Optional[str] = None
    optimize: Optional[bool] = None
    recursively_optimize: Optional[bool] = None
    sample_size: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    validate: Optional[List[str]] = None
    num_retries_on_validate_failure: Optional[int] = None
    gleaning: Optional[Dict[str, Any]] = None
    drop_keys: Optional[List[str]] = None
    timeout: Optional[int] = None


@dataclass
class ResolveOp(BaseOp):
    comparison_prompt: str
    resolution_prompt: str
    output: Optional[Dict[str, Any]] = None
    embedding_model: Optional[str] = None
    resolution_model: Optional[str] = None
    comparison_model: Optional[str] = None
    blocking_keys: Optional[List[str]] = None
    blocking_threshold: Optional[float] = None
    blocking_conditions: Optional[List[str]] = None
    input: Optional[Dict[str, Any]] = None
    embedding_batch_size: Optional[int] = None
    compare_batch_size: Optional[int] = None
    limit_comparisons: Optional[int] = None
    optimize: Optional[bool] = None
    timeout: Optional[int] = None


@dataclass
class ReduceOp(BaseOp):
    reduce_key: Union[str, List[str]]
    output: Optional[Dict[str, Any]] = None
    prompt: Optional[str] = None
    optimize: Optional[bool] = None
    synthesize_resolve: Optional[bool] = None
    model: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    pass_through: Optional[bool] = None
    associative: Optional[bool] = None
    fold_prompt: Optional[str] = None
    fold_batch_size: Optional[int] = None
    value_sampling: Optional[Dict[str, Any]] = None
    verbose: Optional[bool] = None
    timeout: Optional[int] = None


@dataclass
class ParallelMapOp(BaseOp):
    prompts: List[Dict[str, Any]]
    output: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    optimize: Optional[bool] = None
    recursively_optimize: Optional[bool] = None
    sample_size: Optional[int] = None
    drop_keys: Optional[List[str]] = None
    timeout: Optional[int] = None


@dataclass
class FilterOp(BaseOp):
    output: Optional[Dict[str, Any]] = None
    prompt: Optional[str] = None
    model: Optional[str] = None
    optimize: Optional[bool] = None
    recursively_optimize: Optional[bool] = None
    sample_size: Optional[int] = None
    validate: Optional[List[str]] = None
    num_retries_on_validate_failure: Optional[int] = None
    timeout: Optional[int] = None


@dataclass
class EquijoinOp(BaseOp):
    left: str
    right: str
    comparison_prompt: str
    output: Optional[Dict[str, Any]] = None
    blocking_threshold: Optional[float] = None
    blocking_conditions: Optional[Dict[str, List[str]]] = None
    limits: Optional[Dict[str, int]] = None
    comparison_model: Optional[str] = None
    optimize: Optional[bool] = None
    embedding_model: Optional[str] = None
    embedding_batch_size: Optional[int] = None
    compare_batch_size: Optional[int] = None
    limit_comparisons: Optional[int] = None
    blocking_keys: Optional[Dict[str, List[str]]] = None
    timeout: Optional[int] = None


@dataclass
class SplitOp(BaseOp):
    split_key: str
    method: str
    method_kwargs: Dict[str, Any]
    model: Optional[str] = None


@dataclass
class GatherOp(BaseOp):
    content_key: str
    doc_id_key: str
    order_key: str
    peripheral_chunks: Dict[str, Any]
    doc_header_key: Optional[str] = None


@dataclass
class UnnestOp(BaseOp):
    unnest_key: str
    keep_empty: Optional[bool] = None
    expand_fields: Optional[List[str]] = None
    recursive: Optional[bool] = None
    depth: Optional[int] = None


OpType = Union[
    MapOp,
    ResolveOp,
    ReduceOp,
    ParallelMapOp,
    FilterOp,
    EquijoinOp,
    SplitOp,
    GatherOp,
    UnnestOp,
]


@dataclass
class PipelineStep:
    name: str
    operations: List[Union[Dict[str, Any], str]]
    input: Optional[str] = None


@dataclass
class PipelineOutput:
    type: str
    path: str
    intermediate_dir: Optional[str] = None


@dataclass
class Pipeline:
    name: str
    datasets: Dict[str, Dataset]
    operations: List[OpType]
    steps: List[PipelineStep]
    output: PipelineOutput
    default_model: Optional[str] = None

    def optimize(
        self,
        max_threads: Optional[int] = None,
        model: str = "gpt-4o",
        resume: bool = False,
        timeout: int = 60,
    ) -> "Pipeline":
        """
        Optimize the pipeline using the Optimizer.

        Args:
            max_threads (Optional[int]): Maximum number of threads to use for optimization.
            model (str): The model to use for optimization. Defaults to "gpt-4o".
            resume (bool): Whether to resume optimization from a previous state. Defaults to False.
            timeout (int): Timeout for optimization in seconds. Defaults to 60.

        Returns:
            Pipeline: An optimized version of the pipeline.
        """
        config = self._to_dict()
        optimizer = Optimizer(
            config,
            base_name=os.path.join(os.getcwd(), self.name),
            yaml_file_suffix=self.name,
            max_threads=max_threads,
            model=model,
            timeout=timeout,
            resume=resume,
        )
        optimizer.optimize()
        optimized_config = optimizer.clean_optimized_config()

        updated_pipeline = Pipeline(
            name=self.name,
            datasets=self.datasets,
            operations=self.operations,
            steps=self.steps,
            output=self.output,
            default_model=self.default_model,
        )
        updated_pipeline._update_from_dict(optimized_config)
        return updated_pipeline

    def run(self, max_threads: Optional[int] = None) -> float:
        """
        Run the pipeline using the DSLRunner.

        Args:
            max_threads (Optional[int]): Maximum number of threads to use for execution.

        Returns:
            float: The total cost of running the pipeline.
        """
        config = self._to_dict()
        runner = DSLRunner(config, max_threads=max_threads)
        result = runner.run()
        return result

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

    def _to_dict(self) -> Dict[str, Any]:
        """
        Convert the Pipeline object to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representation of the Pipeline.
        """
        return {
            "datasets": {
                name: dataset.__dict__ for name, dataset in self.datasets.items()
            },
            "operations": [
                {k: v for k, v in op.__dict__.items() if v is not None}
                for op in self.operations
            ],
            "pipeline": {
                "steps": [
                    {k: v for k, v in step.__dict__.items() if v is not None}
                    for step in self.steps
                ],
                "output": self.output.__dict__,
            },
            "default_model": self.default_model,
        }

    def _update_from_dict(self, config: Dict[str, Any]):
        """
        Update the Pipeline object from a dictionary representation.

        Args:
            config (Dict[str, Any]): Dictionary representation of the Pipeline.
        """
        self.datasets = {
            name: Dataset(**dataset) for name, dataset in config["datasets"].items()
        }
        self.operations = []
        for op in config["operations"]:
            op_type = op.pop("type")
            if op_type == "map":
                self.operations.append(MapOp(**op, type=op_type))
            elif op_type == "resolve":
                self.operations.append(ResolveOp(**op, type=op_type))
            elif op_type == "reduce":
                self.operations.append(ReduceOp(**op, type=op_type))
            elif op_type == "parallel_map":
                self.operations.append(ParallelMapOp(**op, type=op_type))
            elif op_type == "filter":
                self.operations.append(FilterOp(**op, type=op_type))
            elif op_type == "equijoin":
                self.operations.append(EquijoinOp(**op, type=op_type))
            elif op_type == "split":
                self.operations.append(SplitOp(**op, type=op_type))
            elif op_type == "gather":
                self.operations.append(GatherOp(**op, type=op_type))
            elif op_type == "unnest":
                self.operations.append(UnnestOp(**op, type=op_type))
        self.steps = [PipelineStep(**step) for step in config["pipeline"]["steps"]]
        self.output = PipelineOutput(**config["pipeline"]["output"])
        self.default_model = config.get("default_model")
