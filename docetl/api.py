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
                parsing_tools=[{"name": "txt_to_string", "input_key": "text", "output_key": "content"}]
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

import os
from typing import Any, Dict, List, Optional

import yaml
from rich import print

from docetl.builder import Optimizer
from docetl.runner import DSLRunner
from docetl.schemas import Dataset, EquijoinOp, FilterOp, GatherOp, MapOp, ParallelMapOp
from docetl.schemas import Pipeline as PipelineModel
from docetl.schemas import (
    PipelineOutput,
    PipelineStep,
    ReduceOp,
    ResolveOp,
    SplitOp,
    UnnestOp,
    ParsingTool,
)


class Pipeline(PipelineModel):
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
            parsing_tools=self.parsing_tools,
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
        self.parsing_tools = (
            [ParsingTool(**tool) for tool in config.get("parsing_tools", [])]
            if config.get("parsing_tools")
            else []
        )
