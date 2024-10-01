from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class ToolFunction(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class Tool(BaseModel):
    code: str
    function: ToolFunction


class ParsingTool(BaseModel):
    """
    Represents a parsing tool used for custom data parsing in the pipeline.

    Attributes:
        name (str): The name of the parsing tool. This should be unique within the pipeline configuration.
        function_code (str): The Python code defining the parsing function. This code will be executed
                             to parse the input data according to the specified logic. It should return a list of strings, where each string is its own document.

    Example:
        ```yaml
        parsing_tools:
          - name: ocr_parser
            function_code: |
              import pytesseract
              from pdf2image import convert_from_path
              def ocr_parser(filename: str) -> List[str]:
                  images = convert_from_path(filename)
                  text = ""
                  for image in images:
                      text += pytesseract.image_to_string(image)
                  return [text]
        ```
    """

    name: str
    function_code: str


class Dataset(BaseModel):
    """
    Represents a dataset configuration in the pipeline.

    Attributes:
        type (str): The type of the dataset. Must be either 'file' or 'memory'.
        path (str): The path to the dataset file or the in-memory data, depending on the type.
        source (str): The source of the dataset. Currently, only 'local' is supported. Defaults to 'local'.
        parsing (Optional[List[Dict[str, str]]]): A list of parsing tools to apply to the data. Each parsing tool
                                                  is represented by a dictionary with 'input_key', 'function', and
                                                  'output_key' keys. Defaults to None.

    Example:
        ```yaml
        datasets:
          my_dataset:
            type: file
            path: input.json
            parsing:
              - input_key: file_path
                function: txt_to_string
                output_key: content
        ```

    Note:
        The parsing tools are applied in the order they are listed. Each parsing tool takes the output
        of the previous tool as its input, allowing for chained processing of the data.
    """

    type: str
    path: str
    source: str = "local"
    parsing: Optional[List[Dict[str, str]]] = None


class BaseOp(BaseModel):
    name: str
    type: str


class MapOp(BaseOp):
    type: str = "map"
    output: Optional[Dict[str, Any]] = None
    prompt: Optional[str] = None
    model: Optional[str] = None
    optimize: Optional[bool] = None
    recursively_optimize: Optional[bool] = None
    sample_size: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    validation_rules: Optional[List[str]] = Field(None, alias="validate")
    num_retries_on_validate_failure: Optional[int] = None
    gleaning: Optional[Dict[str, Any]] = None
    drop_keys: Optional[List[str]] = None
    timeout: Optional[int] = None
    batch_size: Optional[int] = None
    clustering_method: Optional[str] = None

    @field_validator("drop_keys")
    def validate_drop_keys(cls, v):
        if isinstance(v, str):
            return [v]
        return v


class ResolveOp(BaseOp):
    type: str = "resolve"
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


class ReduceOp(BaseOp):
    type: str = "reduce"
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


class ParallelMapOp(BaseOp):
    type: str = "parallel_map"
    prompts: List[Dict[str, Any]]
    output: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    optimize: Optional[bool] = None
    recursively_optimize: Optional[bool] = None
    sample_size: Optional[int] = None
    drop_keys: Optional[List[str]] = None
    timeout: Optional[int] = None
    batch_size: Optional[int] = None
    clustering_method: Optional[str] = None

    @field_validator("drop_keys")
    def validate_drop_keys(cls, v):
        if isinstance(v, str):
            return [v]
        return v


class FilterOp(BaseOp):
    type: str = "filter"
    output: Optional[Dict[str, Any]] = None
    prompt: Optional[str] = None
    model: Optional[str] = None
    optimize: Optional[bool] = None
    recursively_optimize: Optional[bool] = None
    sample_size: Optional[int] = None
    validation_rules: Optional[List[str]] = Field(None, alias="validate")
    num_retries_on_validate_failure: Optional[int] = None
    timeout: Optional[int] = None


class EquijoinOp(BaseOp):
    type: str = "equijoin"
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


class SplitOp(BaseOp):
    type: str = "split"
    split_key: str
    method: str
    method_kwargs: Dict[str, Any]
    model: Optional[str] = None


class GatherOp(BaseOp):
    type: str = "gather"
    content_key: str
    doc_id_key: str
    order_key: str
    peripheral_chunks: Dict[str, Any]
    doc_header_key: Optional[str] = None


class UnnestOp(BaseOp):
    type: str = "unnest"
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


class PipelineStep(BaseModel):
    """
    Represents a step in the pipeline.

    Attributes:
        name (str): The name of the step.
        operations (List[Union[Dict[str, Any], str]]): A list of operations to be applied in this step.
            Each operation can be either a string (the name of the operation) or a dictionary
            (for more complex configurations).
        input (Optional[str]): The input for this step. It can be either the name of a dataset
            or the name of a previous step. If not provided, the step will use the output
            of the previous step as its input.

    Example:
        ```python
        # Simple step with a single operation
        process_step = PipelineStep(
            name="process_step",
            input="my_dataset",
            operations=["process"]
        )

        # Step with multiple operations
        summarize_step = PipelineStep(
            name="summarize_step",
            input="process_step",
            operations=["summarize"]
        )

        # Step with a more complex operation configuration
        custom_step = PipelineStep(
            name="custom_step",
            input="previous_step",
            operations=[
                {
                    "custom_operation": {
                        "model": "gpt-4",
                        "prompt": "Perform a custom analysis on the following text:"
                    }
                }
            ]
        )
        ```

    These examples show different ways to configure pipeline steps, from simple
    single-operation steps to more complex configurations with custom parameters.
    """

    name: str
    operations: List[Union[Dict[str, Any], str]]
    input: Optional[str] = None


class PipelineOutput(BaseModel):
    """
    Represents the output configuration for a pipeline.

    Attributes:
        type (str): The type of output. This could be 'file', 'database', etc.
        path (str): The path where the output will be stored. This could be a file path,
                    database connection string, etc., depending on the type.
        intermediate_dir (Optional[str]): The directory to store intermediate results,
                                          if applicable. Defaults to None.

    Example:
        ```python
        output = PipelineOutput(
            type="file",
            path="/path/to/output.json",
            intermediate_dir="/path/to/intermediate/results"
        )
        ```
    """

    type: str
    path: str
    intermediate_dir: Optional[str] = None


class Pipeline(BaseModel):
    """
    Represents a complete document processing pipeline.

    Attributes:
        name (str): The name of the pipeline.
        datasets (Dict[str, Dataset]): A dictionary of datasets used in the pipeline,
                                       where keys are dataset names and values are Dataset objects.
        operations (List[OpType]): A list of operations to be performed in the pipeline.
        steps (List[PipelineStep]): A list of steps that make up the pipeline.
        output (PipelineOutput): The output configuration for the pipeline.
        parsing_tools (List[ParsingTool]): A list of parsing tools used in the pipeline.
                                           Defaults to an empty list.
        default_model (Optional[str]): The default language model to use for operations
                                       that require one. Defaults to None.

    Example:
        ```python
        pipeline = Pipeline(
            name="document_processing_pipeline",
            datasets={
                "input_data": Dataset(type="file", path="/path/to/input.json")
            },
            operations=[
                MapOp(
                    name="process",
                    type="map",
                    prompt="Determine what type of document this is: {{ input.content }}",
                    output={"schema": {"document_type": "string"}}
                ),
                ReduceOp(
                    name="summarize",
                    type="reduce",
                    reduce_key="document_type",
                    prompt="Summarize the processed contents: {% for item in inputs %}{{ item.content }} {% endfor %}",
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

    name: str
    datasets: Dict[str, Dataset]
    operations: List[OpType]
    steps: List[PipelineStep]
    output: PipelineOutput
    parsing_tools: List[ParsingTool] = []
    default_model: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        self._load_env()

    def _load_env(self):
        from dotenv import load_dotenv
        import os

        # Get the current working directory
        cwd = os.getcwd()

        # Load .env file from the current working directory if it exists
        env_file = os.path.join(cwd, ".env")
        if os.path.exists(env_file):
            load_dotenv(env_file)
