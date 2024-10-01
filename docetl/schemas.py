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
    name: str
    function_code: str


class Dataset(BaseModel):
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
    name: str
    operations: List[Union[Dict[str, Any], str]]
    input: Optional[str] = None


class PipelineOutput(BaseModel):
    type: str
    path: str
    intermediate_dir: Optional[str] = None


class Pipeline(BaseModel):
    name: str
    datasets: Dict[str, Dataset]
    operations: List[OpType]
    steps: List[PipelineStep]
    output: PipelineOutput
    parsing_tools: List[ParsingTool] = []
    default_model: Optional[str] = None
