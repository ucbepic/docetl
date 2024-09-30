from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, field_validator


class ToolFunction(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class Tool(BaseModel):
    code: str
    function: ToolFunction


class OutputSchema(BaseModel):
    schema: Dict[str, str]


class MapOperationConfig(BaseModel):
    drop_keys: Optional[List[str]] = None
    prompt: Optional[str] = None
    output: Optional[OutputSchema] = None
    model: Optional[str] = None
    tools: Optional[List[Tool]] = None

    @field_validator("drop_keys")
    def validate_drop_keys(cls, v):
        if isinstance(v, str):
            return [v]
        return v


class ParallelMapOperationConfig(BaseModel):
    prompts: List[MapOperationConfig]
    model: Optional[str] = None
    tools: Optional[List[Tool]] = None


class BatchConfig(BaseModel):
    batch_size: int
    clustering_method: Optional[str] = None


class OperationConfig(BaseModel):
    name: str
    type: str
    config: Union[MapOperationConfig, ParallelMapOperationConfig, BatchConfig]
