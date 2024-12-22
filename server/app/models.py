from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class PipelineRequest(BaseModel):
    yaml_config: str

class PipelineConfigRequest(BaseModel):
    namespace: str
    name: str
    config: str
    input_path: str
    output_path: str

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class OptimizeResult(BaseModel):
    task_id: str
    status: TaskStatus
    should_optimize: Optional[str] = None
    input_data: Optional[List[Dict[str, Any]]] = None
    output_data: Optional[List[Dict[str, Any]]] = None
    cost: Optional[float] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

class OptimizeRequest(BaseModel):
    yaml_config: str
    step_name: str
    op_name: str