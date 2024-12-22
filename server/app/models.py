from pydantic import BaseModel


class PipelineRequest(BaseModel):
    yaml_config: str

class PipelineConfigRequest(BaseModel):
    namespace: str
    name: str
    config: str
    input_path: str
    output_path: str
