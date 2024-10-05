from docetl.utils import load_config
from typing import Any, Dict, List, Optional, Tuple, Union


class Pipeline(object):
    @classmethod
    def from_yaml(cls, yaml_file: str, **kwargs):
        config = load_config(yaml_file)
        return cls(config, **kwargs)

    def __init__(
        self,
        config: Dict):
        
        self.config = config
