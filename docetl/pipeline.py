import os
from docetl.utils import load_config
from typing import Any, Dict, List, Optional, Tuple, Union
from .operations.utils import APIWrapper
from rich.console import Console

class Pipeline(object):
    @classmethod
    def from_yaml(cls, yaml_file: str, **kwargs):
        config = load_config(yaml_file)
        return cls(config, **kwargs)

    def __init__(
        self,
        config: Dict,
        max_threads: int = None):
        
        self.config = config
        self.default_model = self.config.get("default_model", "gpt-4o-mini")
        self.console = Console()
        self.max_threads = max_threads or (os.cpu_count() or 1) * 4
        self.status = None
        self.api = APIWrapper(self)
