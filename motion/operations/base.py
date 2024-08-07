from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional
from rich.console import Console


class BaseOperation(ABC):
    def __init__(
        self,
        config: Dict,
        default_model: str,
        max_threads: int,
        console: Optional[Console] = None,
    ):
        self.config = config
        self.default_model = default_model
        self.max_threads = max_threads
        self.console = console or Console()
        self.syntax_check()

    @abstractmethod
    def execute(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        pass

    @abstractmethod
    def syntax_check(self) -> None:
        pass

    def gleaning_check(self) -> None:
        if "gleaning" in self.config:
            if "num_rounds" not in self.config["gleaning"]:
                raise ValueError("Missing 'num_rounds' in 'gleaning' configuration")
            if not isinstance(self.config["gleaning"]["num_rounds"], int):
                raise TypeError(
                    "'num_rounds' in 'gleaning' configuration must be an integer"
                )
            if self.config["gleaning"]["num_rounds"] < 1:
                raise ValueError(
                    "'num_rounds' in 'gleaning' configuration must be at least 1"
                )

            if "validation_prompt" not in self.config["gleaning"]:
                raise ValueError(
                    "Missing 'validation_prompt' in 'gleaning' configuration"
                )
            if not isinstance(self.config["gleaning"]["validation_prompt"], str):
                raise TypeError(
                    "'validation_prompt' in 'gleaning' configuration must be a string"
                )
            if not self.config["gleaning"]["validation_prompt"].strip():
                raise ValueError(
                    "'validation_prompt' in 'gleaning' configuration cannot be empty"
                )
