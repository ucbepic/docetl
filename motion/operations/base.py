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
