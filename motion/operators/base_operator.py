from abc import ABC, abstractmethod
from typing import Optional, Any, Tuple, Dict
from motion.types import ValidatorAction


class Operator(ABC):
    on_fail: ValidatorAction = ValidatorAction.WARN

    def get_description(self) -> Optional[str]:
        return None  # Default implementation returns None

    @abstractmethod
    def execute(self, key: Any, value: Any) -> Tuple[Any, Dict]:
        pass
