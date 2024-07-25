from abc import ABC, abstractmethod
from typing import Optional, Any, Tuple, Dict
from motion.types import ValidatorAction
from functools import wraps


def build_only(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.is_build_phase:
            return func(self, *args, **kwargs)
        return True  # Skip validation if not in build phase

    return wrapper


class Operator(ABC):
    on_fail: ValidatorAction = ValidatorAction.WARN

    def __init__(self):
        self._is_build_phase: bool = False

    def get_description(self) -> Optional[str]:
        return None  # Default implementation returns None

    @abstractmethod
    def execute(self, key: Any, value: Any) -> Tuple[Any, Dict]:
        pass

    def validate(self, **kwargs) -> None:
        # Return None if validation is successful, otherwise raise an error
        pass

    def set_build_phase(self, is_build: bool):
        self._is_build_phase = is_build

    @property
    def is_build_phase(self) -> bool:
        return self._is_build_phase
