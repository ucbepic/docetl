from enum import Enum
from typing import TypeVar, List, Any, Dict, Tuple, Optional, Set, Union, Iterable

K = TypeVar("K")
V = TypeVar("V")
R = TypeVar("R")


class ValidatorAction(Enum):
    PROMPT = "prompt"
    WARN = "warn"
    FAIL = "fail"
