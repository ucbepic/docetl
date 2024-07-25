from enum import Enum
from typing import TypeVar, List, Any, Dict, Tuple, Optional, Set, Union, Iterable
from collections import namedtuple

K = TypeVar("K")
V = TypeVar("V")
RK = TypeVar("RK")
RV = TypeVar("RV")


class ValidatorAction(Enum):
    PROMPT = "prompt"
    WARN = "warn"
    FAIL = "fail"


OpOutput = namedtuple("OpOutput", ["id", "prompt", "response", "new_key", "new_value"])

OpFlatOutput = namedtuple(
    "OpFlatOutput", ["id", "prompt", "response", "new_key_value_pairs"]
)

OpParallelFlatOutput = namedtuple(
    "OpParallelFlatOutput", ["id", "prompts", "responses", "new_key_value_pairs"]
)

OpFilterOutput = namedtuple(
    "OpFilterOutput", ["id", "prompt", "response", "new_key", "new_value", "filter"]
)

OpError = namedtuple(
    "OpError",
    [
        "id",
        "old_key",
        "old_value",
        "prompt",
        "response",
        "new_key",
        "new_value",
        "error_msg",
    ],
)

OpFlatError = namedtuple(
    "OpFlatError",
    [
        "id",
        "old_key",
        "old_value",
        "prompt",
        "response",
        "new_key_value_pairs",
        "error_msg",
    ],
)

OpParallelFlatError = namedtuple(
    "OpParallelFlatError",
    [
        "id",
        "old_key",
        "old_value",
        "prompts",
        "responses",
        "new_key_value_pairs",
        "error_msg",
    ],
)

OpReduceError = namedtuple(
    "OpReduceError",
    ["id", "old_key", "old_values", "prompt", "response", "new_value", "error_msg"],
)

OpInput = namedtuple("OpInput", ["key", "value"])

OpReduceInput = namedtuple("OpReduceInput", ["key", "values"])

CorrectedOutput = namedtuple("CorrectedOutput", ["id", "key", "value"])
