from typing import Union
from motion.operators import (
    LLMMapper,
    LLMReducer,
    KeyResolver,
    LLMFlatMapper,
    LLMFilterer,
    Splitter,
    LLMParallelFlatMapper,
)


class Operation:
    def __init__(
        self,
        operator: Union[
            LLMMapper,
            LLMReducer,
            KeyResolver,
            LLMFlatMapper,
            LLMFilterer,
            Splitter,
            LLMParallelFlatMapper,
        ],
    ):
        self.operator = operator
        self._is_optimized = False

    @property
    def is_optimized(self) -> bool:
        return self._is_optimized

    @is_optimized.setter
    def is_optimized(self, value: bool) -> None:
        self._is_optimized = value
