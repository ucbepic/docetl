import pytest
from typing import List, Tuple, Any, Set
import random

from motion import (
    Dataset,
    ValidatorAction,
    Mapper,
    Reducer,
    Equality,
)


class IdentityMapper(Mapper[float, int]):
    on_fail = ValidatorAction.WARN

    def map(self, key: Any, value: int) -> List[Tuple[float, int]]:
        return [(float(value), value)]

    def validate(self, key: float, value: int) -> bool:
        if random.random() < 0.2:
            return False
        return 0 <= key <= 10 and 1 <= value <= 100

    def get_description(self) -> str:
        return "Identity Mapper: Convert value to float key"


class SquareMapper(Mapper[float, int]):
    on_fail = ValidatorAction.WARN

    def map(self, key: float, value: int) -> List[Tuple[float, int]]:
        return [(key, value**2)]

    def validate(self, key: float, value: int) -> bool:
        return value >= 0

    def get_description(self) -> str:
        return "Square Mapper: Square the value"


class SumReducer(Reducer[float, int]):
    on_fail = ValidatorAction.WARN

    def reduce(self, key: float, values: List[int]) -> int:
        return sum(values)

    def validate(self, key: float, reduced_value: int) -> bool:
        return 1 <= reduced_value <= 1000000  # Increased upper bound due to squaring

    def get_description(self) -> str:
        return "Sum Reducer: Sum all values"


class FuzzyEquality(Equality[float]):
    def __init__(self, tolerance: float):
        self.tolerance = tolerance

    def precheck(self, x: float, y: float) -> bool:
        return abs(x - y) <= 2 * self.tolerance

    def are_equal(self, x: float, y: float) -> bool:
        return abs(x - y) <= self.tolerance

    def get_label(self, keys: Set[float]) -> float:
        min_key = min(keys)
        max_key = max(keys)
        return sum(keys) / len(keys)

    def get_description(self) -> str:
        return f"Fuzzy Equality: Group keys within {self.tolerance} tolerance"


if __name__ == "__main__":
    input_data = range(1, 10)  # Numbers from 1 to 9

    dataset = Dataset(input_data, enable_tracing=True)
    result = (
        dataset.map(IdentityMapper())
        .reduce(SumReducer(), FuzzyEquality(tolerance=2))
        .map(SquareMapper())
        .reduce(SumReducer(), FuzzyEquality(tolerance=5))
        .execute()
    )

    # Print results in a more readable format
    for record_id, key, value, trace in sorted(result, key=lambda x: x[1]):
        print(f"Record ID: {record_id}")
        print(f"Group {key}: Final Result = {value}")
        print("  Trace:")
        for op, description in trace.items():
            print(f"    {op}: {description}")
        print()
