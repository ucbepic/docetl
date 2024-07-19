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


class TrueEquality(Equality[str]):
    def precheck(self, x, y) -> bool:
        return True

    def are_equal(self, x, y) -> bool:
        return True

    def get_label(self, keys) -> str:
        return list(keys)[0]


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


@pytest.fixture
def input_data():
    return list(range(1, 10))  # Numbers from 1 to 9


@pytest.fixture
def keyed_data():
    return [(1, elem) for elem in list(range(1, 10))]


def test_identity_mapper():
    mapper = IdentityMapper()
    result = mapper.map(None, 5)
    assert result == [(5.0, 5)]


def test_square_mapper():
    mapper = SquareMapper()
    result = mapper.map(2.0, 5)
    assert result == [(2.0, 25)]


def test_sum_reducer():
    reducer = SumReducer()
    result = reducer.reduce(1.0, [1, 2, 3, 4, 5])
    assert result == 15


def test_fuzzy_equality():
    equality = FuzzyEquality(tolerance=1.0)
    assert equality.are_equal(1.0, 1.5)
    assert not equality.are_equal(1.0, 2.5)


def test_dataset_map(input_data):
    dataset = Dataset(input_data)
    result = dataset.map(IdentityMapper()).execute()
    assert len(result) == len(input_data)
    for _, key, value, _ in result:
        assert isinstance(key, float)
        assert isinstance(value, int)


def test_dataset_reduce(input_data):
    dataset = Dataset(input_data)
    result = dataset.reduce(SumReducer(), TrueEquality()).execute()
    assert len(result) == 1  # All inputs should be reduced to one group
    _, key, value, _ = result[0]
    assert isinstance(key, str)
    assert value == sum(input_data)


def test_dataset_map_reduce(input_data):
    dataset = Dataset(input_data)
    result = (
        dataset.map(IdentityMapper())
        .reduce(SumReducer(), FuzzyEquality(tolerance=2))
        .map(SquareMapper())
        .reduce(SumReducer(), FuzzyEquality(tolerance=5))
        .execute()
    )
    assert len(result) == 2
    _, _, value, _ = result[0]
    expected_value = 261
    assert value == expected_value


def test_tracing_enabled(input_data):
    dataset = Dataset(input_data, enable_tracing=True)
    result = (
        dataset.map(IdentityMapper())
        .reduce(SumReducer(), FuzzyEquality(tolerance=2))
        .execute()
    )
    _, _, _, trace = result[0]
    assert trace is not None
    assert "op_0" in trace
    assert "op_1" in trace


def test_tracing_disabled(input_data):
    dataset = Dataset(input_data, enable_tracing=False)
    result = (
        dataset.map(IdentityMapper())
        .reduce(SumReducer(), FuzzyEquality(tolerance=2))
        .execute()
    )
    _, _, _, trace = result[0]
    assert trace is None


class AlwaysFailMapper(Mapper[int, int]):
    on_fail = ValidatorAction.WARN

    def map(self, key: Any, value: int) -> List[Tuple[int, int]]:
        return [(value, value)]

    def validate(self, key: int, value: int) -> bool:
        return False

    def get_description(self) -> str:
        return "Always Fail Mapper"


def test_validation_warning(input_data, capsys):

    dataset = Dataset(input_data)
    dataset.map(AlwaysFailMapper()).execute()
    captured = capsys.readouterr()
    assert "Warning: Validation failed" in captured.err


class PromptMapper(Mapper[int, int]):
    on_fail = ValidatorAction.PROMPT

    def map(self, key: Any, value: int) -> List[Tuple[int, int]]:
        return [(value, value)]

    def validate(self, key: int, value: int) -> bool:
        return False

    def get_description(self) -> str:
        return "Prompt Mapper"


# TODO: fix this
# def test_validation_prompt(input_data, monkeypatch):
#     monkeypatch.setattr("builtins.input", lambda _: '{"some_id": 42}')
#     dataset = Dataset(input_data)
#     result = dataset.map(PromptMapper()).execute()
#     assert any(value == 42 for _, _, value, _ in result)


class FailMapper(Mapper[int, int]):
    on_fail = ValidatorAction.FAIL

    def map(self, key: Any, value: int) -> List[Tuple[int, int]]:
        return [(value, value)]

    def validate(self, key: int, value: int) -> bool:
        return False

    def get_description(self) -> str:
        return "Fail Mapper"


def test_validation_fail(input_data):

    dataset = Dataset(input_data)
    with pytest.raises(ValueError):
        dataset.map(FailMapper()).execute()
