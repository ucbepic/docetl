import pytest
from typing import List, Tuple, Set, Callable
from motion.dataset import Dataset
from motion.operators import (
    Mapper,
    Reducer,
    KeyResolver,
    Filterer,
    FlatMapper,
    ParallelFlatMapper,
)
from motion.types import ValidatorAction


# Test Mapper
class TestMapper(Mapper):
    def map(self, key: str, value: int) -> Tuple[str, int]:
        return (key, value * 2)

    def validate(
        self, input_key: str, input_value: int, output_key: str, output_value: int
    ) -> bool:
        return output_value == input_value * 2


def test_mapper():
    data = [("a", 1), ("b", 2), ("c", 3)]
    dataset = Dataset(data)
    result = dataset.map(TestMapper()).execute()
    assert result == [("a", 2), ("b", 4), ("c", 6)]


# Test FlatMapper
class TestFlatMapper(FlatMapper):
    def map(self, key: str, value: int) -> List[Tuple[str, int]]:
        return [(key, value), (key, value + 1)]

    def validate(
        self,
        key: str,
        value: int,
        output_kv_pairs: List[Tuple[str, int]],
    ) -> bool:
        return len(output_kv_pairs) == 2 and output_kv_pairs[1][1] == value + 1


def test_flatmapper():
    data = [("a", 1), ("b", 2)]
    dataset = Dataset(data)
    result = dataset.flat_map(TestFlatMapper()).execute()
    print(result)
    assert result == [("a", 1), ("a", 2), ("b", 2), ("b", 3)]


# Test Reducer
class TestReducer(Reducer):
    def reduce(self, key: str, values: List[int]) -> int:
        return sum(values)

    def validate(self, key: str, input_values: List[int], output_value: int) -> bool:
        return output_value == sum(input_values)


def test_reducer():
    data = [("a", 1), ("a", 2), ("b", 3), ("b", 4)]
    dataset = Dataset(data)
    result = dataset.reduce(TestReducer()).execute()
    assert result == [("a", 3), ("b", 7)]


# Test KeyResolver
class TestKeyResolver(KeyResolver):
    __test__ = False

    def are_equal(self, x: int, y: int) -> bool:
        return abs(x - y) <= 1

    def get_label_key(self, keys: Set[int]) -> int:
        return min(keys)


def test_key_resolver():
    data = [(1, "a"), (2, "b"), (3, "c"), (5, "d")]
    dataset = Dataset(data)
    result = dataset.resolve_keys(TestKeyResolver()).execute()
    assert set(result) == {(1, "a"), (1, "b"), (3, "c"), (5, "d")}


# Test Filterer
class TestFilterer(Filterer):
    def filter(self, key: str, value: int) -> bool:
        return value % 2 == 0


def test_filterer():
    data = [("a", 1), ("b", 2), ("c", 3), ("d", 4)]
    dataset = Dataset(data)
    result = dataset.filter(TestFilterer()).execute()
    assert result == [("b", 2), ("d", 4)]


class TestMapper2(Mapper):
    def map(self, key: str, value: int) -> Tuple[str, int]:
        if value % 2 == 0:
            return ("even", value)
        return ("odd", value)


class TestReducer2(Reducer):
    def reduce(self, key: str, values: List[int]) -> int:
        return sum(values)


# Test chaining operations
def test_chained_operations():
    data = [("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5)]
    dataset = Dataset(data)
    result = (
        dataset.map(TestMapper2())
        .reduce(TestReducer2())
        .filter(TestFilterer())
        .execute()
    )
    assert result == [("even", 6)]


# Test error handling
class ErrorMapper(Mapper):
    def map(self, key: str, value: int) -> Tuple[str, int]:
        if value == 2:
            raise ValueError("Error processing value 2")
        return (key, value * 2)

    def validate(
        self, input_key: str, input_value: int, output_key: str, output_value: int
    ) -> bool:
        return output_value != 4  # Fail validation for input value 2

    on_fail = ValidatorAction.FAIL


def test_error_handling():
    data = [("a", 1), ("b", 2), ("c", 3)]
    dataset = Dataset(data)
    with pytest.raises(ValueError):
        dataset.map(ErrorMapper()).execute()


# Test ParallelFlatMapper
class TestParallelFlatMapper(ParallelFlatMapper):
    def get_mappers(self) -> List[Callable[[str, int], Tuple[str, int]]]:
        return [
            lambda k, v: (k + "_1", v * 2),
            lambda k, v: (k + "_2", v * 3),
            lambda k, v: (k + "_3", v * 4),
        ]


def test_parallel_flat_mapper():
    data = [("a", 1), ("b", 2), ("c", 3)]
    dataset = Dataset(data)
    result = dataset.flat_map(TestParallelFlatMapper()).execute()

    expected = [
        ("a_1", 2),
        ("a_2", 3),
        ("a_3", 4),
        ("b_1", 4),
        ("b_2", 6),
        ("b_3", 8),
        ("c_1", 6),
        ("c_2", 9),
        ("c_3", 12),
    ]

    assert sorted(result) == sorted(expected)


if __name__ == "__main__":
    test_chained_operations()
