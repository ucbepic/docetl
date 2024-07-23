from typing import List, Tuple, Set
from motion.dataset import Dataset
from motion.operators import Mapper, Reducer, KeyResolver, Filterer
from motion.types import ValidatorAction


class SquareMapper(Mapper):
    def map(self, key: int, value: int) -> Tuple[int, int]:
        return (value**2, value**2)

    def validate(
        self, input_key: int, input_value: int, output_key: int, output_value: int
    ) -> bool:
        return output_value == input_value**2


class SumReducer(Reducer):
    def reduce(self, key: int, values: List[int]) -> int:
        return sum(values)

    def validate(self, key: int, input_values: List[int], output_value: int) -> bool:
        return output_value == sum(input_values)


class WithinFiveKeyResolver(KeyResolver):
    def precheck(self, x: int, y: int) -> bool:
        return abs(x - y) <= 5

    def are_equal(self, x: int, y: int) -> bool:
        return abs(x - y) <= 5

    def get_label(self, keys: Set[int]) -> int:
        return sum(keys) / len(keys)  # Return the average of the keys

    def validate(self, input_key: int, output_key: int) -> bool:
        return True


class PositiveFilter(Filterer):
    def filter(self, key: int, value: int) -> bool:
        return value > 0


if __name__ == "__main__":
    data = [("", i) for i in range(1, 11)]
    dataset = Dataset(data)

    result = (
        dataset.map(SquareMapper())
        .filter(PositiveFilter())
        .resolve_keys(WithinFiveKeyResolver())
        .reduce(SumReducer())
        .execute()
    )

    print(result)
