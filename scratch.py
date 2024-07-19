import sys
import json
from abc import ABC, abstractmethod
import multiprocessing
from typing import (
    List,
    Any,
    Dict,
    Tuple,
    Optional,
    Set,
    Callable,
    TypeVar,
    Generic,
    Union,
)
from collections import defaultdict
from enum import Enum
import random
import uuid

K = TypeVar("K")
V = TypeVar("V")
R = TypeVar("R")


class ValidatorAction(Enum):
    PROMPT = "prompt"
    WARN = "warn"
    FAIL = "fail"


class Operator(ABC, Generic[K, V]):
    on_fail: ValidatorAction = ValidatorAction.WARN

    @abstractmethod
    def validate(self, key: K, value: V) -> bool:
        pass

    @abstractmethod
    def correct(self, key: K, value: V, new_value: Any) -> Tuple[K, V]:
        pass


class Mapper(Operator[K, V]):
    @abstractmethod
    def map(self, key: Any, value: Any) -> List[Tuple[K, V]]:
        pass

    def correct(self, key: K, value: V, new_value: Any) -> Tuple[K, V]:
        try:
            new_key = type(key)(new_value)
            new_value = type(value)(new_value)
            return new_key, new_value
        except (ValueError, TypeError):
            return key, value


class Reducer(Operator[K, V]):
    @abstractmethod
    def reduce(self, key: K, values: List[V]) -> V:
        pass

    def correct(self, key: K, value: V, new_value: Any) -> Tuple[K, V]:
        try:
            new_key = type(key)(new_value)
            new_value = type(value)(new_value)
            return new_key, new_value
        except (ValueError, TypeError):
            return key, value


class Equality(ABC, Generic[K]):
    @abstractmethod
    def precheck(self, x: K, y: K) -> bool:
        pass

    @abstractmethod
    def are_equal(self, x: K, y: K) -> bool:
        pass

    @abstractmethod
    def get_label(self, keys: Set[K]) -> Any:
        pass

    def correct(self, key: K, new_value: Any) -> K:
        try:
            return type(key)(new_value)
        except (ValueError, TypeError):
            return key


class Operation:
    def __init__(
        self, operator: Union[Mapper, Reducer], equality: Optional[Equality] = None
    ):
        self.operator = operator
        self.equality = equality


class MapReduce:
    def __init__(self, data: List[Any], num_workers: int = None):
        self.data = [("", item) for item in data]  # Add a dummy key
        self.num_workers = num_workers or multiprocessing.cpu_count()
        self.operations: List[Operation] = []

    def map(self, mapper: Mapper) -> "MapReduce":
        self.operations.append(Operation(mapper))
        return self

    def reduce(self, reducer: Reducer, equality: Equality) -> "MapReduce":
        self.operations.append(Operation(reducer, equality))
        return self

    def execute(self) -> List[Tuple[Any, Any]]:
        current_data = self.data
        for operation in self.operations:
            current_data = self._apply_operation(current_data, operation)
        return current_data

    def _apply_operation(
        self, data: List[Tuple[Any, Any]], operation: Operation
    ) -> List[Tuple[Any, Any]]:
        chunk_size = max(len(data) // self.num_workers, 1)
        chunks = self._chunk_data(data, chunk_size)

        with multiprocessing.Pool(self.num_workers) as pool:
            if isinstance(operation.operator, Mapper):
                results = pool.starmap(
                    self._map_worker, [(chunk, operation) for chunk in chunks]
                )
            else:  # Reducer
                grouped_data = self._group_by_key(data, operation.equality)
                results = pool.starmap(
                    self._reduce_worker,
                    [(key, values, operation) for key, values in grouped_data.items()],
                )

        processed_data = []
        errors = []
        for result, error in results:
            processed_data.extend(result)
            errors.extend(error)

        corrected_data = self._handle_validation_errors(
            errors, operation.operator.on_fail, operation.operator
        )
        processed_data.extend(corrected_data)

        return processed_data

    def _chunk_data(
        self, data: List[Tuple[Any, Any]], chunk_size: int
    ) -> List[List[Tuple[Any, Any]]]:
        return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

    def _map_worker(
        self, chunk: List[Tuple[Any, Any]], operation: Operation
    ) -> Tuple[List[Tuple[Any, Any]], List[Tuple[str, Any, Any]]]:
        result = []
        errors = []
        for key, value in chunk:
            mapped_items = operation.operator.map(key, value)
            for mapped_key, mapped_value in mapped_items:
                operation_id = str(uuid.uuid4())
                if operation.operator.validate(mapped_key, mapped_value):
                    result.append((mapped_key, mapped_value))
                else:
                    errors.append((operation_id, mapped_key, mapped_value))
        return result, errors

    def _reduce_worker(
        self, key: Any, values: List[Any], operation: Operation
    ) -> Tuple[List[Tuple[Any, Any]], List[Tuple[str, Any, Any]]]:
        operation_id = str(uuid.uuid4())
        reduced_value = operation.operator.reduce(key, values)
        if operation.operator.validate(key, reduced_value):
            return [(key, reduced_value)], []
        else:
            return [], [(operation_id, key, reduced_value)]

    def _group_by_key(
        self, data: List[Tuple[Any, Any]], equality: Equality
    ) -> Dict[Any, List[Any]]:
        grouped_data = defaultdict(list)  # TODO should we change the key names?
        unique_keys = []

        for key, value in data:
            for unique_key in unique_keys:
                if equality.precheck(key, unique_key) and equality.are_equal(
                    key, unique_key
                ):
                    grouped_data[unique_key].append(value)
                    break
            else:
                unique_keys.append(key)
                grouped_data[key].append(value)

        return dict(grouped_data)

    def _handle_validation_errors(
        self,
        errors: List[Tuple[str, Any, Any]],
        action: ValidatorAction,
        operator: Operator,
    ) -> List[Tuple[Any, Any]]:
        if not errors:
            return []

        if action == ValidatorAction.PROMPT:
            print("Validation Errors:", file=sys.stderr)
            for operation_id, error_key, error_value in errors:
                print(
                    f"  - ID: {operation_id}, Key: {error_key}, Value: {error_value}",
                    file=sys.stderr,
                )
            print(
                "\nEnter corrections as a JSON dictionary mapping ID to new value, or press Enter to skip:",
                file=sys.stderr,
            )
            try:
                user_input = sys.stdin.readline().strip()
                if user_input:
                    corrections = json.loads(user_input)
                    corrected_errors = []
                    for operation_id, error_key, error_value in errors:
                        if operation_id in corrections:
                            new_value = corrections[operation_id]
                            corrected_key, corrected_value = operator.correct(
                                error_key, error_value, new_value
                            )
                            corrected_errors.append((corrected_key, corrected_value))
                    return corrected_errors
                return []
            except json.JSONDecodeError:
                print("Invalid JSON input. Skipping corrections.", file=sys.stderr)
            except KeyboardInterrupt:
                print("\nOperation aborted by user.", file=sys.stderr)
                sys.exit(1)
        elif action == ValidatorAction.WARN:
            for operation_id, error_key, error_value in errors:
                print(
                    f"Warning: Validation failed for ID: {operation_id}, Key: {error_key}, Value: {error_value}",
                    file=sys.stderr,
                )
        elif action == ValidatorAction.FAIL:
            error_message = "\n".join(
                f"ID: {operation_id}, Key: {error_key}, Value: {error_value}"
                for operation_id, error_key, error_value in errors
            )
            raise ValueError(f"Validation Errors:\n{error_message}")

        return []


# Example implementations
class NumberMapper(Mapper[float, int]):
    on_fail = ValidatorAction.PROMPT

    def map(self, key: Any, value: int) -> List[Tuple[float, int]]:
        return [(value / 10, value)]

    def validate(self, key: float, value: int) -> bool:
        if random.random() < 0.2:
            return False
        return 0 <= key <= 10 and 1 <= value <= 100


class SquareMapper(Mapper[float, int]):
    on_fail = ValidatorAction.WARN

    def map(self, key: float, value: int) -> List[Tuple[float, int]]:
        return [(key, value**2)]

    def validate(self, key: float, value: int) -> bool:
        return value >= 0


class SumReducer(Reducer[float, int]):
    on_fail = ValidatorAction.WARN

    def reduce(self, key: float, values: List[int]) -> int:
        return sum(values)

    def validate(self, key: float, reduced_value: int) -> bool:
        if random.random() < 0.2:
            return False

        return 1 <= reduced_value <= 1000000  # Increased upper bound due to squaring


class FuzzyEquality(Equality[float]):
    def __init__(self, tolerance: float):
        self.tolerance = tolerance

    def precheck(self, x: float, y: float) -> bool:
        return abs(x - y) <= 2 * self.tolerance

    def are_equal(self, x: float, y: float) -> bool:
        return abs(x - y) <= self.tolerance

    def get_label(self, keys: Set[float]) -> str:
        min_key = min(keys)
        max_key = max(keys)
        return f"[{min_key:.1f}, {max_key:.1f}]"


if __name__ == "__main__":
    input_data = list(range(1, 10))  # Numbers from 1 to 10

    mr = MapReduce(input_data)
    result = (
        mr.map(NumberMapper())
        .reduce(SumReducer(), FuzzyEquality(tolerance=0.1))
        .map(SquareMapper())
        .reduce(SumReducer(), FuzzyEquality(tolerance=0.01))
        .execute()
    )

    # Print results in a more readable format
    for key, value in sorted(result):
        print(f"Group {key}: Final Result = {value}")
