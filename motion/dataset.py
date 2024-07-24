import random
import multiprocessing
from typing import List, Any, Tuple, Iterable

from motion.types import K, V
from motion.operators import (
    Mapper,
    Reducer,
    KeyResolver,
    FlatMapper,
    Filterer,
)

from motion.optimizer import optimize
from motion.workers import apply_operation, Operation


class Dataset:
    def __init__(self, data: Iterable[Tuple[str, Any]], num_workers: int = None):
        for item in data[:10]:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("Each item must be a tuple of (K, V)")

        self.data = list(data)
        self.num_workers = num_workers or multiprocessing.cpu_count()
        self.operations: List[Operation] = []
        self.optimized_operations: List[Operation] = []

    def map(self, mapper: Mapper) -> "Dataset":
        self.operations.append(Operation(mapper))
        return self

    def flat_map(self, flatmapper: FlatMapper) -> "Dataset":
        self.operations.append(Operation(flatmapper))
        return self

    def resolve_keys(self, key_resolver: KeyResolver) -> "Dataset":
        self.operations.append(Operation(key_resolver))
        return self

    def reduce(self, reducer: Reducer) -> "Dataset":
        self.operations.append(Operation(reducer))
        return self

    def filter(self, filterer: Filterer) -> "Dataset":
        self.operations.append(Operation(filterer))
        return self

    def build(self, sample_size: int = 1000) -> "Dataset":
        # Sample the data
        sample_data = random.sample(self.data, min(sample_size, len(self.data)))

        optimized_operations = []

        for operation in self.operations:
            # Apply the operation to the sample data
            try:
                result, errors = apply_operation(
                    sample_data, operation, self.num_workers, building=True
                )

                if errors:
                    # If there are validation errors, attempt to optimize the operation
                    optimized_operation, sample_data = optimize(
                        operation,
                        sample_data,
                        errors,
                        self.num_workers,
                    )
                    optimized_operations.append(optimized_operation)
                else:
                    # If no errors, keep the original operation
                    optimized_operations.append(operation)
                    sample_data = result

            except Exception as e:
                print(
                    f"Error applying operation: {str(e)}. Keeping original operation."
                )
                optimized_operations.append(operation)

        # Update the operations with the optimized version
        self.optimized_operations = optimized_operations

        return self

    def execute(self) -> List[Tuple[str, Any, Any]]:
        ops = (
            self.optimized_operations if self.optimized_operations else self.operations
        )

        current_data = self.data
        for operation in ops:
            current_data = apply_operation(current_data, operation, self.num_workers)
        return current_data
