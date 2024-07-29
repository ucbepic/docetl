import os
import random
from typing import List, Any, Tuple, Iterable, Union

from motion.types import K, V
from motion.operators import (
    LLMMapper,
    LLMReducer,
    KeyResolver,
    LLMFlatMapper,
    LLMFilterer,
    Splitter,
    Mapper,
)

from motion.optimizer import optimize
from motion.executor import apply_operation, Operation
from rich.console import Console
from rich import print as rprint


class Dataset:
    def __init__(self, data: Iterable[Tuple[str, Any]], num_workers: int = None):
        for item in data[:10]:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("Each item must be a tuple of (K, V)")

        self.data = list(data)
        self.num_workers = num_workers or (os.cpu_count() or 1) * 4
        self.operations: List[Operation] = []
        self.optimized_operations: List[Operation] = []
        self.console = Console()

    def map(self, mapper: Union[LLMMapper, Mapper]) -> "Dataset":
        self.operations.append(Operation(mapper))
        return self

    def flat_map(self, flatmapper: LLMFlatMapper) -> "Dataset":
        self.operations.append(Operation(flatmapper))
        return self

    def split(self, splitter: Splitter) -> "Dataset":
        self.operations.append(Operation(splitter))
        return self

    def resolve_keys(self, key_resolver: KeyResolver) -> "Dataset":
        self.operations.append(Operation(key_resolver))
        return self

    def reduce(self, reducer: LLMReducer) -> "Dataset":
        self.operations.append(Operation(reducer))
        return self

    def filter(self, filterer: LLMFilterer) -> "Dataset":
        self.operations.append(Operation(filterer))
        return self

    def build(self, sample_size: int = 1000) -> "Dataset":
        # Create a new dataset
        ds = Dataset(self.data, self.num_workers)

        # Sample the data
        sample_data = random.sample(self.data, min(sample_size, len(self.data)))

        for operation in self.operations:
            # Apply the operation to the sample data
            # try:
            with self.console.status(
                f"Building [cyan]{operation.operator.__class__.__name__}[/cyan]..."
            ) as status:
                operation.operator.set_build_phase(True)
                result, errors, base_cost = apply_operation(
                    sample_data, operation, self.num_workers, building=True
                )

                if errors:
                    # If there are validation errors, attempt to optimize the operation
                    status.stop()
                    status = self.console.status(
                        f"Optimizing [cyan]{operation.operator.__class__.__name__}[/cyan]..."
                    )
                    status.start()
                    optimized_operations, sample_data = optimize(
                        operation,
                        len(self.data),
                        sample_data,
                        result,
                        errors,
                        self.num_workers,
                        base_cost,
                    )
                    ds.operations.extend(optimized_operations)
                else:
                    # If no errors, keep the original operation
                    ds.operations.append(operation)
                    sample_data = result

                status.stop()
                status = self.console.status(
                    f"[cyan]{operation.operator.__class__.__name__}[/cyan] built successfully."
                )
                status.start()

        # except Exception as e:
        #     self.console.print(f"[bold red]Error applying operation:[/bold red] {str(e)}. Keeping original operation.")
        #     optimized_operations.append(operation)
        # finally:
        #     operation.operator.set_build_phase(False)

        return ds

    def execute(self) -> List[Tuple[str, Any, Any]]:
        total_cost = 0

        current_data = self.data
        for operation in self.operations:
            with self.console.status(
                f"Running [cyan]{operation.operator.__class__.__name__}[/cyan]...",
            ) as status:
                current_data, cost = apply_operation(
                    current_data, operation, self.num_workers
                )
                status.update(
                    f"[cyan]{operation.operator.__class__.__name__}[/cyan] completed. Cost: [green]${cost:.2f}[/green]"
                )
            total_cost += cost

        rprint(f"[bold]Total cost:[/bold] [green]${total_cost:.2f}[/green]")

        return current_data
