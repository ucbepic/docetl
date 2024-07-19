# Motion API

This README provides an overview of Motion's API implementation. This system allows for the creation of complex data processing pipelines with customizable map and reduce operations.

## Table of Contents

1. [Overview](#overview)
2. [Key Components](#key-components)
3. [Component Classes](#component-classes)
4. [API Usage](#api-usage)
5. [Example Implementation](#example-implementation)
6. [Error Handling](#error-handling)

## Overview

This implementation provides a flexible framework for processing large datasets using a series of map and reduce operations. It supports:

- Chaining multiple map and reduce operations in any order
- Custom mapper and reducer implementations
- Customizable error handling and validation

## Key Components

### MapReduce

The main class that orchestrates the entire process. It allows you to define a series of map and reduce operations and execute them on your data.

Methods (the user does not write these):
- `__init__(data: List[Any], num_workers: int = None)`: Initialize with input data and optional number of worker processes.
- `map(mapper: Mapper) -> MapReduce`: Add a map operation to the pipeline.
- `reduce(reducer: Reducer, equality: Equality) -> MapReduce`: Add a reduce operation to the pipeline.
- `execute() -> List[Tuple[Any, Any]]`: Execute the defined pipeline and return the results.

### ValidatorAction

An enumeration defining possible actions to take when validation fails:
- `PROMPT`: Prompts the user for corrections when validation fails. The user will provide corrections for all failed records before the next operation executes.
- `WARN`: Prints a warning message but continues execution.
- `FAIL`: Raises an exception, halting execution.

### Operator[K, V]

An abstract base class for both Mapper and Reducer.

Attributes:
- `on_fail`: ValidatorAction - Defines the action to take when validation fails. Defaults to `WARN`.

Methods:
- `validate(key: K, value: V) -> bool`: Abstract method to validate the key-value pair.
- `correct(key: K, value: V, new_value: Any) -> Tuple[K, V]`: Method to correct invalid values given user-provided new values.

### Mapper[K, V] (subclass of Operator)

Abstract base class for defining map operations. The user will subclass this to create custom mappers.

Methods:
- `map(key: Any, value: Any) -> List[Tuple[K, V]]`: Abstract method to define the mapping operation. Key can be empty.

### Reducer[K, V] (subclass of Operator)

Abstract base class for defining reduce operations. Subclass this to create custom reducers.

Methods:
- `reduce(key: K, values: List[V]) -> V`: Abstract method to define the reduce operation. Key can be empty.

### Equality[K]

Abstract base class for defining equality operations used in the reduce step for a fuzzy grouping of keys.

Methods:
- `precheck(x: K, y: K) -> bool`: Abstract method for a quick check of potential equality. If this returns false, `are_equal` is never run.
- `are_equal(x: K, y: K) -> bool`: Abstract method to determine if two keys are equal. This can be expensive.
- `get_label(keys: Set[K]) -> Any`: Abstract method to generate a label for a group of keys.

The `precheck` method is an optimization step. It should quickly determine if two keys might be equal, avoiding the potentially more expensive `are_equal` check when keys are clearly not equal. For example, in a fuzzy floating-point comparison, `precheck` might check if the absolute difference is within a larger tolerance, while `are_equal` uses a stricter tolerance.

## API Usage

### Initializing MapReduce

```python
mr = MapReduce(input_data, num_workers=None)
```

- `input_data`: A list of input items to process
- `num_workers`: Number of worker processes (default: number of CPU cores)

### Adding Operations

```python
mr.map(mapper: Mapper) -> MapReduce
mr.reduce(reducer: Reducer, equality: Equality) -> MapReduce
```

Both methods return the MapReduce instance, allowing for method chaining.

### Executing the Pipeline

```python
result = mr.execute()
```

Returns a list of key-value pairs representing the final output.

## Example Implementation

Here's a simple example of how to use the API:

```python
class NumberMapper(Mapper[float, int]):
    def map(self, key: Any, value: int) -> List[Tuple[float, int]]:
        return [(value / 10, value)]

class SquareMapper(Mapper[float, int]):
    def map(self, key: float, value: int) -> List[Tuple[float, int]]:
        return [(key, value**2)]

class SumReducer(Reducer[float, int]):
    def reduce(self, key: float, values: List[int]) -> int:
        return sum(values)

class FuzzyEquality(Equality[float]):
    def __init__(self, tolerance: float):
        self.tolerance = tolerance

    def precheck(self, x: float, y: float) -> bool:
        return abs(x - y) <= 2 * self.tolerance

    def are_equal(self, x: float, y: float) -> bool:
        return abs(x - y) <= self.tolerance

    def get_label(self, keys: Set[float]) -> str:
        return f"[{min(keys):.1f}, {max(keys):.1f}]"

# Usage
input_data = list(range(1, 101))
mr = MapReduce(input_data)
result = (mr
    .map(NumberMapper())
    .reduce(SumReducer(), FuzzyEquality(tolerance=0.5))
    .map(SquareMapper())
    .reduce(SumReducer(), FuzzyEquality(tolerance=0.5))
    .execute()
)
```

## Error Handling

The system provides flexible error handling through the `ValidatorAction` enum and the `on_fail` attribute of Operators:

- `ValidatorAction.PROMPT`: Prompts the user for corrections when validation fails
- `ValidatorAction.WARN`: Prints a warning message but continues execution
- `ValidatorAction.FAIL`: Raises an exception, halting execution

To set the error handling behavior for a custom Mapper or Reducer:

```python
class CustomMapper(Mapper[K, V]):
    on_fail = ValidatorAction.PROMPT
    # ... rest of the implementation
```

This allows for fine-grained control over how errors are handled at each stage of the MapReduce pipeline.