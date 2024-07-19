# Motion API

This README provides an overview of Motion's API implementation. This system allows for the creation of complex data processing pipelines with customizable map and reduce operations.

## TODOs
- [ ] Nit: flexible key handling on dataset creation; users should be able to pass in their own key
- [ ] Make reducer stateful: decompose Reducer's reduce function into aggregate and merge
- [ ] Embedding-based fuzzy equality/grouping
- [ ] Flat map operator
- [ ] Implement optimizations
- [ ] Implement accuracy boosters
    - [ ] "Agentic" decisions, e.g., employing an LLM to write the equality precheck & equality prompts
    - [ ] "Adaptive" decisions, e.g., using previous results that pass validators as few shot examples in prompts
    - [ ] Allow traces to be accessed in operators (so one can use raw sources if they want)
- [ ] Implement streaming API

## Table of Contents

1. [Overview](#overview)
2. [Key Components](#key-components)
3. [Component Classes](#component-classes)
4. [API Usage](#api-usage)
5. [Example Implementation](#example-implementation)
6. [Error Handling](#error-handling)
7. [Tracing](#tracing)

## Overview

This implementation provides a flexible framework for processing large datasets using a series of map and reduce operations. It supports:

- Chaining multiple map and reduce operations in any order
- Custom mapper and reducer implementations
- Customizable error handling and validation
- Tracing of operations for debugging and analysis

## Key Components

### Dataset

The main class that orchestrates the entire process. It allows you to define a series of map and reduce operations and execute them on your data.

Methods:
- `__init__(data: Iterable[Tuple[Any, Any]], num_workers: int = None, enable_tracing: bool = True)`: Initialize with input data, optional number of worker processes, and tracing option.
- `map(mapper: Mapper) -> Dataset`: Add a map operation to the pipeline.
- `reduce(reducer: Reducer, equality: Equality) -> Dataset`: Add a reduce operation to the pipeline.
- `execute() -> List[Tuple[str, Any, Any, Optional[Dict[str, str]]]]`: Execute the defined pipeline and return the results.

### ValidatorAction

An enumeration defining possible actions to take when validation fails:
- `PROMPT`: Prompts the user for corrections when validation fails.
- `WARN`: Prints a warning message but continues execution.
- `FAIL`: Raises an exception, halting execution.

### Operator[K, V]

An abstract base class for both Mapper and Reducer.

Attributes:
- `on_fail`: ValidatorAction - Defines the action to take when validation fails. Defaults to `WARN`.

Methods:
- `validate(key: K, value: V) -> bool`: Abstract method to validate the key-value pair.
- `correct(key: K, value: V, new_value: Any) -> Tuple[K, V]`: Method to correct invalid values given user-provided new values.
- `get_description() -> Optional[str]`: Method to provide a description of the operation for tracing.

### Mapper[K, V] (subclass of Operator)

Abstract base class for defining map operations.

Methods:
- `map(key: Any, value: Any) -> List[Tuple[K, V]]`: Abstract method to define the mapping operation.

### Reducer[K, V] (subclass of Operator)

Abstract base class for defining reduce operations.

Methods:
- `reduce(key: K, values: List[V]) -> V`: Abstract method to define the reduce operation.

### Equality[K]

Abstract base class for defining equality operations used in the reduce step for a fuzzy grouping of keys.

Methods:
- `precheck(x: K, y: K) -> bool`: Abstract method for a quick check of potential equality.
- `are_equal(x: K, y: K) -> bool`: Abstract method to determine if two keys are equal.
- `get_label(keys: Set[K]) -> Any`: Abstract method to generate a label for a group of keys.
- `get_description() -> Optional[str]`: Method to provide a description of the equality operation for tracing.

## API Usage

### Initializing Dataset

```python
dataset = Dataset(input_data, num_workers=None, enable_tracing=True)
```

- `input_data`: An iterable of key-value pairs to process
- `num_workers`: Number of worker processes (default: number of CPU cores)
- `enable_tracing`: Whether to enable operation tracing (default: True)

### Adding Operations

```python
dataset.map(mapper: Mapper) -> Dataset
dataset.reduce(reducer: Reducer, equality: Equality) -> Dataset
```

Both methods return the Dataset instance, allowing for method chaining.

### Executing the Pipeline

```python
result = dataset.execute()
```

Returns a list of tuples (id, key, value, trace), where trace is an optional dictionary of operation descriptions.

## Example Implementation

Here's a simple example of how to use the API:

```python
class NumberMapper(Mapper[float, int]):
    on_fail = ValidatorAction.WARN

    def map(self, key: Any, value: int) -> List[Tuple[float, int]]:
        return [(float(value) / 10, value)]

    def validate(self, key: float, value: int) -> bool:
        return 0 <= key <= 10 and 1 <= value <= 100

    def get_description(self) -> str:
        return "Convert value to float key"

class SquareMapper(Mapper[float, int]):
    on_fail = ValidatorAction.WARN

    def map(self, key: float, value: int) -> List[Tuple[float, int]]:
        return [(key, value**2)]

    def validate(self, key: float, value: int) -> bool:
        return value >= 0

    def get_description(self) -> str:
        return "Square the value"

class SumReducer(Reducer[float, int]):
    on_fail = ValidatorAction.WARN

    def reduce(self, key: float, values: List[int]) -> int:
        return sum(values)

    def validate(self, key: float, reduced_value: int) -> bool:
        return 1 <= reduced_value <= 1000000

    def get_description(self) -> str:
        return "Sum all values"

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

# Usage
input_data = range(1, 10)  # Numbers from 1 to 9

dataset = Dataset(input_data, enable_tracing=True)
result = (dataset
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

This allows for fine-grained control over how errors are handled at each stage of the data processing pipeline.

## Tracing

The Dataset class supports operation tracing, which can be useful for debugging and understanding the flow of data through your pipeline.

### Enabling/Disabling Tracing

Tracing is enabled by default when creating a Dataset. You can disable it by setting `enable_tracing=False`:

```python
dataset = Dataset(input_data, enable_tracing=False)
```

### Trace Information

When tracing is enabled, each item in the result will include a trace dictionary. This dictionary contains descriptions of the operations applied to the data, keyed by operation index.

For example, a trace might look like this:

```python
{
    'op_0': 'Convert value to float key',
    'op_1': 'Sum all values',
    'op_1_equality': 'Group keys within 0.5 tolerance',
    'op_2': 'Square the value',
    'op_3': 'Sum all values',
    'op_3_equality': 'Group keys within 0.5 tolerance'
}
```

This trace shows the sequence of operations applied to the data, including both map and reduce operations, as well as the equality operations used in reductions.

### Implementing Tracing in Custom Operators

To support tracing in your custom Mapper, Reducer, or Equality classes, implement the `get_description()` method:

```python
class CustomMapper(Mapper[K, V]):
    def get_description(self) -> str:
        return "Description of the mapping operation"

class CustomReducer(Reducer[K, V]):
    def get_description(self) -> str:
        return "Description of the reducing operation"

class CustomEquality(Equality[K]):
    def get_description(self) -> str:
        return "Description of the equality operation"
```

These descriptions will be included in the trace information when the operations are executed.