# Motion API

Motion API is a powerful and flexible data processing framework that allows you to create complex data processing pipelines with customizable operations.

## Table of Contents

1. [Overview](#overview)
2. [Key Components](#key-components)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Operator Types](#operator-types)
6. [Error Handling](#error-handling)
7. [Example](#example)

## Overview

Motion API provides a flexible framework for processing large datasets using a series of operations such as map, reduce, filter, and key resolution. It supports:

- Chaining multiple operations in any order
- Custom implementations of various operator types
- Customizable error handling and validation
- Easy-to-use API for defining and executing data processing pipelines

## Key Components

### Dataset

The main class that orchestrates the entire process. It allows you to define a series of operations and execute them on your data.

### Operator

The base class for all operation types. It includes common functionality such as error handling and description methods.

### Operator Types

- **Mapper**: Transforms each key-value pair into a new key-value pair.
- **FlatMapper**: Transforms each key-value pair into multiple key-value pairs.
- **Reducer**: Combines multiple values for each key into a single value.
- **KeyResolver**: Groups similar keys together.
- **Filterer**: Removes certain key-value pairs based on a condition.

## Usage

1. Import the necessary classes from the Motion API:

```python
from motion.dataset import Dataset
from motion.operators import Mapper, Reducer, KeyResolver, Filterer
```

2. Create custom operators by subclassing the appropriate operator type.

3. Initialize a Dataset with your input data.

4. Chain operations using the Dataset methods.

5. Execute the pipeline using the `execute()` method.

## Operator Types

### Mapper

```python
class CustomMapper(Mapper):
    def map(self, key: Any, value: Any) -> Tuple[K, V]:
        # Implementation here

    def validate(self, input_key: K, input_value: V, output_key: K, output_value: V) -> bool:
        # Validation logic here
```

### FlatMapper

```python
class CustomFlatMapper(FlatMapper):
    def map(self, key: Any, value: Any) -> List[Tuple[K, V]]:
        # Implementation here

    def validate(self, key: K, value: V, mapped_kv_pairs: List[Tuple[K, V]]) -> bool:
        # Validation logic here
```

### Reducer

```python
class CustomReducer(Reducer):
    def reduce(self, key: K, values: List[V]) -> V:
        # Implementation here

    def validate(self, key: K, input_values: List[V], output_value: V) -> bool:
        # Validation logic here
```

### KeyResolver

```python
class CustomKeyResolver(KeyResolver):
    def are_equal(self, x: K, y: K) -> bool:
        # Implementation here

    def get_label_key(self, keys: Set[K]) -> K:
        # Implementation here

    def validate(self, input_key: K, output_key: K) -> bool:
        # Validation logic here
```

### Filterer

```python
class CustomFilterer(Filterer):
    def filter(self, key: K, value: V) -> bool:
        # Implementation here

    def validate(self, key: K, value: V, output: bool) -> bool:
        # Validation logic here
```

## Error Handling

Motion API provides flexible error handling through the `ValidatorAction` enum and the `on_fail` attribute of Operators:

- `ValidatorAction.PROMPT`: Prompts the user for corrections when validation fails.
- `ValidatorAction.WARN`: Prints a warning message but continues execution.
- `ValidatorAction.FAIL`: Raises an exception, halting execution.

To set the error handling behavior for a custom operator:

```python
class CustomMapper(Mapper):
    on_fail = ValidatorAction.PROMPT
    # ... rest of the implementation
```

## Example

Here's a simple example demonstrating the usage of Motion API:

```python
from typing import List, Tuple, Set
from motion.dataset import Dataset
from motion.operators import Mapper, Reducer, KeyResolver, Filterer
from motion.types import ValidatorAction

class SquareMapper(Mapper):
    def map(self, key: int, value: int) -> Tuple[int, int]:
        return (value**2, value**2)

    def validate(self, input_key: int, input_value: int, output_key: int, output_value: int) -> bool:
        return output_value == input_value**2

class SumReducer(Reducer):
    def reduce(self, key: int, values: List[int]) -> int:
        return sum(values)

    def validate(self, key: int, input_values: List[int], output_value: int) -> bool:
        return output_value == sum(input_values)

class WithinFiveKeyResolver(KeyResolver):
    def are_equal(self, x: int, y: int) -> bool:
        return abs(x - y) <= 5

    def get_label_key(self, keys: Set[int]) -> int:
        return sum(keys) // len(keys)  # Return the average of the keys

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
```

This example demonstrates a pipeline that squares numbers, filters out non-positive values, resolves keys that are within 5 of each other, and then sums the values for each resolved key.
