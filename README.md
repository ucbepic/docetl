# Motion API

Motion API is a powerful and flexible data processing framework that allows you to create complex data processing pipelines with customizable operations, leveraging Large Language Models (LLMs) for advanced data manipulation.

## Table of Contents

1. [Overview](#overview)
2. [Key Components](#key-components)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Operator Types](#operator-types)
6. [Complete Example](#complete-example)
7. [Testing](#testing)

## Overview

Motion API provides a flexible framework for processing datasets using a series of LLM-powered operations such as map, reduce, filter, and key resolution. It supports:

- Chaining multiple operations in any order
- Custom implementations of various LLM-based operator types
- Easy-to-use API for defining and executing data processing pipelines
- Integration with OpenAI's GPT models

## Key Components

### Dataset

The main class that orchestrates the entire process. It allows you to define a series of operations and execute them on your data.

### Operators

Various operator types that can be applied to the dataset:

- LLMMapper
- LLMReducer
- LLMPairwiseKeyResolver
- LLMListKeyResolver
- LLMFilterer
- LLMFlatMapper
- LLMParallelFlatMapper
- Splitter

## Installation

To install Motion API, use pip:

```
pip install .
```

## Usage

1. Import the necessary classes from the Motion API:

```python
from motion.dataset import Dataset
from motion.operators import LLMMapper, LLMReducer, LLMListKeyResolver  # and other operators as needed
```

2. Create a Dataset object and define your operations:

```python
dataset = Dataset(your_data)
result = (
    dataset.map(YourCustomMapper())
           .resolve_keys(YourCustomKeyResolver())
           .reduce(YourCustomReducer())
           .execute()
)
```

## Operator Types

### LLMMapper

Transforms each key-value pair using an LLM. Implement the `generate_prompt` and `process_response` methods:

```python
class CustomMapper(LLMMapper):
    def generate_prompt(self, key: str, value: Any) -> list:
        # Generate the prompt for the LLM
        pass

    def process_response(self, response: Any, **prompt_kwargs) -> Tuple[str, Any]:
        # Process the LLM's response
        pass
```

### LLMReducer

Combines multiple values for each key using an LLM. Implement the `generate_prompt` and `process_response` methods:

```python
class CustomReducer(LLMReducer):
    def generate_prompt(self, key: str, values: List[Any]) -> list:
        # Generate the prompt for the LLM
        pass

    def process_response(self, response: Any, **prompt_kwargs) -> Any:
        # Process the LLM's response
        pass
```

### LLMListKeyResolver

Resolves and consolidates keys using an LLM. Implement the `generate_prompt`, `process_response`, and `get_label_key` methods:

```python
class CustomListKeyResolver(LLMListKeyResolver):
    def generate_prompt(self, key: str, label_keys: List[str]) -> list:
        # Generate the prompt for the LLM
        pass

    def process_response(self, response: Any, **prompt_kwargs) -> str:
        # Process the LLM's response
        pass

    def get_label_key(self, keys: set) -> str:
        # Define how to select a label key
        pass
```

## Complete Example

Here's a complete example that demonstrates how to use the Motion API to process a dataset of items, categorize them, resolve categories, and generate stories for each category:

```python
import random
from typing import List, Tuple, Any
from motion.dataset import Dataset
from motion.operators import LLMMapper, LLMReducer, LLMListKeyResolver

MODEL = "gpt-4-mini"

# Synthetic data
items = ["apple", "banana", "car", "dog", "elephant", "fish", "guitar", "house", "igloo", "jacket"]
data = [(item, random.randint(1, 5)) for item in items]
dataset = Dataset(data)

class CategoryMapper(LLMMapper):
    def generate_prompt(self, key: str, value: int) -> list:
        return [
            {"role": "system", "content": "You are a helpful assistant that categorizes items."},
            {"role": "user", "content": f"Please categorize this item into a higher-level category. Item: {key}\nYour answer should be a single word."},
        ]

    def process_response(self, response: Any, **prompt_kwargs) -> Tuple[str, Tuple[str, int]]:
        category = response.choices[0].message.content.strip()
        return (category, (prompt_kwargs["key"], prompt_kwargs["value"]))

class CategoryResolver(LLMListKeyResolver):
    def generate_prompt(self, key: str, label_keys: List[str]) -> list:
        return [
            {"role": "system", "content": "You are a helpful assistant that determines the most appropriate category for an item."},
            {"role": "user", "content": f"Given the category '{key}' and the existing categories {label_keys}, which category should it be assigned to? If it doesn't match any existing categories, respond with 'NEW'. Provide your answer as a single word or 'NEW'."},
        ]

    def process_response(self, response: Any, **prompt_kwargs) -> str:
        content = response.choices[0].message.content.strip()
        return content if content != "NEW" else prompt_kwargs["key"]

    def get_label_key(self, keys: set) -> str:
        return random.choice(list(keys))

class StoryGenerator(LLMReducer):
    def generate_prompt(self, key: str, values: List[Tuple[str, int]]) -> list:
        items = ", ".join([f"{item} (importance: {importance}/5)" for item, importance in values])
        return [
            {"role": "system", "content": "You are a creative storyteller that creates cute, short stories."},
            {"role": "user", "content": f"Create a cute, short story about these items in the category '{key}'. Focus more on items with higher importance: {items}"},
        ]

    def process_response(self, response: Any, **prompt_kwargs) -> str:
        return response.choices[0].message.content.strip()

# Apply operations
result = (
    dataset.map(CategoryMapper(model=MODEL))
    .resolve_keys(CategoryResolver(model=MODEL))
    .reduce(StoryGenerator(model=MODEL))
    .execute()
)

# Print results
for category, story in result:
    print(f"\nCategory: {category}")
    print(f"Story: {story}")
    print("-" * 50)

print(f"Number of categories: {len(result)}")
```

This example demonstrates the following steps:

1. Creates a synthetic dataset of items with random importance scores.
2. Uses an LLMMapper to categorize each item.
3. Applies an LLMListKeyResolver to resolve and consolidate categories.
4. Generates a story for each category using an LLMReducer.
5. Prints the resulting stories and the number of categories.

## Testing

The Motion API includes a comprehensive test suite. To run the tests, use pytest:

```
pytest test_motion_api.py
```

For more detailed information and advanced usage, please refer to the source code and tests.
