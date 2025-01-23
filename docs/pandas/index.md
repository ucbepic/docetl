# Pandas Integration

DocETL provides seamless integration for a few operators (map, filter, merge, agg) with pandas through a dataframe accessor. This idea was proposed by LOTUS[^1]. 

## Installation

The pandas integration is included in the main DocETL package:

```bash
pip install docetl
```

## Overview

The pandas integration provides a `.semantic` accessor that enables:

- Semantic mapping with LLMs (`df.semantic.map()`)
- Intelligent filtering (`df.semantic.filter()`)
- Fuzzy merging of DataFrames (`df.semantic.merge()`)
- Semantic aggregation (`df.semantic.agg()`)
- Cost tracking and operation history

## Quick Example

```python
import pandas as pd
from docetl import SemanticAccessor

# Create a DataFrame
df = pd.DataFrame({
    "text": [
        "Apple released the iPhone 15 with USB-C port",
        "Microsoft's new Surface laptops feature AI capabilities",
        "Google announces Pixel 8 with enhanced camera features"
    ]
})

# Configure the semantic accessor
df.semantic.set_config(default_model="gpt-4o-mini")

# Extract structured information
result = df.semantic.map(
    prompt="Extract company and product from: {{input.text}}",
    output_schema={
        "company": "str",
        "product": "str",
        "features": "list[str]"
    }
)

# Track costs
print(f"Operation cost: ${result.semantic.total_cost}")
```

## Configuration

Configure the semantic accessor with your preferred settings:

```python
df.semantic.set_config(
    default_model="gpt-4o-mini",  # Default LLM to use
    max_threads=64,              # Maximum concurrent threads,
    rate_limits={
        "embedding_call": [
            {"count": 1000, "per": 1, "unit": "second"}
        ],
        "llm_call": [
            {"count": 1, "per": 1, "unit": "second"},
            {"count": 10, "per": 5, "unit": "hour"}
        ]
    } 
)
```

!!! note "Pipeline Optimization"

    While individual semantic operations are optimized internally, pipelines created through the pandas `.semantic` accessor (sequences of operations like `map` → `filter` → `merge`) cannot be optimized as a whole. For pipeline-level optimizations like operation rewriting and automatic resolve operation insertion, you must use either:
    
    - The YAML configuration interface
    - The Python API

For detailed configuration options and best practices, refer to:

- [DocETL Best Practices](../best-practices.md)
- [Pipeline Configuration](../concepts/pipelines.md)
- [Output Schemas](../concepts/schemas.md)
- [Rate Limiting](../examples/rate-limiting.md) 

## Cost Tracking

All semantic operations track their LLM usage costs:

```python
# Get total cost of operations
total_cost = df.semantic.total_cost

# Get operation history
history = df.semantic.history
for op in history:
    print(f"Operation: {op.op_type}")
    print(f"Modified columns: {op.output_columns}")
```

## Implementation

This implementation is inspired by [LOTUS](https://github.com/guestrin-lab/lotus), a system introduced by Patel et al. [^1]. Our implementation has a few differences:

- We use DocETL's query engine to run the LLM operations. This allows us to use retries, validation, well-defined output schemas, and other features described in our documentation.
- Our aggregation operator combines the `resolve` and `reduce` operators, so you can get a fuzzy groupby.
- Our merge operator is based on our equijoin operator implementation, which optimizes LLM call usage by generating blocking rules before running the LLM. See the [Equijoin Operator](../operators/equijoin.md) for more details.
- We do not implement LOTUS's `sem_extract`, `sem_topk`, `sem_sim_join`, and `sem_search` operators. However, `sem_extract` can effectively be implemented by running the `map` operator with a prompt that describes the extraction.

[^1]: Patel, L., Jha, S., Asawa, P., Pan, M., Guestrin, C., & Zaharia, M. (2024). Semantic Operators: A Declarative Model for Rich, AI-based Analytics Over Text Data. arXiv preprint arXiv:2407.11418. [https://arxiv.org/abs/2407.11418](https://arxiv.org/abs/2407.11418) 