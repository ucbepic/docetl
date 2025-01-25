# Semantic Operations

The pandas integration provides several semantic operations through the `.semantic` accessor. Each operation is designed to handle specific types of transformations and analyses using LLMs.

All semantic operations return a new DataFrame that preserves the original columns and adds new columns based on the `output_schema`. For example, if your original DataFrame has a column `text` and you use `map` with an `output_schema={"sentiment": "str", "keywords": "list[str]"}`, the resulting DataFrame will have three columns: `text`, `sentiment`, and `keywords`. This makes it easy to chain operations and maintain data lineage.

## Map Operation

::: docetl.apis.pd_accessors.SemanticAccessor.map
    options:
        show_root_heading: false
        heading_level: 3

Example usage:
```python
df.semantic.map(
    prompt="Extract sentiment and key points from: {{input.text}}",
    output_schema={
        "sentiment": "str",
        "key_points": "list[str]"
    },
    validate=["len(output['key_points']) <= 5"],
    num_retries_on_validate_failure=2
)
```

## Filter Operation

::: docetl.apis.pd_accessors.SemanticAccessor.filter
    options:
        show_root_heading: false
        heading_level: 3

Example usage:
```python
# Simple filtering
df.semantic.filter(
    prompt="Is this text about technology? {{input.text}}"
)

# Custom output schema with reasons
df.semantic.filter(
    prompt="Analyze if this is relevant: {{input.text}}",
    output_schema={
        "keep": "bool",
        "reason": "str"
    }
)
```

## Merge Operation (Experimental)

> **Note**: The merge operation is an experimental feature based on our equijoin operator. It provides a pandas-like interface for semantic record matching and deduplication. When `fuzzy=True`, it automatically invokes optimization to improve performance while maintaining accuracy.

::: docetl.apis.pd_accessors.SemanticAccessor.merge
    options:
        show_root_heading: false
        heading_level: 3

Example usage:
```python
# Simple merge
merged_df = df1.semantic.merge(
    df2,
    comparison_prompt="Are these records about the same entity? {{input1}} vs {{input2}}"
)

# Fuzzy merge with optimization
merged_df = df1.semantic.merge(
    df2,
    comparison_prompt="Compare: {{input1}} vs {{input2}}",
    fuzzy=True,
    target_recall=0.9
)
```

## Aggregate Operation

::: docetl.apis.pd_accessors.SemanticAccessor.agg
    options:
        show_root_heading: false
        heading_level: 3

Example usage:
```python
# Simple aggregation
df.semantic.agg(
    reduce_prompt="Summarize these items: {{input.text}}",
    output_schema={"summary": "str"}
)

# Fuzzy matching with custom resolution
df.semantic.agg(
    reduce_prompt="Combine these items: {{input.text}}",
    output_schema={"combined": "str"},
    fuzzy=True,
    comparison_prompt="Are these items similar: {{input1.text}} vs {{input2.text}}",
    resolution_prompt="Resolve conflicts between: {{items}}",
    resolution_output_schema={"resolved": "str"}
)
```

## Common Features

All operations support:

1. **Cost Tracking**
```python
# After any operation
print(f"Operation cost: ${df.semantic.total_cost}")
```

2. **Operation History**
```python
# View operation history
for op in df.semantic.history:
    print(f"{op.op_type}: {op.output_columns}")
```

3. **Validation Rules**
```python
# Add validation rules to any  map or filter operation
validate=["len(output['tags']) <= 5", "output['score'] >= 0"]
```




For more details on configuration options and best practices, refer to:
- [DocETL Best Practices](../best-practices.md)
- [Pipeline Configuration](../concepts/pipelines.md)
- [Output Schemas](../concepts/schemas.md) 