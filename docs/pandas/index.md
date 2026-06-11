# Pandas Accessor

The `.semantic` accessor runs DocETL operations directly on pandas DataFrames.
It is a convenience layer over the [Python API](../api-reference/python.md) for quick,
single-operation work; for multi-step pipelines (and pipeline optimization),
use Frames.

```bash
pip install docetl
```

## Quick example

```python
import pandas as pd
import docetl

docetl.default_model = "gpt-4o-mini"

df = pd.DataFrame({"text": [
    "Apple released the iPhone 15 with USB-C port",
    "Microsoft's new Surface laptops feature AI capabilities",
]})

result = df.semantic.map(
    prompt="Extract company and product from: {{input.text}}",
    output={"schema": {"company": "str", "product": "str"}},
)
print(f"Cost: ${result.semantic.total_cost}")
```

Configuration uses the same `docetl.*` globals as the Python API — see
[Configuration](../api-reference/python.md#configuration). Prompts are Jinja
templates over `{{input.<column>}}`; output schemas are documented in
[Output Schemas](../concepts/schemas.md).

## Operations

### map

```python
df.semantic.map(
    prompt="Extract entities from: {{input.text}}",
    output={"schema": {"entities": "list[str]"}},
)
```

### filter

```python
df.semantic.filter(
    prompt="Is this about technology? {{input.text}}",
)  # default output schema: {"keep": "bool"}
```

### merge

Semantic join of two DataFrames. With `fuzzy=True`, blocking is configured
automatically to reduce comparisons:

```python
merged = df1.semantic.merge(
    df2,
    comparison_prompt="Are these the same entity? {{input1}} vs {{input2}}",
    fuzzy=True,
    target_recall=0.9,
)
```

### agg

Group and reduce. With `fuzzy=True`, similar group keys are resolved first:

```python
df.semantic.agg(
    reduce_prompt="Summarize these items: {{input.text}}",
    output={"schema": {"summary": "str"}},
    fuzzy=True,
    comparison_prompt="Are these similar? {{input1.text}} vs {{input2.text}}",
)
```

### split / gather / unnest

No LLM calls:

```python
df.semantic.split(split_key="content", method="token_count",
                  method_kwargs={"num_tokens": 100})

df.semantic.gather(content_key="content_chunk", doc_id_key="split_id",
                   order_key="split_chunk_num")

df.semantic.unnest(unnest_key="tags")
```

## Cost and history

```python
result.semantic.total_cost   # dollars spent across accessor operations
result.semantic.history      # list of (op_type, config, output_columns)
```

`map` and `filter` accept `validate=` with Python expressions, e.g.
`validate=["len(output['tags']) <= 5"]`.

## Limits

Accessor calls execute one operation at a time, so sequences of them cannot be
optimized as a pipeline. Use the [Python API](../api-reference/python.md) or YAML for
pipeline-level optimization.
