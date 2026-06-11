# Python API Examples

## Example 1: Document Chunking with Context

Split long documents into chunks, add surrounding context, then extract structured information:

```python
import docetl

docetl.default_model = "gpt-4o-mini"

df = (
    docetl.read_json("papers.json")
    .split(
        split_key="full_text",
        method="delimiter",
        method_kwargs={"delimiter": "\n\n", "num_splits_to_group": 2},
    )
    .gather(
        content_key="full_text_chunk",
        doc_id_key="split_0_id",
        order_key="split_0_chunk_num",
        peripheral_chunks={
            "previous": {"head": {"count": 1}},
            "next": {"head": {"count": 1}},
        },
    )
    .map(
        prompt="""Analyze this paper section with its surrounding context:

        Paper: {{ input.title }}
        Section: {{ input.full_text_chunk_rendered }}

        Extract the section type, key findings, and technical concepts.""",
        output={"schema": {
            "section_type": "str",
            "key_findings": "list[str]",
            "technical_concepts": "list[str]",
        }},
    )
    .reduce(
        reduce_key="paper_id",
        prompt="""Create a comprehensive analysis of this paper:

        {% for section in inputs %}
        {{ section.section_type }}: {{ section.key_findings | join(", ") }}
        {% endfor %}""",
        output={"schema": {
            "summary": "str",
            "main_contributions": "list[str]",
        }},
    )
    .collect()
)

print(df)
```

## Example 2: Fuzzy Aggregation with the Pandas Accessor

The pandas `.semantic` accessor runs operations on existing DataFrames:

```python
import pandas as pd
import docetl

docetl.default_model = "gpt-4o-mini"

posts = pd.DataFrame({
    "text": [
        "Just tried the new iPhone 15!",
        "Having issues with iOS 17",
        "Android is way better",
    ],
    "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
})

# Extract structured data
analyzed = posts.semantic.map(
    prompt="""Extract product and sentiment from: {{ input.text }}""",
    output={"schema": {"product": "str", "sentiment": "str"}},
)

# Filter
relevant = analyzed.semantic.filter(
    prompt="Is this about Apple products? {{ input }}"
)

# Fuzzy group-by and summarize
summaries = relevant.semantic.agg(
    fuzzy=True,
    reduce_keys=["product"],
    comparison_prompt="Do these posts discuss the same product?",
    reduce_prompt="Summarize the feedback about this product",
    output={"schema": {"summary": "str", "frequency": "int"}},
)

print(f"Cost: ${summaries.semantic.total_cost:.4f}")
print(summaries)
```

Datasets can be JSON, CSV, or Parquet.
