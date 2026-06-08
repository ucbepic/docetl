# Python API Examples

## Example 1: Extract and Summarize Product Review Themes

Extract themes from product reviews, then summarize across all documents:

```python
import docetl

docetl.default_model = "gpt-4o-mini"

results = (
    docetl.read_csv("product_reviews.csv")
    .map(
        prompt="""Analyze this product review and extract the key themes and representative quotes:

        Review: {{ input.review_text }}
        Rating: {{ input.rating }}

        Identify 2-3 major themes (e.g., usability, quality, value) and extract
        direct quotes that best represent each theme.""",
        output={"schema": {
            "themes": "list[string]",
            "quotes": "list[string]",
            "sentiment": "string",
        }},
    )
    .reduce(
        reduce_key="_all",
        prompt="""Synthesize themes and quotes from these product reviews:

        {% for item in inputs %}
        Review ID: {{ item.review_id }}
        Themes: {{ item.themes | join(", ") }}
        Quotes: {% for q in item.quotes %}"{{ q }}" {% endfor %}
        Sentiment: {{ item.sentiment }}
        {% endfor %}

        Summarize the most frequent themes and representative quotes.""",
        output={"schema": {"summary": "string"}},
    )
    .collect()
)

print(results)
```

## Example 2: Map → Unnest → Resolve → Reduce

Extract theme-quote pairs, unnest into rows, deduplicate similar themes, then aggregate by theme. This example also runs optimization:

```python
import docetl

docetl.default_model = "gpt-4o"

frame = (
    docetl.read_csv("product_reviews.csv")
    .map(
        prompt="""Extract theme and quote pairs from this product review:

        Review: {{ input.review_text }}
        Product: {{ input.product_name }}
        Rating: {{ input.rating }}

        Return each theme and its representative quote as a separate object
        in the "theme_quotes" array.""",
        output={"schema": {"theme_quotes": "list[{theme: string, quote: string}]"}},
    )
    .unnest(unnest_key="theme_quotes")
    .resolve(
        comparison_prompt="""Are these two themes the same or closely related?

        Theme 1: {{ input1.theme }}
        Theme 2: {{ input2.theme }}""",
        resolution_prompt="""Choose a canonical name for these similar themes:

        {% for item in inputs %}Theme: {{ item.theme }}
        {% endfor %}""",
    )
    .reduce(
        reduce_key="theme",
        prompt="""Summarize all quotes related to "{{ reduce_key }}":

        {% for item in inputs %}
        Product: {{ item.product_name }}, Rating: {{ item.rating }}
        Quote: "{{ item.quote }}"
        {% endfor %}""",
        output={"schema": {"summary": "string"}},
    )
)

# Optionally optimize before running
@docetl.register_eval
def eval_themes(results):
    return {"quality": score_summaries(results)}

optimized = frame.optimize(
    eval_fn=eval_themes,
    metric_key="quality",
    models=["gpt-4o-mini", "gpt-4o"],
    max_iterations=15,
    save_dir="theme_optimization",
)

df = optimized.collect()
print(f"Cost: ${optimized.total_cost:.4f}")
print(df)
```

## Example 3: Document Chunking with Context

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

## Example 4: Fuzzy Aggregation with the Pandas Accessor

The pandas `.semantic` accessor provides a quick way to run operations on existing DataFrames:

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

Note that datasets can be JSON, CSV, or Parquet.
