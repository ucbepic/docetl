# Python API

DocETL's Python API lets you build, run, and optimize LLM-powered data pipelines with chainable operations — similar to PySpark or pandas.

## Quick Start

```python
import docetl

docetl.default_model = "gpt-4o-mini"

results = (
    docetl.read_json("input.json")
    .map(
        prompt="Classify this document: {{ input.text }}",
        output={"schema": {"category": "string"}},
    )
    .filter(prompt="Is this document about technology? {{ input.text }}")
    .reduce(
        reduce_key="category",
        prompt="Summarize these documents: {% for item in inputs %}{{ item.text }}{% endfor %}",
        output={"schema": {"summary": "string"}},
    )
    .collect()
)

print(results)  # pandas DataFrame
```

## Configuration

Set global defaults as module-level attributes:

```python
import docetl

# Model selection
docetl.default_model = "gpt-4o-mini"
docetl.agent_model = "gpt-4o"                          # model for optimizer rewrites
docetl.fallback_models = ["gpt-4o", "gpt-4o-mini"]     # fallback chain on failure
docetl.fallback_embedding_models = ["text-embedding-3-small"]

# Execution
docetl.max_threads = 64            # concurrent threads (default: cpu_count * 4)
docetl.bypass_cache = True         # skip LLM cache
docetl.intermediate_dir = ".cache" # intermediate results directory

# Rate limiting
docetl.rate_limits = {
    "llm_call": [{"count": 10, "per": 1, "unit": "second"}]
}
```

## Reading Data

```python
# From files
frame = docetl.read_json("data.json")
frame = docetl.read_csv("data.csv")
frame = docetl.read_parquet("data.parquet")

# From memory
frame = docetl.from_list([
    {"text": "First document", "id": 1},
    {"text": "Second document", "id": 2},
])

# From a YAML pipeline
frame = docetl.Frame.from_yaml("pipeline.yaml")
```

## Operations

All operations return a new `Frame` (immutable). Chain them freely:

```python
frame = docetl.read_json("input.json")

# LLM-powered operations
frame = frame.map(prompt="...", output={"schema": {"field": "type"}})
frame = frame.parallel_map(prompt="...", output={"schema": {"field": "type"}})
frame = frame.filter(prompt="...", output={"schema": {"keep": "bool"}})
frame = frame.reduce(reduce_key="col", prompt="...", output={"schema": {"result": "str"}})
frame = frame.resolve(comparison_prompt="...", output={"schema": {"resolved": "str"}})
frame = frame.extract(prompt="...", output={"schema": {"extracted": "str"}})

# Joining two datasets
right = docetl.read_json("other.json")
frame = frame.equijoin(right, comparison_prompt="...")

# Structural operations (no LLM calls)
frame = frame.split(split_key="text", method="token_count", method_kwargs={"num_tokens": 500})
frame = frame.gather(content_key="chunk", doc_id_key="doc_id", order_key="chunk_num")
frame = frame.unnest(unnest_key="items")
frame = frame.cluster(embedding_keys=["text"])
frame = frame.sample(samples={"category": {"n": 5}}, random=True)

# Code operations (Python functions, no LLM)
frame = frame.code_map(code="def transform(doc): return {'word_count': len(doc['text'].split())}")
frame = frame.code_filter(code="def keep(doc): return len(doc['text']) > 100")
frame = frame.code_reduce(reduce_key="category", code="def aggregate(items): ...")
```

## Retrievers

Augment LLM operations with context retrieved from a LanceDB index. Create a `Retriever` object and pass it directly to operations:

```python
retriever = docetl.Retriever(
    dataset="kb",                       # dataset name to index
    index_dir="./lance_index",
    index_types=["fts", "embedding"],
    fts={
        "index_phrase": "{{ input.text }}",
        "query_phrase": "{{ input.question }}",
    },
    embedding={
        "model": "text-embedding-3-small",
        "index_phrase": "{{ input.text }}",
        "query_phrase": "{{ input.question }}",
    },
    query={"mode": "hybrid", "top_k": 5},
)

df = (
    docetl.read_json("queries.json")
    .map(
        prompt="Answer: {{ input.question }}\nContext: {{ retrieval_context }}",
        output={"schema": {"answer": "str"}},
        retriever=retriever,
    )
    .collect()
)
```

The `retriever` parameter is available on `map`, `filter`, `reduce`, and `extract`. The retrieved context is injected as `{{ retrieval_context }}` in your prompt template.

## Inspection (no execution)

```python
frame.schema()        # {'category': 'str', 'summary': 'str', ...}
frame.count()         # number of input docs (or output rows if ops are present)
frame.to_yaml()       # export pipeline as YAML config string
frame.to_yaml("pipeline.yaml")  # also write to file
frame.to_python()     # export as Python source code
```

## Terminal Actions

```python
# Preview — run on a small sample and print results
frame.show()          # default: 5 input documents
frame.show(max=10)    # custom sample size

# Also works on bare datasets
docetl.read_json("data.json").show()

# Collect as DataFrame
df = frame.collect()

# Collect as list of dicts
data = frame.to_list()

# Write directly to file
frame.write_json("output.json")
frame.write_csv("output.csv")
frame.write_parquet("output.parquet")
```

## Cost & Token Tracking

After execution, cost and token usage are available:

```python
df = frame.collect()

# On the Frame object
print(f"Cost: ${frame.total_cost:.4f}")
print(f"Tokens: {frame.token_usage}")

# Also stored on the DataFrame
print(f"Cost: ${df.attrs['_total_cost']:.4f}")
```

Write methods return the cost directly:

```python
cost = frame.write_json("output.json")
print(f"Cost: ${cost:.4f}")
```

## Optimization

Optimize a pipeline with [MOAR](../optimization/moar.md) (Multi-Objective Agentic Rewrites):

```python
import docetl

@docetl.register_eval
def my_eval(results):
    # score the pipeline output
    return {"accuracy": compute_accuracy(results)}

optimized = (
    docetl.read_json("input.json")
    .map(prompt="...", output={"schema": {"summary": "str"}})
    .optimize(
        eval_fn=my_eval,
        metric_key="accuracy",
        models=["gpt-4o-mini", "gpt-4o"],
        max_iterations=20,
        save_dir="optimization_results",
    )
)

# Run the optimized pipeline
df = optimized.collect()

# Inspect search results
print(optimized.search_results.to_df())
print(optimized.search_results.best())
print(optimized.search_results.cheapest())
```

## Code Generation

Generate Python source from a Frame or convert a YAML pipeline:

```python
# From a Frame
code = frame.to_python()
print(code)

# From a YAML file
code = docetl.yaml_to_python("pipeline.yaml")
print(code)
```

## API Reference

For the complete reference of all operation parameters, see the [API Reference](../api-reference/python.md).
