# Python API Reference

The Python API lets you build, run, and optimize DocETL pipelines with chainable operations.

```python
import docetl
```

---

## Configuration

Set global defaults as module-level attributes:

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `docetl.default_model` | `str` | `None` | Default LLM model for all operations |
| `docetl.agent_model` | `str` | `None` | Model for optimizer rewrites |
| `docetl.max_threads` | `int` | `cpu_count * 4` | Concurrent threads |
| `docetl.bypass_cache` | `bool` | `False` | Skip LLM cache |
| `docetl.intermediate_dir` | `str` | `None` | Directory for intermediate results |
| `docetl.rate_limits` | `dict` | `None` | Rate limits per model |
| `docetl.fallback_models` | `list[str]` | `None` | Fallback chain on failure |
| `docetl.fallback_embedding_models` | `list[str]` | `None` | Fallback embedding models |

---

## Reading Data

| Function | Description |
|----------|-------------|
| `docetl.read_json(path)` | Load from JSON file |
| `docetl.read_csv(path)` | Load from CSV file |
| `docetl.read_parquet(path)` | Load from Parquet file |
| `docetl.from_list(data)` | Load from a list of dicts |
| `docetl.Frame.from_yaml(path)` | Load from a YAML pipeline config |

All return a `Frame`.

---

## Frame

A lazy pipeline — operations are recorded but not executed until a terminal action is called. Frames are immutable; every operation returns a new `Frame`.

### LLM Operations

#### `.map()`

Applies an LLM prompt to each document independently.

```python
frame.map(
    prompt="Classify: {{ input.text }}",
    output={"schema": {"category": "str"}},
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | — | Jinja2 template. Access fields via `{{ input.key }}`. |
| `output` | `dict` | — | Output schema, e.g. `{"schema": {"field": "str"}}` |
| `model` | `str` | `None` | Override default model |
| `validate` | `list[str]` | `None` | Python expressions to validate output |
| `num_retries_on_validate_failure` | `int` | `None` | Retries on validation failure |
| `sample` | `int` | `None` | Process only N documents |
| `tools` | `list[dict]` | `None` | Tool definitions for function calling |
| `drop_keys` | `list[str]` | `None` | Keys to remove from output |
| `timeout` | `int` | `None` | Timeout per LLM call (seconds) |
| `max_batch_size` | `int` | `None` | Batch size for batch processing |
| `batch_prompt` | `str` | `None` | Jinja2 template for batch mode |
| `retriever` | `Retriever` | `None` | Retriever for context augmentation |
| `optimize` | `bool` | `None` | Mark for optimization |
| `limit` | `int` | `None` | Max documents to process |

#### `.filter()`

Keeps or removes documents based on an LLM prompt. Output schema must have one boolean field.

```python
frame.filter(
    prompt="Is this about technology? {{ input.text }}",
    output={"schema": {"is_tech": "bool"}},
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | — | Jinja2 template |
| `output` | `dict` | — | Schema with one boolean field |
| `model` | `str` | `None` | Override default model |
| `validate` | `list[str]` | `None` | Validation expressions |
| `retriever` | `Retriever` | `None` | Retriever for context augmentation |

#### `.reduce()`

Groups documents by key and reduces each group with an LLM.

```python
frame.reduce(
    reduce_key="category",
    prompt="Summarize: {% for item in inputs %}{{ item.text }}{% endfor %}",
    output={"schema": {"summary": "str"}},
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reduce_key` | `str \| list[str]` | — | Key(s) to group by. Use `"_all"` for one group. |
| `prompt` | `str` | — | Jinja2 template. Iterate with `{% for item in inputs %}`. |
| `output` | `dict` | — | Output schema |
| `fold_prompt` | `str` | `None` | Prompt for incremental folding |
| `fold_batch_size` | `int` | `None` | Items per fold iteration |
| `merge_prompt` | `str` | `None` | Prompt for merging fold results |
| `pass_through` | `bool` | `None` | Pass through non-reduced keys |
| `associative` | `bool` | `None` | Enable parallel reduction |
| `retriever` | `Retriever` | `None` | Retriever for context augmentation |

#### `.resolve()`

Deduplicates entities by pairwise LLM comparison.

```python
frame.resolve(
    comparison_prompt="Same person? {{ input1.name }} vs {{ input2.name }}",
    resolution_prompt="Canonical name: {% for e in inputs %}{{ e.name }}{% endfor %}",
    output={"schema": {"name": "str"}},
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `comparison_prompt` | `str` | — | Jinja2 template comparing `{{ input1 }}` and `{{ input2 }}` |
| `resolution_prompt` | `str` | `None` | Prompt for resolving matched groups |
| `output` | `dict` | `None` | Output schema |
| `blocking_keys` | `list[str]` | `None` | Keys for blocking |
| `blocking_threshold` | `float` | `None` | Similarity threshold |
| `embedding_model` | `str` | `None` | Model for blocking embeddings |
| `optimize` | `bool` | `None` | Mark for optimization |

#### `.extract()`

Extracts information from documents with line-level precision.

```python
frame.extract(
    prompt="Extract key findings from this paper.",
    document_keys=["content"],
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | — | Extraction prompt |
| `document_keys` | `list[str]` | — | Keys containing document text |
| `retriever` | `Retriever` | `None` | Retriever for context augmentation |

#### `.parallel_map()`

Runs multiple prompts on each document in parallel.

```python
frame.parallel_map(
    prompts=[{"prompt": "...", "output_keys": ["field1"]}, ...],
    output={"schema": {"field1": "str", "field2": "str"}},
)
```

#### `.equijoin()`

Joins two datasets by LLM comparison.

```python
right = docetl.read_json("other.json")
frame.equijoin(right, comparison_prompt="Are these related? {{ left.x }} {{ right.y }}")
```

### Structural Operations (no LLM calls)

| Method | Description |
|--------|-------------|
| `.split(split_key, method, method_kwargs)` | Split documents into chunks |
| `.gather(content_key, doc_id_key, order_key)` | Add surrounding context to chunks |
| `.unnest(unnest_key)` | Flatten a list field into separate rows |
| `.cluster(embedding_keys)` | Cluster documents by embedding similarity |
| `.sample(samples, method)` | Sample a subset of documents |

### Code Operations (no LLM calls)

| Method | Description |
|--------|-------------|
| `.code_map(code="def transform(doc): ...")` | Per-document Python transform |
| `.code_filter(code="def transform(doc): ...")` | Per-document Python filter (return bool) |
| `.code_reduce(reduce_key, code="def transform(items): ...")` | Per-group Python aggregation |

---

## Retriever

Augment LLM operations with retrieved context from a LanceDB index.

```python
retriever = docetl.Retriever(
    dataset="knowledge_base",
    index_dir="./lance_index",
    index_types=["fts", "embedding"],
    fts={"index_phrase": "{{ input.text }}", "query_phrase": "{{ input.question }}"},
    embedding={"model": "text-embedding-3-small", "index_phrase": "...", "query_phrase": "..."},
    query={"mode": "hybrid", "top_k": 5},
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `str` | — | Name of the dataset to index |
| `index_dir` | `str` | — | Directory for the LanceDB index |
| `index_types` | `list[str]` | — | `["fts"]`, `["embedding"]`, or both |
| `fts` | `dict` | `None` | Full-text search config (`index_phrase`, `query_phrase`) |
| `embedding` | `dict` | `None` | Embedding config (`model`, `index_phrase`, `query_phrase`) |
| `query` | `dict` | `None` | Query config (`mode`, `top_k`) |
| `build_index` | `str` | `"if_missing"` | `"if_missing"`, `"always"`, or `"never"` |

Pass to any LLM operation via `retriever=`. Retrieved context is available as `{{ retrieval_context }}` in prompts.

`dataset` must name a dataset visible to the pipeline: an auxiliary dataset registered with `.with_dataset()`, the frame's own input dataset, or a previous step's output (`step_<operation_name>`). See the [Retrievers guide](../retrievers.md) for details.

### `Frame.with_dataset(name, data, *, parsing=None)`

Register an auxiliary dataset (e.g. a retriever's knowledge base) alongside the frame's input. `data` is a file path (JSON/CSV/Parquet) or an in-memory list of dicts. The dataset does not flow through the operation chain.

```python
frame = (
    docetl.read_json("questions.json")
    .with_dataset("kb", "knowledge_base.json")
    .map(prompt="...", output={...}, retriever=docetl.Retriever(dataset="kb", ...))
)
```

---

## Inspection (no execution)

| Method | Return Type | Description |
|--------|-------------|-------------|
| `frame.schema()` | `dict[str, str]` | Output schema from operation definitions, including structural ops (split/unnest/gather/extract). Best-effort: `code_*` op outputs can't be known statically |
| `frame.count()` | `int` | Input count (no ops) or output count (executes if ops present) |
| `frame.to_yaml()` | `str` | Export pipeline as YAML config |
| `frame.to_yaml(path)` | `str` | Also write YAML to file |
| `frame.to_python()` | `str` | Export as Python source code |

---

## Terminal Actions

| Method | Return Type | Description |
|--------|-------------|-------------|
| `frame.show(max=5)` | `DataFrame` | Run on a sample and print results. Works on bare datasets too. |
| `frame.collect()` | `DataFrame` | Execute full pipeline, return DataFrame |
| `frame.to_list()` | `list[dict]` | Execute full pipeline, return list of dicts |
| `frame.write_json(path)` | `float` | Execute and write to JSON. Returns cost. |
| `frame.write_csv(path)` | `float` | Execute and write to CSV. Returns cost. |
| `frame.write_parquet(path)` | `float` | Execute and write to Parquet. Returns cost. |

Terminal actions are memoized on the Frame: repeated calls with an unchanged configuration reuse the previous result instead of re-running the pipeline. Changing ops, in-memory data, or `docetl.*` settings invalidates the memo; edits to input *files* between calls are not detected.

### Cost & Token Tracking

```python
df = frame.collect()

print(f"Cost: ${frame.total_cost:.4f}")
print(f"Tokens: {frame.token_usage}")
print(f"Cost: ${df.attrs['_total_cost']:.4f}")  # also on DataFrame
```

---

## Optimization

```python
@docetl.register_eval
def evaluate(results):
    correct = sum(1 for r in results if r.get("correct"))
    return {"score": correct}

optimized = frame.optimize(
    eval_fn=evaluate,
    metric_key="score",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eval_fn` | `Callable` | **required** | Scores pipeline output. Returns a dict of metrics. |
| `metric_key` | `str` | **required** | Key in eval_fn's return dict to optimize |
| `models` | `list[str]` | Auto-detect | LiteLLM model names to explore |
| `agent_model` | `str` | Auto-select | Model for rewrite agent |
| `max_iterations` | `int` | `20` | Search budget |
| `save_dir` | `str` | Temp dir | Where to save results |
| `exploration_weight` | `float` | `1.414` | UCB exploration constant |
| `dataset_path` | `str` | Pipeline's dataset | Sample dataset for optimization |
| `max_threads` | `int` | `None` | Max concurrent LLM calls per run |
| `max_concurrent_agents` | `int` | `3` | Parallel MCTS search agents |

Returns an optimized `Frame`. Access search results via `optimized.search_results`:

| Method / Property | Return Type | Description |
|-------------------|-------------|-------------|
| `.best()` | `OptimizedPipeline` | Highest-accuracy solution |
| `.cheapest()` | `OptimizedPipeline` | Lowest-cost solution |
| `.frontier` | `list[OptimizedPipeline]` | All Pareto-optimal solutions |
| `.to_df()` | `DataFrame` | All explored plans |
