# Python API Reference

The Python API lets you define, optimize, and run DocETL pipelines programmatically. All classes are importable from `docetl.api`.

```python
from docetl.api import (
    Pipeline, Dataset, PipelineStep, PipelineOutput,
    MapOp, ReduceOp, ResolveOp, FilterOp, ParallelMapOp,
    EquijoinOp, SplitOp, GatherOp, UnnestOp, SampleOp,
    CodeMapOp, CodeReduceOp, CodeFilterOp, ExtractOp,
)
```

---

## Pipeline

The main class for defining and running a complete document processing pipeline.

```python
Pipeline(
    name: str,
    datasets: dict[str, Dataset],
    operations: list[OpType],
    steps: list[PipelineStep],
    output: PipelineOutput,
    default_model: str | None = None,
    parsing_tools: list[ParsingTool] = [],
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Name of the pipeline. |
| `datasets` | `dict[str, Dataset]` | Datasets keyed by name. |
| `operations` | `list[OpType]` | List of operation definitions. |
| `steps` | `list[PipelineStep]` | Ordered steps to execute. |
| `output` | `PipelineOutput` | Output configuration. |
| `default_model` | `str \| None` | Default LLM model for all operations. |
| `parsing_tools` | `list[ParsingTool]` | Custom parsing functions. Can be `ParsingTool` objects or plain Python functions. |

**Methods:**

- `pipeline.run() -> float` — Execute the pipeline. Returns total cost.
- `pipeline.optimize(**kwargs) -> Pipeline` — Return an optimized copy of the pipeline.

---

## Dataset

```python
Dataset(
    type: "file" | "memory",
    path: str | list[dict] | pd.DataFrame,
    source: str = "local",
    parsing: list[dict[str, str]] | None = None,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `type` | `"file"` or `"memory"` | `"file"` to load from disk, `"memory"` for in-memory data. |
| `path` | `str \| list[dict] \| DataFrame` | File path (for `"file"` type) or data (for `"memory"` type). |
| `source` | `str` | Source identifier. Defaults to `"local"`. |
| `parsing` | `list[dict]` | Parsing instructions. Each dict has `input_key`, `function`, and `output_key`. |

---

## PipelineStep

```python
PipelineStep(
    name: str,
    input: str | None,
    operations: list[str | dict],
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Step name. |
| `input` | `str \| None` | Name of a dataset or previous step to use as input. |
| `operations` | `list[str \| dict]` | Operation names (or dicts for more complex configs) to run in this step. |

---

## PipelineOutput

```python
PipelineOutput(
    type: str,
    path: str,
    intermediate_dir: str | None = None,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `type` | `str` | Output type (e.g., `"file"`). |
| `path` | `str` | Path to write output. |
| `intermediate_dir` | `str \| None` | Directory for intermediate results. |

---

## LLM-Powered Operations

All operations share these base fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | *required* | Unique operation name. |
| `type` | `str` | *required* | Operation type (must match the Op class). |
| `skip_on_error` | `bool` | `False` | Skip documents that cause errors. |

### MapOp

Applies an LLM prompt to each document independently.

```python
MapOp(name="...", type="map", prompt="...", output={"schema": {...}})
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | `str` | — | Jinja2 template. Use `{{ input.key }}` to access fields. |
| `output` | `dict` | — | Output schema, e.g. `{"schema": {"field": "string"}}`. |
| `model` | `str` | `None` | Override the default model. |
| `drop_keys` | `list[str]` | `None` | Keys to drop from output. |
| `batch_size` | `int` | `None` | Process documents in batches. |
| `batch_prompt` | `str` | `None` | Jinja2 template for batch processing. Uses `{{ inputs }}`. |
| `timeout` | `int` | `None` | Timeout in seconds per LLM call. |
| `optimize` | `bool` | `None` | Mark for optimization. |
| `limit` | `int` | `None` | Max documents to process. |
| `litellm_completion_kwargs` | `dict` | `{}` | Extra kwargs passed to litellm. |
| `enable_observability` | `bool` | `False` | Enable observability logging. |

### ReduceOp

Groups documents by key(s) and reduces each group with an LLM.

```python
ReduceOp(name="...", type="reduce", reduce_key="key", prompt="...", output={"schema": {...}})
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `reduce_key` | `str \| list[str]` | *required* | Key(s) to group by. Use `"_all"` for a single group. |
| `prompt` | `str` | *required* | Jinja2 template. Use `{% for item in inputs %}` to iterate. |
| `output` | `dict` | *required* | Output schema. |
| `model` | `str` | `None` | Override the default model. |
| `input` | `dict` | `None` | Input schema constraints. |
| `pass_through` | `bool` | `None` | Pass through non-reduced keys. |
| `associative` | `bool` | `None` | Whether reduce is associative (enables parallelism). |
| `fold_prompt` | `str` | `None` | Prompt for incremental fold operations. |
| `fold_batch_size` | `int` | `None` | Batch size for fold. |
| `merge_prompt` | `str` | `None` | Prompt for merging fold results. |
| `merge_batch_size` | `int` | `None` | Batch size for merge. |
| `optimize` | `bool` | `None` | Mark for optimization. |
| `timeout` | `int` | `None` | Timeout in seconds. |
| `limit` | `int` | `None` | Max groups to process. |
| `litellm_completion_kwargs` | `dict` | `{}` | Extra kwargs passed to litellm. |

### ResolveOp

Deduplicates/resolves entities by comparing pairs of documents.

```python
ResolveOp(name="...", type="resolve", comparison_prompt="...", resolution_prompt="...")
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `comparison_prompt` | `str` | *required* | Jinja2 template comparing `{{ input1 }}` and `{{ input2 }}`. |
| `resolution_prompt` | `str` | `None` | Prompt for resolving matched pairs. |
| `output` | `dict` | `None` | Output schema. |
| `embedding_model` | `str` | `None` | Model for blocking embeddings. |
| `comparison_model` | `str` | `None` | Model for comparisons. |
| `resolution_model` | `str` | `None` | Model for resolution. |
| `blocking_keys` | `list[str]` | `None` | Keys to use for blocking. |
| `blocking_threshold` | `float` | `None` | Similarity threshold for blocking (0–1). |
| `blocking_target_recall` | `float` | `None` | Target recall for blocking (0–1). |
| `blocking_conditions` | `list[str]` | `None` | Custom blocking conditions. |
| `optimize` | `bool` | `None` | Mark for optimization. |
| `timeout` | `int` | `None` | Timeout in seconds. |
| `litellm_completion_kwargs` | `dict` | `{}` | Extra kwargs passed to litellm. |

### FilterOp

Filters documents using an LLM prompt that returns a boolean.

```python
FilterOp(name="...", type="filter", prompt="...", output={"schema": {...}})
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | `str` | *required* | Jinja2 template. Use `{{ input.key }}`. |
| `output` | `dict` | *required* | Must include a boolean field in schema. |

### ParallelMapOp

Runs multiple prompts on each document in parallel.

```python
ParallelMapOp(name="...", type="parallel_map", prompts=[...], output={"schema": {...}})
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompts` | `list[dict]` | — | List of prompt configurations. |
| `output` | `dict` | — | Combined output schema. |
| `drop_keys` | `list[str]` | `None` | Keys to drop from output. |

### EquijoinOp

Joins two datasets by comparing document pairs with an LLM.

```python
EquijoinOp(name="...", type="equijoin", comparison_prompt="...")
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `comparison_prompt` | `str` | *required* | Jinja2 template comparing `{{ left }}` and `{{ right }}`. |
| `output` | `dict` | `None` | Output schema. |
| `blocking_keys` | `dict[str, list[str]]` | `None` | Keys for blocking per dataset. |
| `blocking_threshold` | `float` | `None` | Similarity threshold. |
| `blocking_conditions` | `list[str]` | `None` | Custom blocking conditions. |
| `limits` | `dict[str, int]` | `None` | Max matches per side. |
| `comparison_model` | `str` | `None` | Model for comparisons. |
| `embedding_model` | `str` | `None` | Model for embeddings. |
| `optimize` | `bool` | `None` | Mark for optimization. |
| `timeout` | `int` | `None` | Timeout in seconds. |
| `litellm_completion_kwargs` | `dict` | `{}` | Extra kwargs passed to litellm. |

### ExtractOp

Extracts specific information from documents with line-level precision.

```python
ExtractOp(name="...", type="extract", prompt="...", document_keys=["content"])
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | `str` | *required* | Extraction prompt. |
| `document_keys` | `list[str]` | *required* | Keys containing document text. |
| `model` | `str` | `None` | Override the default model. |
| `format_extraction` | `bool` | `True` | Format extracted content. |
| `extraction_key_suffix` | `str` | `None` | Suffix for extraction output keys. |
| `extraction_method` | `"line_number" \| "regex"` | `"line_number"` | Extraction method. |
| `timeout` | `int` | `None` | Timeout in seconds. |
| `limit` | `int` | `None` | Max documents to process. |
| `litellm_completion_kwargs` | `dict` | `{}` | Extra kwargs passed to litellm. |

---

## Auxiliary Operations

### SplitOp

Splits documents into chunks.

```python
SplitOp(name="...", type="split", split_key="content", method="token_count", method_kwargs={"num_tokens": 500})
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `split_key` | `str` | *required* | Key containing text to split. |
| `method` | `str` | *required* | Split method (e.g., `"token_count"`, `"delimiter"`). |
| `method_kwargs` | `dict` | *required* | Arguments for the split method. |
| `model` | `str` | `None` | Model for token counting. |

### GatherOp

Adds surrounding context to chunks created by split.

```python
GatherOp(name="...", type="gather", content_key="...", doc_id_key="...", order_key="...")
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `content_key` | `str` | *required* | Key with chunk content. |
| `doc_id_key` | `str` | *required* | Key identifying the source document. |
| `order_key` | `str` | *required* | Key for chunk ordering. |
| `peripheral_chunks` | `dict` | `None` | Configuration for surrounding context. |
| `doc_header_key` | `str` | `None` | Key for document headers. |

### UnnestOp

Flattens a list-valued field into separate documents.

```python
UnnestOp(name="...", type="unnest", unnest_key="items")
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `unnest_key` | `str` | *required* | Key containing the list to unnest. |
| `keep_empty` | `bool` | `None` | Keep documents with empty lists. |
| `expand_fields` | `list[str]` | `None` | Additional fields to expand. |
| `recursive` | `bool` | `None` | Recursively unnest nested lists. |
| `depth` | `int` | `None` | Max recursion depth. |

### SampleOp

Samples a subset of documents.

```python
SampleOp(name="...", type="sample", method="uniform", samples=100)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `method` | `str` | *required* | One of `"uniform"`, `"outliers"`, `"custom"`, `"first"`, `"top_embedding"`, `"top_fts"`. |
| `samples` | `int \| float \| list` | `None` | Number of samples or fraction. |
| `stratify_key` | `str \| list[str]` | `None` | Key(s) for stratified sampling. |
| `samples_per_group` | `bool` | `False` | Apply sample count per group. |
| `method_kwargs` | `dict` | `{}` | Extra arguments for the sampling method. |
| `random_state` | `int` | `None` | Random seed for reproducibility. |

---

## Code Operations

Code operations run Python functions instead of LLM calls. The `code` parameter accepts either a string containing Python code that defines a `transform` function, or a regular Python function.

```python
def my_transform(doc: dict) -> dict:
    return {"doubled": doc["value"] * 2}

op = CodeMapOp(name="double", type="code_map", code=my_transform)
```

### CodeMapOp

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `code` | `str \| Callable` | *required* | `fn(doc: dict) -> dict` |
| `drop_keys` | `list[str]` | `None` | Keys to drop from output. |
| `limit` | `int` | `None` | Max documents to process. |

### CodeReduceOp

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `code` | `str \| Callable` | *required* | `fn(group: list[dict]) -> dict` |
| `limit` | `int` | `None` | Max groups to process. |

### CodeFilterOp

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `code` | `str \| Callable` | *required* | `fn(doc: dict) -> bool` |
| `limit` | `int` | `None` | Max documents to process. |
