## Retrievers (LanceDB OSS)

Retrievers let you augment LLM operations with retrieved context from a LanceDB index built over one of your DocETL datasets. You define retrievers once at the top-level, then attach them to any LLM-powered operation using `retriever: <name>`. At runtime, DocETL performs full‑text, vector, or hybrid search and injects the results into your prompt as `{{ retrieval_context }}`.

LanceDB supports built-in full-text search, vector search, and hybrid with RRF reranking. See the official docs: [LanceDB Hybrid Search docs](https://lancedb.com/docs/search/hybrid-search/).

### Key points
- Always OSS LanceDB (local `index_dir`).
- A retriever references an existing dataset from the pipeline config.
- Operations do not override retriever settings. One source of truth = consistency.
- `{{ retrieval_context }}` is available to your prompt; if not used, DocETL prepends a short “extra context” section automatically.

## Configuration (clear separation of index vs query)

Add a top-level `retrievers` section. Each retriever has:
- `dataset`: dataset name to index
- `index_dir`: LanceDB path
- `index_types`: which indexes to build: `fts`, `embedding`, or `hybrid` (interpreted as both `fts` and `embedding`)
- `fts.index_phrase`: Jinja for how to index each dataset row for FTS (context: `input`)
- `fts.query_phrase`: Jinja for how to build the FTS query (context: operation context)
- `embedding.model`: embedding model used for the vector index and for query vectors
- `embedding.index_phrase`: Jinja for how to index each dataset row for embedding (context: `input`)
- `embedding.query_phrase`: Jinja for how to build the embedding query text (context: operation context)
- `query.mode`: `fts` | `embedding` | `hybrid` (defaults to `hybrid` when both indexes exist)
- `query.top_k`: number of results to retrieve

```yaml
datasets:
  transcripts:
    type: file
    path: workloads/medical/raw.json

default_model: gpt-4o-mini

retrievers:
  medical_r:
    type: lancedb
    dataset: transcripts
    index_dir: workloads/medical/lance_index
    build_index: if_missing            # if_missing | always | never
    index_types: ["fts", "embedding"]  # or "hybrid"
    fts:
      # How to index each row (context: input == dataset row)
      index_phrase: >
        {{ input.src }}
      # How to build the query (map/filter/extract context: input; reduce: reduce_key & inputs)
      query_phrase: >
        {{ input.get("src","")[:1000] if input else "" }}
    embedding:
      model: openai/text-embedding-3-small
      # How to index each row for embedding (context: input == dataset row)
      index_phrase: >
        {{ input.src }}
      # How to build the query text to embed (op context)
      query_phrase: >
        {{ input.get("src","")[:1000] if input else "" }}
    query:
      mode: hybrid
      top_k: 8
```

Notes:
- Index build is automatic the first time a retriever is used (when `build_index: if_missing`).
- `fts.index_phrase` and `embedding.index_phrase` are evaluated with `input` for each dataset record (here `input` is the dataset row).
- `fts.query_phrase` and `embedding.query_phrase` are evaluated with the operation context.

## Configuration reference

Top-level (retrievers.<name>):

| Parameter | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| type | string | yes | - | Must be `lancedb`. |
| dataset | string | yes | - | Name of an existing dataset in the pipeline config. |
| index_dir | string | yes | - | Filesystem path for the LanceDB database. Created if missing. |
| build_index | enum | no | `if_missing` | `if_missing` \| `always` \| `never`. Controls when to build the index. |
| index_types | list[string] \| string | yes | - | Which indexes to build: `fts`, `embedding`, or `"hybrid"` (interpreted as both). |

FTS section (retrievers.<name>.fts):

| Parameter | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| index_phrase | jinja string | required if `fts` in index_types | - | How to index each dataset row. Context: `row`. |
| query_phrase | jinja string | recommended for FTS/hybrid queries | - | How to construct the FTS query. Context: op context (see below). |

Embedding section (retrievers.<name>.embedding):

| Parameter | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| model | string | required if `embedding` in index_types | - | Embedding model used for both index vectors and query vectors. |
| index_phrase | jinja string | no | falls back to `fts.index_phrase` if present | How to index each dataset row for embedding. Context: `row`. |
| query_phrase | jinja string | recommended for embedding/hybrid queries | - | How to construct the text to embed at query time. Context: op context. |

Query section (retrievers.<name>.query):

| Parameter | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| mode | enum | no | auto | `fts` \| `embedding` \| `hybrid`. If omitted: `hybrid` when both indexes exist, else whichever index exists. |
| top_k | int | no | 5 | Number of results to return. |

Notes:
- Hybrid search uses LanceDB’s built-in reranking (RRF) by default.
- Jinja contexts:
  - Map / Filter / Extract: `{"input": <current item>}`
  - Reduce: `{"reduce_key": {...}, "inputs": [items]}`
- Jinja for indexing uses `{"input": <dataset row>}`
- Keep query phrases concise; slice long fields, e.g. `{{ input.src[:1000] }}`.
- The injected `retrieval_context` is truncated conservatively (~1000 chars per doc).

## Using a retriever in operations

Attach the retriever to any LLM-powered op with `retriever: <name>`. Include `{{ retrieval_context }}` in your prompt or let DocETL prepend it automatically.

### Operation Parameters

When using a retriever with an operation, the following additional parameters are available:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| retriever | string | - | Name of the retriever to use (must be defined in the `retrievers` section). |
| save_retriever_output | bool | false | If true, saves the retrieved context to `_<operation_name>_retrieved_context` in the output. Useful for debugging and verifying retrieval quality. |

### Map
```yaml
operations:
  - name: tag_visit
    type: map
    retriever: medical_r
    save_retriever_output: true  # Optional: save retrieved context to output
    output:
      schema:
        tag: string
        confidence: float
    prompt: |
      Classify the medical visit. Use the extra context if helpful:
      {{ retrieval_context }}
      Transcript:
      {{ input.src }}
```

When `save_retriever_output: true`, each output document will include a `_tag_visit_retrieved_context` field containing the exact context that was retrieved and used for that document.

### Extract
```yaml
  - name: extract_side_effects
    type: extract
    retriever: medical_r
    document_keys: ["src"]
    prompt: "Extract side effects mentioned in the text."
```

### Filter
```yaml
  - name: filter_relevant
    type: filter
    retriever: medical_r
    prompt: "Is this transcript relevant to medication counseling? Return is_relevant: boolean."
    output:
      schema:
        is_relevant: bool
        _short_explanation: string
```

### Reduce
When using reduce, the retrieval context is computed per group. The Jinja context provides both `reduce_key` and `inputs`.
```yaml
  - name: summarize_by_medication
    type: reduce
    retriever: medical_r
    reduce_key: "medication"
    output:
      schema:
        summary: string
    prompt: |
      Summarize key points for medication '{{ reduce_key.medication }}'.
      Use the extra context if helpful:
      {{ retrieval_context }}
      Inputs:
      {{ inputs }}
```

## Jinja template contexts
- Map / Filter / Extract: `{"input": current_item}`
- Reduce: `{"reduce_key": {...}, "inputs": [items]}`

## Token budget and truncation
- DocETL uses a conservative default to limit the size of `retrieval_context` by truncating each retrieved text to ~1000 characters.

## Troubleshooting
- No results: the retriever injects “No extra context available.” and continues.
- Index issues: set `build_index: always` to rebuild; ensure `index_dir` exists and is writable.
- Embeddings: DocETL uses its embedding router and caches results where possible.


