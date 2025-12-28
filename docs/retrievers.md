## Retrievers (LanceDB OSS)

Retrievers let you augment LLM operations with retrieved context from a LanceDB index built over a DocETL dataset. You define retrievers once at the top-level, then attach them to any LLM-powered operation using `retriever: <name>`. At runtime, DocETL performs full-text, vector, or hybrid search and injects the results into your prompt as `{{ retrieval_context }}`.

LanceDB supports built-in full-text search, vector search, and hybrid with RRF reranking. See the official docs: [LanceDB Hybrid Search docs](https://lancedb.com/docs/search/hybrid-search/).

### Key points

- Always OSS LanceDB (local `index_dir`).
- A retriever references a dataset from the pipeline config, or the output of a previous pipeline step.
- Operations do not override retriever settings. One source of truth = consistency.
- `{{ retrieval_context }}` is available to your prompt; if not used, DocETL prepends a short "extra context" section automatically.

## Configuration

Add a top-level `retrievers` section. Each retriever has:

- `dataset`: dataset name to index (can be a dataset or output of a previous pipeline step)
- `index_dir`: LanceDB path
- `index_types`: which indexes to build: `fts`, `embedding`, or `hybrid` (both)
- `fts.index_phrase`: Jinja template for indexing each row for full-text search
- `fts.query_phrase`: Jinja template for building the FTS query at runtime
- `embedding.model`: embedding model for vector index and queries
- `embedding.index_phrase`: Jinja template for indexing each row for embeddings
- `embedding.query_phrase`: Jinja template for building the embedding query
- `query.mode`: `fts` | `embedding` | `hybrid` (defaults to `hybrid` when both indexes exist)
- `query.top_k`: number of results to retrieve

### Basic example

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
    build_index: if_missing  # if_missing | always | never
    index_types: ["fts", "embedding"]
    fts:
      index_phrase: "{{ input.src }}"
      query_phrase: "{{ input.src[:1000] }}"
    embedding:
      model: openai/text-embedding-3-small
      index_phrase: "{{ input.src }}"
      query_phrase: "{{ input.src[:1000] }}"
    query:
      mode: hybrid
      top_k: 8
```

## Multi-step pipelines with retrieval

Most pipelines have a single step, but you can define multiple steps where **the output of one step becomes the input (and retriever source) for the next**. This is powerful for patterns like:

1. Extract structured data from documents
2. Build a retrieval index on that extracted data
3. Use retrieval to find related items and process them

### Example: Extract facts, then find conflicts

```yaml
datasets:
  articles:
    type: file
    path: workloads/wiki/articles.json

default_model: gpt-4o-mini

# Retriever indexes output of step 1 (extract_facts_step)
retrievers:
  facts_index:
    type: lancedb
    dataset: extract_facts_step  # References output of a pipeline step!
    index_dir: workloads/wiki/facts_lance_index
    build_index: if_missing
    index_types: ["fts", "embedding"]
    fts:
      index_phrase: "{{ input.fact }} from {{ input.title }}"
      query_phrase: "{{ input.fact }}"
    embedding:
      model: openai/text-embedding-3-small
      index_phrase: "{{ input.fact }}"
      query_phrase: "{{ input.fact }}"
    query:
      mode: hybrid
      top_k: 5

operations:
  - name: extract_facts
    type: map
    prompt: |
      Extract factual claims from this article.
      Article: {{ input.title }}
      Text: {{ input.text }}
    output:
      schema:
        facts: list[string]

  - name: unnest_facts
    type: unnest
    unnest_key: facts

  - name: find_conflicts
    type: map
    retriever: facts_index  # Uses the retriever
    prompt: |
      Check if this fact conflicts with similar facts from other articles.

      Current fact: {{ input.facts }} (from {{ input.title }})

      Similar facts from other articles:
      {{ retrieval_context }}

      Return true only if there's a genuine contradiction.
    output:
      schema:
        has_conflict: boolean

pipeline:
  steps:
    # Step 1: Extract and unnest facts
    - name: extract_facts_step
      input: articles
      operations:
        - extract_facts
        - unnest_facts

    # Step 2: Use retrieval to find conflicts
    - name: find_conflicts_step
      input: extract_facts_step  # Input is output of step 1
      operations:
        - find_conflicts

  output:
    type: file
    path: workloads/wiki/conflicts.json
    intermediate_dir: workloads/wiki/intermediates
```

In this example:
- **Step 1** (`extract_facts_step`) extracts facts from articles
- The **retriever** (`facts_index`) indexes the output of step 1
- **Step 2** (`find_conflicts_step`) processes each fact, using retrieval to find similar facts from other articles

## Configuration reference

### Minimal example

Here's the simplest possible retriever config (FTS only):

```yaml
retrievers:
  my_search:                              # name can be anything you want
    type: lancedb
    dataset: my_dataset                   # must match a dataset name or pipeline step
    index_dir: ./my_lance_index
    index_types: ["fts"]
    fts:
      index_phrase: "{{ input.text }}"    # what to index from each row
      query_phrase: "{{ input.query }}"   # what to search for at runtime
```

### Full example with all options

```yaml
retrievers:
  my_search:
    type: lancedb
    dataset: my_dataset
    index_dir: ./my_lance_index
    build_index: if_missing               # optional, default: if_missing
    index_types: ["fts", "embedding"]     # can be ["fts"], ["embedding"], or both
    fts:
      index_phrase: "{{ input.text }}"
      query_phrase: "{{ input.query }}"
    embedding:
      model: openai/text-embedding-3-small
      index_phrase: "{{ input.text }}"    # optional, falls back to fts.index_phrase
      query_phrase: "{{ input.query }}"
    query:                                # optional section
      mode: hybrid                        # optional, auto-selects based on index_types
      top_k: 10                           # optional, default: 5
```

---

### Required fields

| Field | Description |
| --- | --- |
| `type` | Must be `lancedb` |
| `dataset` | Name of a dataset or pipeline step to index |
| `index_dir` | Path where LanceDB stores the index (created if missing) |
| `index_types` | List of index types: `["fts"]`, `["embedding"]`, or `["fts", "embedding"]` |

---

### Optional fields

| Field | Default | Description |
| --- | --- | --- |
| `build_index` | `if_missing` | When to build: `if_missing`, `always`, or `never` |
| `query.mode` | auto | `fts`, `embedding`, or `hybrid`. Auto-selects based on what indexes exist |
| `query.top_k` | 5 | Number of results to return |

---

### The `fts` section

Required if `"fts"` is in `index_types`. Configures full-text search.

| Field | Required | Description |
| --- | --- | --- |
| `index_phrase` | yes | Jinja template: what text to index from each dataset row |
| `query_phrase` | yes | Jinja template: what text to search for at query time |

**Jinja variables available:**

| Template | Variables | When it runs |
| --- | --- | --- |
| `index_phrase` | `input` = the dataset row | Once per row when building the index |
| `query_phrase` | `input` = current item (map/filter/extract) | At query time for each item processed |
| `query_phrase` | `reduce_key`, `inputs` (reduce operations) | At query time for each group |

**Example - Medical knowledge base:**

```yaml
datasets:
  drugs:
    type: file
    path: drugs.json  # [{"name": "Aspirin", "uses": "pain, fever"}, ...]

  patient_notes:
    type: file
    path: notes.json  # [{"symptoms": "headache and fever"}, ...]

retrievers:
  drug_lookup:
    type: lancedb
    dataset: drugs                        # index the drugs dataset
    index_dir: ./drug_index
    index_types: ["fts"]
    fts:
      index_phrase: "{{ input.name }}: {{ input.uses }}"   # index: "Aspirin: pain, fever"
      query_phrase: "{{ input.symptoms }}"                  # search with patient symptoms

operations:
  - name: find_treatment
    type: map
    retriever: drug_lookup                # attach the retriever
    prompt: |
      Patient symptoms: {{ input.symptoms }}

      Relevant drugs from knowledge base:
      {{ retrieval_context }}

      Recommend a treatment.
    output:
      schema:
        recommendation: string
```

When processing `{"symptoms": "headache and fever"}`:

1. `query_phrase` renders to `"headache and fever"`
2. FTS searches the index and finds `"Aspirin: pain, fever"` as a match
3. `{{ retrieval_context }}` in your prompt contains the matched results

---

### The `embedding` section

Required if `"embedding"` is in `index_types`. Configures vector/semantic search.

| Field | Required | Description |
| --- | --- | --- |
| `model` | yes | Embedding model, e.g. `openai/text-embedding-3-small` |
| `index_phrase` | no | Jinja template for text to embed. Falls back to `fts.index_phrase` |
| `query_phrase` | yes | Jinja template for query text to embed |

**Jinja variables:** Same as FTS section.

**Example - Semantic search:**

```yaml
retrievers:
  semantic_docs:
    type: lancedb
    dataset: documentation
    index_dir: ./docs_index
    index_types: ["embedding"]
    embedding:
      model: openai/text-embedding-3-small
      index_phrase: "{{ input.content }}"
      query_phrase: "{{ input.question }}"
```

---

### The `query` section (optional)

Controls search behavior. You can omit this entire section.

| Field | Default | Description |
| --- | --- | --- |
| `mode` | auto | `fts`, `embedding`, or `hybrid`. Auto-selects `hybrid` if both indexes exist |
| `top_k` | 5 | Number of results to retrieve |

**Example - Override defaults:**

```yaml
retrievers:
  my_search:
    # ... other config ...
    query:
      mode: fts      # force FTS even if embedding index exists
      top_k: 20      # return more results
```

---

## Using a retriever in operations

Attach a retriever to any LLM operation (map, filter, reduce, extract) with `retriever: <retriever_name>`. The retrieved results are available as `{{ retrieval_context }}` in your prompt.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| retriever | string | - | Name of the retriever to use (must match a key in `retrievers`). |
| save_retriever_output | bool | false | If true, saves retrieved context to `_<operation_name>_retrieved_context` in output. |

### Map example

```yaml
- name: tag_visit
  type: map
  retriever: medical_r
  save_retriever_output: true
  output:
    schema:
      tag: string
  prompt: |
    Classify this medical visit. Related context:
    {{ retrieval_context }}

    Transcript: {{ input.src }}
```

### Filter example

```yaml
- name: filter_relevant
  type: filter
  retriever: medical_r
  prompt: |
    Is this transcript relevant to medication counseling?
    Context: {{ retrieval_context }}
    Transcript: {{ input.src }}
  output:
    schema:
      is_relevant: boolean
```

### Reduce example

When using reduce, the retrieval context is computed per group.

```yaml
- name: summarize_by_medication
  type: reduce
  retriever: medical_r
  reduce_key: medication
  output:
    schema:
      summary: string
  prompt: |
    Summarize key points for medication '{{ reduce_key.medication }}'.
    Related context: {{ retrieval_context }}

    Inputs:
    {% for item in inputs %}
    - {{ item.src }}
    {% endfor %}
```

## Troubleshooting

- **No results**: the retriever injects "No extra context available." and continues.
- **Index issues**: set `build_index: always` to rebuild; ensure `index_dir` exists and is writable.
- **Token limits**: `retrieval_context` is truncated to ~1000 chars per retrieved doc.
