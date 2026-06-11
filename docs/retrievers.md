## Retrievers (LanceDB OSS)

Retrievers augment LLM operations with context retrieved from a LanceDB index built over a DocETL dataset. Define retrievers once at the top level, attach one to any LLM-powered operation with `retriever: <name>`, and DocETL runs full-text, vector, or hybrid search at runtime and injects the results into your prompt as `{{ retrieval_context }}`.

- Always OSS LanceDB (local `index_dir`). Hybrid search uses RRF reranking; see the [LanceDB docs](https://lancedb.com/docs/search/hybrid-search/).
- A retriever references a dataset from the pipeline config, or the output of a previous pipeline step.
- Operations do not override retriever settings.
- If your prompt does not use `{{ retrieval_context }}`, DocETL prepends a short "extra context" section automatically.

All fields are documented in the [configuration reference](#configuration-reference). This page covers both the [YAML configuration](#configuration) and the [Python Frame API](#python-api).

## Python API

Create a `docetl.Retriever` and pass it to any LLM operation (`.map()`, `.filter()`, `.reduce()`, `.extract()`) via `retriever=`. The constructor takes the same fields as the YAML config; see the [configuration reference](#configuration-reference).

The `dataset` field names what gets indexed. In the Python API that can be:

- **An auxiliary dataset** registered with `.with_dataset(name, data)` — a file path or in-memory list of dicts, equivalent to a separate `datasets` entry in YAML.
- **The frame's own input** — the reader's dataset name (the file's basename for `read_json`/`read_csv`/`read_parquet`, or the `name=` given to `from_list`, default `"data"`).
- **A previous step's output** — step names are `step_<operation_name>`, e.g. `step_extract_facts`.

```python
import docetl

docetl.default_model = "gpt-4o-mini"

retriever = docetl.Retriever(
    dataset="kb",                       # the auxiliary dataset registered below
    index_dir="./lance_index",
    index_types=["fts", "embedding"],
    fts={
        "index_phrase": "{{ input.text }}",
        "query_phrase": "{{ input.question }}",
    },
    embedding={
        "model": "openai/text-embedding-3-small",
        "index_phrase": "{{ input.text }}",
        "query_phrase": "{{ input.question }}",
    },
    query={"mode": "hybrid", "top_k": 5},
)

results = (
    docetl.read_json("questions.json")
    .with_dataset("kb", "knowledge_base.json")   # what the retriever indexes
    .map(
        prompt="Answer: {{ input.question }}\nContext: {{ retrieval_context }}",
        output={"schema": {"answer": "str"}},
        retriever=retriever,
    )
    .collect()
)
```

To index an intermediate result instead, point `dataset` at the producing step:

```python
facts = (
    docetl.read_json("articles.json")
    .map("extract_facts", prompt="...", output={"schema": {"facts": "list[str]"}})
    .unnest("explode", unnest_key="facts")
)

facts_index = docetl.Retriever(
    dataset="step_explode",             # output of the unnest step
    index_dir="./facts_index",
    index_types=["fts"],
    fts={"index_phrase": "{{ input.facts }}", "query_phrase": "{{ input.facts }}"},
)

conflicts = facts.map(
    "find_conflicts",
    prompt="Does this fact conflict with any of these?\nFact: {{ input.facts }}\n{{ retrieval_context }}",
    output={"schema": {"has_conflict": "bool"}},
    retriever=facts_index,
).collect()
```

As in YAML, pass `save_retriever_output=True` on the operation to keep the retrieved context in the output (under `_<operation_name>_retrieved_context`) for debugging.

## Configuration

Add a top-level `retrievers` section. Each retriever names a dataset (or pipeline step) to index, where to store the index, which index types to build, and how to query. See the [configuration reference](#configuration-reference) for all fields.

### Basic example

=== "YAML"

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

=== "Python"

    ```python
    import docetl

    docetl.default_model = "gpt-4o-mini"

    medical_r = docetl.Retriever(
        dataset="raw",  # the frame's own input: basename of raw.json
        index_dir="workloads/medical/lance_index",
        index_types=["fts", "embedding"],
        build_index="if_missing",  # if_missing | always | never
        fts={
            "index_phrase": "{{ input.src }}",
            "query_phrase": "{{ input.src[:1000] }}",
        },
        embedding={
            "model": "openai/text-embedding-3-small",
            "index_phrase": "{{ input.src }}",
            "query_phrase": "{{ input.src[:1000] }}",
        },
        query={"mode": "hybrid", "top_k": 8},
    )

    pipeline = docetl.read_json("workloads/medical/raw.json")
    # ... attach medical_r to operations via retriever=medical_r
    ```

## Multi-step pipelines with retrieval

A retriever can index the output of a previous pipeline step: extract structured data in step 1, index it, then retrieve over it in step 2.

=== "YAML"

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

=== "Python"

    ```python
    import docetl

    docetl.default_model = "gpt-4o-mini"
    docetl.intermediate_dir = "workloads/wiki/intermediates"

    # Retriever indexes output of the unnest step (step names are step_<op_name>)
    facts_index = docetl.Retriever(
        dataset="step_unnest_facts",  # References output of a pipeline step!
        index_dir="workloads/wiki/facts_lance_index",
        build_index="if_missing",
        index_types=["fts", "embedding"],
        fts={
            "index_phrase": "{{ input.fact }} from {{ input.title }}",
            "query_phrase": "{{ input.fact }}",
        },
        embedding={
            "model": "openai/text-embedding-3-small",
            "index_phrase": "{{ input.fact }}",
            "query_phrase": "{{ input.fact }}",
        },
        query={"mode": "hybrid", "top_k": 5},
    )

    pipeline = docetl.read_json("workloads/wiki/articles.json")

    # Step 1: Extract and unnest facts
    pipeline = pipeline.map(
        "extract_facts",
        prompt="""Extract factual claims from this article.
    Article: {{ input.title }}
    Text: {{ input.text }}""",
        output={"schema": {"facts": "list[string]"}},
    )
    pipeline = pipeline.unnest("unnest_facts", unnest_key="facts")

    # Step 2: Use retrieval to find conflicts
    pipeline = pipeline.map(
        "find_conflicts",
        retriever=facts_index,  # Uses the retriever
        prompt="""Check if this fact conflicts with similar facts from other articles.

    Current fact: {{ input.facts }} (from {{ input.title }})

    Similar facts from other articles:
    {{ retrieval_context }}

    Return true only if there's a genuine contradiction.""",
        output={"schema": {"has_conflict": "boolean"}},
    )

    pipeline.write_json("workloads/wiki/conflicts.json")
    ```

## Configuration reference

All retriever fields, for both YAML and the `docetl.Retriever` constructor. For a complete example, see [Configuration](#configuration).

### Required fields

| Field | Description |
| --- | --- |
| `type` | Must be `lancedb` |
| `dataset` | Name of a dataset or pipeline step to index |
| `index_dir` | Path where LanceDB stores the index (created if missing) |
| `index_types` | List of index types: `["fts"]`, `["embedding"]`, or `["fts", "embedding"]` |

### Optional fields

| Field | Default | Description |
| --- | --- | --- |
| `build_index` | `if_missing` | When to build: `if_missing`, `always`, or `never` |
| `query.mode` | auto | `fts`, `embedding`, or `hybrid`. Auto-selects based on what indexes exist |
| `query.top_k` | 5 | Number of results to return |

### The `fts` section

Required if `"fts"` is in `index_types`.

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

=== "YAML"

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

=== "Python"

    ```python
    import docetl

    drug_lookup = docetl.Retriever(
        dataset="drugs",                      # index the drugs dataset
        index_dir="./drug_index",
        index_types=["fts"],
        fts={
            "index_phrase": "{{ input.name }}: {{ input.uses }}",  # index: "Aspirin: pain, fever"
            "query_phrase": "{{ input.symptoms }}",                # search with patient symptoms
        },
    )

    pipeline = docetl.read_json("notes.json")  # [{"symptoms": "headache and fever"}, ...]
    pipeline = pipeline.with_dataset(
        "drugs", "drugs.json"  # [{"name": "Aspirin", "uses": "pain, fever"}, ...]
    )
    pipeline = pipeline.map(
        "find_treatment",
        retriever=drug_lookup,                 # attach the retriever
        prompt="""Patient symptoms: {{ input.symptoms }}

    Relevant drugs from knowledge base:
    {{ retrieval_context }}

    Recommend a treatment.""",
        output={"schema": {"recommendation": "string"}},
    )
    ```

### The `embedding` section

Required if `"embedding"` is in `index_types`.

| Field | Required | Description |
| --- | --- | --- |
| `model` | yes | Embedding model, e.g. `openai/text-embedding-3-small` |
| `index_phrase` | no | Jinja template for text to embed. Falls back to `fts.index_phrase` |
| `query_phrase` | yes | Jinja template for query text to embed |

**Jinja variables:** Same as FTS section. For an embedding-only index, set `index_types: ["embedding"]` and omit the `fts` section.

## Using a retriever in operations

Operation-level parameters:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| retriever | string | - | Name of the retriever to use (must match a key in `retrievers`). |
| save_retriever_output | bool | false | If true, saves retrieved context to `_<operation_name>_retrieved_context` in output. |

Map examples appear above ([Python API](#python-api), [Multi-step pipelines](#multi-step-pipelines-with-retrieval)); filter and extract work the same way.

### Reduce example

When using reduce, the retrieval context is computed per group.

=== "YAML"

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

=== "Python"

    ```python
    pipeline = pipeline.reduce(
        "summarize_by_medication",
        retriever=medical_r,
        reduce_key="medication",
        prompt="""Summarize key points for medication '{{ reduce_key.medication }}'.
    Related context: {{ retrieval_context }}

    Inputs:
    {% for item in inputs %}
    - {{ item.src }}
    {% endfor %}""",
        output={"schema": {"summary": "string"}},
    )
    ```

## Troubleshooting

- **No results**: the retriever injects "No extra context available." and continues.
- **Index issues**: set `build_index: always` to rebuild; ensure `index_dir` exists and is writable.
- **Token limits**: `retrieval_context` is truncated to ~1000 chars per retrieved doc.
