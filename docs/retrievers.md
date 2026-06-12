# Retrievers

Sometimes an operation's prompt needs information that isn't in the row being
processed — e.g., answering each question in a dataset requires the relevant
entry from a knowledge base. Putting the whole knowledge base in every prompt
is expensive and often exceeds the context window.

A **retriever** indexes a dataset once and, for each item an operation
processes, searches the index and injects the top matches into the prompt as
`{{ retrieval_context }}`. You define a retriever at the top level of the
pipeline and attach it to any LLM-powered operation.

- The index is built with the [LanceDB](https://lancedb.com) library and
  stored in a local directory (`index_dir`) — there is no server or external
  service. It supports full-text search, vector search, or both combined
  ([hybrid search](https://lancedb.com/docs/search/hybrid-search/)).
- The indexed dataset can be any dataset in the pipeline config or the output
  of a previous pipeline step.
- Retriever settings live on the retriever, not on operations.
- If your prompt does not use `{{ retrieval_context }}`, DocETL appends the
  retrieved matches to the prompt automatically.

All fields are documented in the [configuration reference](#configuration-reference).

## Example

Answer questions using a knowledge base. The retriever indexes the knowledge
base; for each question, the top matches are injected into the prompt.

=== "YAML"

    ```yaml
    datasets:
      questions:
        type: file
        path: questions.json
      kb:
        type: file
        path: knowledge_base.json

    default_model: gpt-4o-mini

    retrievers:
      kb_search:
        type: lancedb
        dataset: kb                        # what to index
        index_dir: ./lance_index
        index_types: ["fts", "embedding"]
        fts:
          index_phrase: "{{ input.text }}"
          query_phrase: "{{ input.question }}"
        embedding:
          model: openai/text-embedding-3-small
          index_phrase: "{{ input.text }}"
          query_phrase: "{{ input.question }}"
        query:
          mode: hybrid
          top_k: 5

    operations:
      - name: answer
        type: map
        retriever: kb_search               # attach to the operation
        prompt: |
          Answer: {{ input.question }}
          Context: {{ retrieval_context }}
        output:
          schema:
            answer: str

    pipeline:
      steps:
        - name: answer_step
          input: questions
          operations:
            - answer
      output:
        type: file
        path: answers.json
    ```

=== "Python"

    ```python
    import docetl

    docetl.default_model = "gpt-4o-mini"

    kb_search = docetl.Retriever(
        data="knowledge_base.json",         # what to index (a path or list of dicts)
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
        .map(
            prompt="Answer: {{ input.question }}\nContext: {{ retrieval_context }}",
            output={"schema": {"answer": "str"}},
            retriever=kb_search,             # attach to the operation
        )
        .collect()
    )
    ```

The operation-level parameters:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `retriever` | string / `Retriever` | - | The retriever to use. Available on `map`, `filter`, `reduce`, and `extract`. |
| `save_retriever_output` | bool | false | Save the retrieved context to `_<operation_name>_retrieved_context` in the output. |

For `reduce`, the context is retrieved once per group (the `query_phrase` sees
`reduce_key` and `inputs` instead of `input` — see the
[Jinja variables table](#the-fts-section)).

In the Python API, a `Retriever` takes its data one of two ways

- `data=` — a file path or list of dicts to index, as above;
- `dataset=` — the name of an existing pipeline dataset (the frame's own
  input, named by the file's basename or `from_list`'s `name=`) or a previous
  step's output, named `step_<operation_name>`.

## Indexing a previous step's output

Extract structured data in step 1, index it, retrieve over it in step 2.

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

All retriever fields, for both YAML and the `docetl.Retriever` constructor. For a complete example, see [Example](#example).

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

Required if `"fts"` is in `index_types`. It has two Jinja templates. Both are
required, and `{{ input }}` refers to a different row in each:

- `index_phrase` produces the text stored in the index. It runs once per row
  of the **indexed dataset** when the index is built, and `input` is that row.
- `query_phrase` produces the search query. It runs once per item the
  **operation** processes, and `input` is that item. (In a `reduce`
  operation it runs once per group, with `reduce_key` and `inputs` instead
  of `input`.)

In the example below, `index_phrase` reads drug rows and `query_phrase`
reads patient rows.

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
        data="drugs.json",  # [{"name": "Aspirin", "uses": "pain, fever"}, ...]
        index_dir="./drug_index",
        index_types=["fts"],
        fts={
            "index_phrase": "{{ input.name }}: {{ input.uses }}",  # index: "Aspirin: pain, fever"
            "query_phrase": "{{ input.symptoms }}",                # search with patient symptoms
        },
    )

    pipeline = docetl.read_json("notes.json")  # [{"symptoms": "headache and fever"}, ...]
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

## Troubleshooting

- **No results**: the retriever injects "No extra context available." and continues.
- **Index issues**: set `build_index: always` to rebuild; ensure `index_dir` exists and is writable.
- **Token limits**: `retrieval_context` is truncated to ~1000 chars per retrieved doc.
