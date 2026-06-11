# Resolve Operation

The Resolve operation identifies and canonicalizes duplicate entities in your data. LLM-generated fields and multi-source data often refer to the same entity inconsistently (e.g., "Mrs. Smith" vs. "Jane Smith"); resolving the field standardizes it before further analysis.

## Example: Standardizing Patient Names

=== "YAML"

    ```yaml
    - name: standardize_patient_names
      type: resolve
      optimize: true
      comparison_prompt: |
        Compare the following two patient name entries:

        Patient 1: {{ input1.patient_name }}
        Date of Birth 1: {{ input1.date_of_birth }}

        Patient 2: {{ input2.patient_name }}
        Date of Birth 2: {{ input2.date_of_birth }}

        Are these entries likely referring to the same patient? Consider name similarity and date of birth. Respond with "True" if they are likely the same patient, or "False" if they are likely different patients.
      resolution_prompt: |
        Standardize the following patient name entries into a single, consistent format:

        {% for entry in inputs %}
        Patient Name {{ loop.index }}: {{ entry.patient_name }}
        {% endfor %}

        Provide a single, standardized patient name that represents all the matched entries. Use the format "LastName, FirstName MiddleInitial" if available.
      output:
        schema:
          patient_name: string
    ```

=== "Python"

    ```python
    import docetl

    docetl.default_model = "gpt-4o-mini"

    frame = docetl.read_json("patients.json")
    frame = frame.resolve(
        optimize=True,
        comparison_prompt="""Compare the following two patient name entries:

    Patient 1: {{ input1.patient_name }}
    Date of Birth 1: {{ input1.date_of_birth }}

    Patient 2: {{ input2.patient_name }}
    Date of Birth 2: {{ input2.date_of_birth }}

    Are these entries likely referring to the same patient? Consider name similarity and date of birth. Respond with "True" if they are likely the same patient, or "False" if they are likely different patients.""",
        resolution_prompt="""Standardize the following patient name entries into a single, consistent format:

    {% for entry in inputs %}
    Patient Name {{ loop.index }}: {{ entry.patient_name }}
    {% endfor %}

    Provide a single, standardized patient name that represents all the matched entries. Use the format "LastName, FirstName MiddleInitial" if available.""",
        output={"schema": {"patient_name": "string"}},
    )
    df = frame.collect()
    ```

- The `comparison_prompt` compares pairs of entries; reference the two documents via `input1` and `input2`.
- For identified duplicates, the `resolution_prompt` generates a standardized value; reference all matched entries via the `inputs` variable.
- Prompts use Jinja2 syntax (e.g., `input1.patient_name`).

!!! info "Automatic Blocking"

    If you don't specify any blocking configuration (`blocking_threshold`, `blocking_conditions`, or `limit_comparisons`), the Resolve operation will automatically compute an optimal embedding-based blocking threshold at runtime. It samples pairs from your data, runs LLM comparisons on the sample, and finds a threshold that achieves 95% recall by default. You can adjust this with the `blocking_target_recall` parameter.

## Blocking

Blocking reduces the number of comparisons by only comparing entries that are likely to be matches. Two types work together:

1. **Code-based blocking**: Apply custom Python expressions to determine if a pair should be compared.
2. **Embedding-based blocking**: Compare embeddings of specified fields and only process pairs above a certain similarity threshold.

### How Blocking Works

The Resolve operation creates a **union** of pairs that pass either blocking method:
- First, pairs that satisfy any of the `blocking_conditions` are selected
- Then, pairs that meet the `blocking_threshold` for embedding similarity are added (if not already included)
- When sampling is needed (via `limit_comparisons`), code-based pairs are prioritized over embedding-based pairs

Example with both blocking methods:

=== "YAML"

    ```yaml
    - name: standardize_patient_names
      type: resolve
      comparison_prompt: |
        # (Same as previous example)
      resolution_prompt: |
        # (Same as previous example)
      output:
        schema:
          patient_name: string
      blocking_keys:
        - last_name
        - date_of_birth
      blocking_threshold: 0.8
      blocking_conditions:
        - "input1['last_name'][:2].lower() == input2['last_name'][:2].lower()"
        - "input1['first_name'][:2].lower() == input2['first_name'][:2].lower()"
        - "input1['date_of_birth'] == input2['date_of_birth']"
        - "input1['ssn'][-4:] == input2['ssn'][-4:]"
    ```

=== "Python"

    ```python
    frame = frame.resolve(
        name="standardize_patient_names",
        comparison_prompt="...",  # (Same as previous example)
        resolution_prompt="...",  # (Same as previous example)
        output={"schema": {"patient_name": "string"}},
        blocking_keys=["last_name", "date_of_birth"],
        blocking_threshold=0.8,
        blocking_conditions=[
            "input1['last_name'][:2].lower() == input2['last_name'][:2].lower()",
            "input1['first_name'][:2].lower() == input2['first_name'][:2].lower()",
            "input1['date_of_birth'] == input2['date_of_birth']",
            "input1['ssn'][-4:] == input2['ssn'][-4:]",
        ],
    )
    ```

In this example, pairs will be considered for comparison if they satisfy **any** of the following:

**Code-based conditions:**
- The `last_name` fields start with the same two characters, OR
- The `first_name` fields start with the same two characters, OR  
- The `date_of_birth` fields match exactly, OR
- The last four digits of the `ssn` fields match

**OR**

**Embedding-based condition:**
- The embedding similarity of their `last_name` and `date_of_birth` fields is above 0.8

## How the Comparison Algorithm Works

After determining eligible pairs, the Resolve operation groups similar items with a Union-Find (Disjoint Set Union) algorithm:

1. **Initialization**: Each item starts in its own cluster.
2. **Pair Generation**: All possible pairs of items are generated for comparison.
3. **Batch Processing**: Pairs are processed in batches (controlled by `compare_batch_size`).
4. **Comparison**: For each batch:
   a. An LLM performs pairwise comparisons to determine if items match.
   b. Matching pairs trigger a `merge_clusters` operation to combine their clusters.
5. **Iteration**: Steps 3-4 repeat until all pairs are compared.
6. **Result Collection**: All non-empty clusters are collected as the final result.

!!! note "Efficiency"

    Batched comparisons let clusters update incrementally as matches are found, with LLM calls running in parallel. Parallelism is capped at the batch size, so set `compare_batch_size` based on your dataset size and rate limits.

## Required Parameters

- `type`: Must be set to "resolve".
- `comparison_prompt`: The prompt template to use for comparing potential matches.
- `resolution_prompt`: The prompt template to use for reducing matched entries.
- `output`: Schema definition for the output from the LLM.

## Optional Parameters

| Parameter                 | Description                                                                       | Default                       |
| ------------------------- | --------------------------------------------------------------------------------- | ----------------------------- |
| `embedding_model`         | The model to use for creating embeddings                                          | Falls back to `default_model` |
| `resolution_model`        | The language model to use for reducing matched entries                            | Falls back to `default_model` |
| `comparison_model`        | The language model to use for comparing potential matches                         | Falls back to `default_model` |
| `blocking_keys`           | List of keys to use for initial blocking                                          | All keys in the input data    |
| `blocking_threshold`      | Embedding similarity threshold for considering entries as potential matches       | Auto-computed if not set      |
| `blocking_target_recall`  | Target recall when auto-computing blocking threshold (0.0 to 1.0)                 | 0.95                          |
| `blocking_conditions`     | List of conditions for initial blocking                                           | []                            |
| `input`                   | Specifies the schema or keys to subselect from each item to pass into the prompts | All keys from input items     |
| `embedding_batch_size`    | The number of entries to send to the embedding model at a time                    | 1000                          |
| `compare_batch_size`      | The number of entity pairs processed in each batch during the comparison phase    | 500                           |
| `limit_comparisons`       | Maximum number of comparisons to perform                                          | None                          |
| `timeout`                 | Timeout for each LLM call in seconds                                              | 120                           |
| `max_retries_per_timeout` | Maximum number of retries per timeout                                             | 2                             |
| `sample`                  | Number of samples to use for the operation                                        | None                          |
| `litellm_completion_kwargs` | Additional parameters to pass to LiteLLM completion calls.                      | {}                            |
| `bypass_cache`            | If true, bypass the cache for this operation.                                     | False                         |
| `cascade`                 | Model cascade config for cost reduction on candidate pair comparisons (see below) | None                          |

### Model Cascade (cost reduction)

A `cascade` block runs a cheap proxy model on all candidate pairs first and only escalates uncertain comparisons to the expensive oracle, with a statistical quality guarantee. Most effective when blocking produces many candidate pairs.

=== "YAML"

    ```yaml
    cascade:
      proxy_model: gpt-4o-mini
      target: 0.9
    ```

=== "Python"

    ```python
    # Pass via the cascade= kwarg on a resolve call
    frame = frame.resolve(
        ...,
        cascade={"proxy_model": "gpt-4o-mini", "target": 0.9},
    )
    ```

| Parameter | Description | Default |
|---|---|---|
| `proxy_model` | The cheap model for the proxy pass (required) | — |
| `guarantee` | `accuracy`, `precision`, or `recall` | `precision` |
| `target` | Target value for the guarantee metric, in `(0, 1)` (required) | — |
| `delta` | Failure probability; guarantee holds w.p. `1 - delta` | `0.05` |
| `label_budget` | Max oracle calls spent learning the threshold | `400` |

See [Model Cascades with BARGAIN](../optimization/cascades.md) for full details,
guarantee explanations, and examples.

## Best Practices

1. **Anticipate Resolve Needs**: If you anticipate needing a Resolve operation and want to control the prompts, create it in your pipeline and let the optimizer find the appropriate blocking rules and thresholds.
2. **Let the Optimizer Help**: The optimizer can detect if you need a Resolve operation (e.g., because there's a downstream reduce operation you're optimizing) and can create one with suitable prompts and blocking rules.
3. **Optimize Batch Size**: `compare_batch_size` caps parallelism, so increase it when comparing many pairs — but very large values can exceed memory or API rate limits.
