# Equijoin Operation (Experimental)

The Equijoin operation (experimental) joins two datasets based on LLM-evaluated criteria, allowing matches based on semantic similarity or complex conditions rather than exact equality. It uses many of the same techniques as the [Resolve operation](resolve.md).

## Example: Matching Job Candidates to Job Postings

=== "YAML"

    ```yaml
    - name: match_candidates_to_jobs
      type: equijoin
      comparison_prompt: |
        Compare the following job candidate and job posting:

        Candidate Skills: {{ left.skills }}
        Candidate Experience: {{ left.years_experience }}

        Job Required Skills: {{ right.required_skills }}
        Job Desired Experience: {{ right.desired_experience }}

        Is this candidate a good match for the job? Consider both the overlap in skills and the candidate's experience level. Respond with "True" if it's a good match, or "False" if it's not a suitable match.
    ```

=== "Python"

    ```python
    import docetl

    docetl.default_model = "gpt-4o-mini"

    candidates = docetl.read_json("candidates.json")
    job_postings = docetl.read_json("job_postings.json")
    frame = candidates.equijoin(
        job_postings,
        comparison_prompt="""Compare the following job candidate and job posting:

    Candidate Skills: {{ left.skills }}
    Candidate Experience: {{ left.years_experience }}

    Job Required Skills: {{ right.required_skills }}
    Job Desired Experience: {{ right.desired_experience }}

    Is this candidate a good match for the job? Consider both the overlap in skills and the candidate's experience level. Respond with "True" if it's a good match, or "False" if it's not a suitable match.""",
    )
    df = frame.collect()
    ```

!!! note "Jinja2 Syntax with left and right"

    The `comparison_prompt` uses Jinja2 syntax; reference the left and right documents via `left` and `right` (e.g., `left.skills`).

!!! info "Automatic Blocking"

    If you don't specify any blocking configuration (`blocking_threshold`, `blocking_conditions`, or `limit_comparisons`), the Equijoin operation will automatically compute an optimal embedding-based blocking threshold at runtime. It samples pairs from your data, runs LLM comparisons on the sample, and finds a threshold that achieves 95% recall by default. You can adjust this with the `blocking_target_recall` parameter.

## Blocking

Equijoin supports the same blocking techniques as Resolve; see the [Blocking section in the Resolve documentation](resolve.md#blocking).

### Adding Blocking Rules

Equijoin lets you specify **explicit blocking logic** to skip record pairs that are obviously unrelated *before* any LLM calls are made.

#### `blocking_keys`
Provide one or more **field names** for each side of the join. The selected values are concatenated and **embedded**; the cosine similarity of the left vs. right embeddings is then compared against `blocking_threshold` (defaults to `1.0`). If the similarity meets or exceeds that threshold, the pair moves on to the `comparison_prompt`; otherwise it is skipped.  
If you omit `blocking_keys`, **all key–value pairs of each record are embedded by default**.

=== "YAML"

    ```yaml
    blocking_keys:
      left:
        - medicine
      right:
        - extracted_medications
    ```

=== "Python"

    ```python
    # Pass via the blocking_keys= kwarg on an equijoin call
    blocking_keys={
        "left": ["medicine"],
        "right": ["extracted_medications"],
    }
    ```

#### `blocking_threshold`
Optionally set a numeric `blocking_threshold` \(0 – 1\) representing the minimum cosine similarity (computed with the selected `embedding_model`) that the concatenated blocking keys must achieve to be considered a candidate pair. Anything below the threshold is filtered out without invoking the LLM.

=== "YAML"

    ```yaml
    blocking_threshold: 0.35
    embedding_model: text-embedding-3-small
    ```

=== "Python"

    ```python
    # Pass via kwargs on an equijoin call
    blocking_threshold=0.35,
    embedding_model="text-embedding-3-small",
    ```

A full Equijoin step combining both ideas might look like:

=== "YAML"

    ```yaml
    - name: join_meds_transcripts
      type: equijoin
      blocking_keys:
        left:
          - medicine
        right:
          - extracted_medications
      blocking_threshold: 0.3535
      embedding_model: text-embedding-3-small
      comparison_prompt: |
        Compare the following medication names:

        {{ left.medicine }}

        {{ right.extracted_medications }}

        Determine if these entries refer to the same medication.
    ```

=== "Python"

    ```python
    frame = meds.equijoin(
        transcripts,
        name="join_meds_transcripts",
        blocking_keys={
            "left": ["medicine"],
            "right": ["extracted_medications"],
        },
        blocking_threshold=0.3535,
        embedding_model="text-embedding-3-small",
        comparison_prompt="""Compare the following medication names:

    {{ left.medicine }}

    {{ right.extracted_medications }}

    Determine if these entries refer to the same medication.""",
    )
    ```

#### Auto-generating Rules (Experimental)
`docetl build pipeline.yaml` can call the **Optimizer** to propose `blocking_keys` and an appropriate `blocking_threshold` based on a sample of your data. This feature is experimental; always review the suggested rules to ensure they do not exclude valid matches.

## Parameters

Equijoin shares many parameters with Resolve; see the [Parameters section in the Resolve documentation](resolve.md#required-parameters).

### Equijoin-Specific Parameters

| Parameter                 | Description                                                                       | Default                       |
| ------------------------- | --------------------------------------------------------------------------------- | ----------------------------- |
| `limits`                  | Maximum matches for each left/right item: `{"left": n, "right": m}`               | No limit                      |
| `blocking_keys`           | Keys for embedding blocking: `{"left": [...], "right": [...]}`                    | All keys from each dataset    |
| `blocking_threshold`      | Embedding similarity threshold for considering pairs                              | Auto-computed if not set      |
| `blocking_target_recall`  | Target recall when auto-computing blocking threshold (0.0 to 1.0)                 | 0.95                          |

Key differences from Resolve:

- `resolution_prompt` is not used in Equijoin.
- `blocking_keys` uses a dict with `left` and `right` keys instead of a simple list.

### Model Cascade (cost reduction)

Like resolve, equijoin supports a `cascade` block to run a cheap proxy model on
candidate pairs and only escalate uncertain comparisons to the oracle.
Default guarantee is `precision` (don't over-join). See
[Model Cascades with BARGAIN](../optimization/cascades.md) for full details.

## Incorporating Into a Pipeline

=== "YAML"

    ```yaml
    model: gpt-4o-mini

    datasets:
      candidates:
        type: file
        path: /path/to/candidates.json
      job_postings:
        type: file
        path: /path/to/job_postings.json

    operations:
      - name: match_candidates_to_jobs:
        type: equijoin
        comparison_prompt: |
          Compare the following job candidate and job posting:

          Candidate Skills: {{ left.skills }}
          Candidate Experience: {{ left.years_experience }}

          Job Required Skills: {{ right.required_skills }}
          Job Desired Experience: {{ right.desired_experience }}

          Is this candidate a good match for the job? Consider both the overlap in skills and the candidate's experience level. Respond with "True" if it's a good match, or "False" if it's not a suitable match.

    pipeline:
      steps:
        - name: match_candidates_to_jobs
          operations:
            - match_candidates_to_jobs:
                left: candidates
                right: job_postings

      output:
        type: file
        path: "/path/to/matched_candidates_jobs.json"
    ```

=== "Python"

    ```python
    import docetl

    docetl.default_model = "gpt-4o-mini"

    candidates = docetl.read_json("/path/to/candidates.json")
    job_postings = docetl.read_json("/path/to/job_postings.json")
    frame = candidates.equijoin(
        job_postings,
        name="match_candidates_to_jobs",
        comparison_prompt="""Compare the following job candidate and job posting:

    Candidate Skills: {{ left.skills }}
    Candidate Experience: {{ left.years_experience }}

    Job Required Skills: {{ right.required_skills }}
    Job Desired Experience: {{ right.desired_experience }}

    Is this candidate a good match for the job? Consider both the overlap in skills and the candidate's experience level. Respond with "True" if it's a good match, or "False" if it's not a suitable match.""",
    )
    frame.write_json("/path/to/matched_candidates_jobs.json")
    ```

## Best Practices

1. **Use the Optimizer**: `docetl build pipeline.yaml` can generate efficient blocking rules for your Equijoin operation.
2. **Balance Precision and Recall**: When optimizing, consider the trade-off between catching all potential matches and reducing unnecessary comparisons.
3. **Mind Resource Constraints**: Use `limit_comparisons` to cap the total number of comparisons for large datasets.

For best practices that apply to both Resolve and Equijoin, see the [Best Practices section in the Resolve documentation](resolve.md#best-practices).

<!-- ## Performance Considerations

Equijoin operations can be computationally intensive, especially for large datasets. It uses multiprocessing for initial blocking and a ThreadPoolExecutor for LLM-based comparisons to improve performance. However, be mindful of the following:

- The number of comparisons grows with the product of the sizes of your datasets.
- Each comparison involves an LLM call, which can be time-consuming and costly.
- Using optimizer-generated blocking rules can significantly reduce the number of required comparisons.

Always monitor the operation's progress and consider using sampling or more stringent blocking rules if the number of comparisons becomes too large.

The Equijoin operation is particularly useful for scenarios where traditional exact-match joins are insufficient, such as matching job candidates to positions, aligning customer inquiries with product offerings, or connecting research papers with relevant funding opportunities. -->
