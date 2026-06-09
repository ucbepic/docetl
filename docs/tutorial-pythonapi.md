# Tutorial: Analyzing Medical Transcripts with the Python API

This tutorial walks through using DocETL's Python API to analyze medical transcripts and extract medication information. We'll build a pipeline that identifies medications, resolves similar names, and generates summaries of side effects and therapeutic uses.

## Prerequisites and Setup

For installation instructions, API key setup, and data preparation, please refer to the [YAML-based tutorial](tutorial.md#installation). The prerequisite steps are identical for both approaches.

!!! note "Common Setup"
    Both this Python API tutorial and the YAML-based tutorial share the same:
    
    - Installation requirements
    - API key configuration
    - Data format and preparation steps
    
    Once you've completed those steps from the [main tutorial](tutorial.md), you can continue here.

## Building the Pipeline

```python
import docetl

docetl.default_model = "gpt-4o-mini"
docetl.intermediate_dir = "intermediate_results"

pipeline = (
    docetl.read_json("medical_transcripts.json")

    # 1. Extract medications from each transcript
    .map(
        name="extract_medications",
        prompt="""
        Analyze the following transcript of a conversation between a doctor and a patient:
        {{ input.src }}
        Extract and list all medications mentioned in the transcript.
        If no medications are mentioned, return an empty list.
        """,
        output={"schema": {"medication": "list[str]"}},
    )

    # 2. Flatten so each medication is its own row
    .unnest(unnest_key="medication")

    # 3. Resolve similar medication names
    .resolve(
        name="resolve_medications",
        comparison_prompt="""
        Compare the following two medication entries:
        Entry 1: {{ input1.medication }}
        Entry 2: {{ input2.medication }}
        Are these medications likely to be the same or closely related?
        """,
        resolution_prompt="""
        Given the following matched medication entries:
        {% for entry in inputs %}
        Entry {{ loop.index }}: {{ entry.medication }}
        {% endfor %}
        Determine the best resolved medication name for this group.
        """,
        output={"schema": {"medication": "str"}},
        blocking_keys=["medication"],
        blocking_threshold=0.6162,
        embedding_model="text-embedding-3-small",
    )

    # 4. Summarize side effects and uses per medication
    .reduce(
        name="summarize_prescriptions",
        reduce_key="medication",
        prompt="""
        Here are some transcripts of conversations between a doctor and a patient:

        {% for value in inputs %}
        Transcript {{ loop.index }}:
        {{ value.src }}
        {% endfor %}

        For the medication {{ reduce_key }}, provide:

        1. Side Effects: Summarize all mentioned side effects.
        2. Therapeutic Uses: Explain the conditions for which it was prescribed.

        Base your summary solely on the provided transcripts.
        Include relevant quotes.
        """,
        output={"schema": {"side_effects": "str", "uses": "str"}},
    )
)
```

## Running the Pipeline

```python
# Preview on a small sample first
pipeline.show()

# Run the full pipeline
df = pipeline.collect()
print(f"Total cost: ${pipeline.total_cost:.2f}")
print(df)
```

Or write directly to a file:

```python
cost = pipeline.write_json("medication_summaries.json")
print(f"Total cost: ${cost:.2f}")
```

!!! info "Pipeline Performance"

    When running this pipeline on a sample dataset with `gpt-4o-mini`:

    - Total cost: ~$0.10
    - Total execution time: ~49 seconds

## Exploring the Data

You can inspect the data at any point in the chain:

```python
# How many transcripts?
docetl.read_json("medical_transcripts.json").count()

# Preview the raw data
docetl.read_json("medical_transcripts.json").show()

# Check the output schema
pipeline.schema()
# {'side_effects': 'str', 'uses': 'str'}
```

## Exporting to YAML

If you want to run the same pipeline from the command line:

```python
print(pipeline.to_yaml())

# Or write directly to a file
pipeline.to_yaml("pipeline.yaml")
```

Then run with:

```bash
docetl run pipeline.yaml
```

## Optimizing the Pipeline

Use MOAR to search for better model/prompt configurations:

```python
@docetl.register_eval
def evaluate(results):
    # Score based on how many medications have both side effects and uses
    complete = sum(
        1 for r in results
        if r.get("side_effects", "").strip() and r.get("uses", "").strip()
    )
    return {"completeness": complete}

optimized = pipeline.optimize(
    eval_fn=evaluate,
    metric_key="completeness",
)

df = optimized.collect()
print(f"Optimized cost: ${optimized.total_cost:.2f}")
```

## Using Code Operations

Insert deterministic Python transformations without LLM calls:

```python
pipeline = (
    docetl.read_json("medical_transcripts.json")
    .map(
        prompt="Extract medications from: {{ input.src }}",
        output={"schema": {"medication": "list[str]"}},
    )
    .unnest(unnest_key="medication")
    .code_map(
        code="def transform(doc): return {'medication': doc['medication'].lower().strip()}"
    )
    .resolve(
        comparison_prompt="Are these the same? {{ input1.medication }} vs {{ input2.medication }}",
        output={"schema": {"medication": "str"}},
    )
)
```

## Further Questions

??? question "What if I want to focus on a specific type of medication?"

    Modify the prompts in the `map` and `reduce` operations. For example, update the extraction prompt to only list medications related to cardiovascular diseases.

??? question "How can I improve medication name resolution?"

    Adjust `blocking_threshold` — lower values match more aggressively, higher values require closer matches. You can also customize the comparison and resolution prompts.

??? question "Can I process other types of medical documents?"

    Yes — adapt the input data format and adjust the prompts. This pipeline pattern (extract, unnest, resolve, reduce) works for discharge summaries, clinical notes, research papers, etc.
