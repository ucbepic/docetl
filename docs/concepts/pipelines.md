# Pipelines

A pipeline applies a sequence of [operators](../concepts/operators.md) to a
dataset and writes the result.

## Two interfaces

DocETL has two first-class ways to write pipelines:

- **YAML** — declare the pipeline in a config file and run it with
  `docetl run pipeline.yaml`. No code required.
- **Python** — build the same pipeline with chained methods (the
  [Frame API](../api-reference/python.md)) and run it with `.collect()`.

Every example in these docs shows both, in tabs. The two convert into each
other: `frame.to_yaml("pipeline.yaml")` and `docetl.Frame.from_yaml("pipeline.yaml")`.

## Components

The required pieces are:

1. **Datasets**: the input data.
2. **Operators**: the processing steps.
3. **Pipeline specification**: the order of steps and the output location.

Optional settings (default model, system prompts) are covered
[below](#configuration).

## Datasets

The input data: JSON, CSV, or Parquet files, a directory of documents, or an
in-memory list of dicts. See [Datasets](datasets.md).

## Operators

Operators define the transformations applied to your data — map, filter,
reduce, resolve, and others. See the [Operators](../concepts/operators.md)
documentation.

## Pipeline Specification

The pipeline specification lists the steps to execute and the output:

=== "YAML"

    ```yaml
    pipeline:
      steps:
        - name: analyze_user_logs
          input: user_logs
          operations:
            - extract_insights
            - unnest_insights
            - summarize_by_country
      output:
        type: file
        path: "country_summaries.json"
        intermediate_dir: "intermediate_data" # Optional: saves each operation's output
    ```

=== "Python"

    In the Frame API, the pipeline is the chain of operations itself, and the
    terminal write method defines the output:

    ```python
    import docetl

    docetl.intermediate_dir = "intermediate_data"  # Optional: saves each operation's output

    pipeline = docetl.read_json("user_logs.json")
    pipeline = pipeline.map(name="extract_insights", ...)
    pipeline = pipeline.unnest(name="unnest_insights", ...)
    pipeline = pipeline.reduce(name="summarize_by_country", ...)
    pipeline.write_json("country_summaries.json")
    ```

## Running a Pipeline

=== "YAML"

    ```bash
    docetl run pipeline.yaml
    ```

=== "Python"

    ```python
    rows = pipeline.collect()              # result rows as a list of dicts
    df = pipeline.to_pandas()              # or a pandas DataFrame
    pipeline.write_json("output.json")     # or write to a file
    ```

DocETL caches LLM results by default, so re-running a pipeline retrieves
unchanged results from the cache instead of recomputing them. Clear it with
`docetl clear-cache`.

Relative paths — dataset `path`, output `path`, and `intermediate_dir` —
resolve against the directory you run from, not the location of the YAML file
or Python script.

For a complete worked pipeline, see the [Tutorial](../tutorial.md).

The `docetl run` command's options:

::: docetl.cli.run
  handler: python
  options:
    members: - run
  show_root_full_path: false
  show_root_toc_entry: false
  show_root_heading: false
  show_source: false
  show_name: false

## Optimizing a Pipeline

Two optimizers are available, covered in the [Optimization](../optimization/overview.md) section:

- [Model cascades](../optimization/cascades.md): cost optimization of a single
  operator, applied during execution itself.
- [MOAR](../optimization/moar.md): joint accuracy and cost optimization of the
  whole pipeline, run as an offline search.

## Configuration

### Default Model

Operations without an explicit `model` use the pipeline default. `bypass_cache`
skips the LLM cache for the whole pipeline (overridable per operation):

=== "YAML"

    ```yaml
    default_model: gpt-4o-mini
    bypass_cache: true  # optional – defaults to false
    ```

=== "Python"

    ```python
    import docetl

    docetl.default_model = "gpt-4o-mini"
    docetl.bypass_cache = True  # optional – defaults to False
    ```

!!! note "Self-hosted models"
    If you're hosting your own models with an OpenAI-compatible API (Ollama,
    LM Studio, etc.), you can specify the base URLs:

    ```yaml
    default_lm_api_base: https://your-custom-llm-endpoint.com/v1
    default_embedding_api_base: https://your-custom-embedding-endpoint.com/v1
    ```

### System Prompts

An optional description of the dataset and the persona the LLM should adopt,
applied to every operation in the pipeline:

=== "YAML"

    ```yaml
    system_prompt:
      dataset_description: a collection of transcripts of doctor visits
      persona: a medical practitioner analyzing patient symptoms and reactions to medications
    ```

=== "Python"

    ```python
    import docetl

    docetl.system_prompt = {
        "dataset_description": "a collection of transcripts of doctor visits",
        "persona": "a medical practitioner analyzing patient symptoms and reactions to medications",
    }
    ```
