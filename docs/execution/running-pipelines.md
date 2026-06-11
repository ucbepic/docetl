# Running Pipelines from the CLI

Run a YAML pipeline with:

```bash
docetl run pipeline.yaml
```

DocETL caches LLM results by default, so re-running a pipeline retrieves
unchanged results from the cache instead of recomputing them. Clear it with
`docetl clear-cache`.

To save the output of each operation for inspection and resumability, set
`intermediate_dir` (see the [tutorial's note on relative paths](../tutorial.md#running-the-pipeline)):

=== "YAML"

    ```yaml
    pipeline:
      output:
        type: file
        path: medication_summaries.json
        intermediate_dir: intermediate_results
    ```

=== "Python"

    ```python
    docetl.intermediate_dir = "intermediate_results"
    pipeline.write_json("medication_summaries.json")
    ```

## The `run` command

::: docetl.cli.run
  handler: python
  options:
    members: - run
  show_root_full_path: true
  show_root_toc_entry: true
  show_root_heading: true
  show_source: false
  show_name: true
