# Pipelines

Pipelines in docetl are the core structures that define the flow of data processing. They orchestrate the application of operators to datasets, creating a seamless workflow for complex document processing tasks.

## Components of a Pipeline

A pipeline in docetl consists of four main components:

1. **Default Model**: The language model to use for the pipeline.
2. **Datasets**: The input data sources for your pipeline.
3. **Operators**: The processing steps that transform your data.
4. **Pipeline Specification**: The sequence of steps and the output configuration.

### Default Model

You can set the default model for a pipeline in the YAML configuration file. If no model is specified at the operation level, the default model will be used.

```yaml
default_model: gpt-4o-mini
```

### Datasets

Datasets define the input data for your pipeline. They are collections of documents, where each document is an object in a JSON list. Datasets are typically specified in the YAML configuration file, indicating the type and path of the data source. For example:

```yaml
datasets:
  user_logs:
    type: file
    path: "user_logs.json"
```

!!! note

    Currently, docetl only supports JSON files as input datasets. If you're interested in support for other data types or cloud-based datasets, please reach out to us or join our open-source community and contribute! We welcome new ideas and contributions to expand the capabilities of docetl.

### Operators

Operators are the building blocks of your pipeline, defining the transformations and analyses to be performed on your data. They are detailed in the [Operators](../concepts/operators.md) documentation. Operators can include map, reduce, filter, and other types of operations.

### Pipeline Specification

The pipeline specification outlines the sequence of steps to be executed and the final output configuration. It typically includes:

- Steps: The sequence of operations to be applied to the data.
- Output: The configuration for the final output of the pipeline.

For example:

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
```

For a practical example of how these components come together, refer to the [Tutorial](../tutorial.md), which demonstrates a complete pipeline for analyzing user behavior data.
