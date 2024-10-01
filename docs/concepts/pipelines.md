# Pipelines

Pipelines in DocETL are the core structures that define the flow of data processing. They orchestrate the application of operators to datasets, creating a seamless workflow for complex document processing tasks.

## Components of a Pipeline

A pipeline in DocETL consists of four main components:

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

Datasets define the input data for your pipeline. They are collections of documents, where each document is an object in a JSON list (or row in a CSV file). Datasets are typically specified in the YAML configuration file, indicating the type and path of the data source. For example:

```yaml
datasets:
  user_logs:
    type: file
    path: "user_logs.json"
```

#### Dynamic Data Loading

DocETL supports dynamic data loading, allowing you to process various file types by specifying a key that points to a path or using a custom parsing function. This feature is particularly useful for handling diverse data sources, such as audio files, PDFs, or any other non-standard format.

To implement dynamic data loading, you can use parsing tools in your dataset configuration. Here's an example:

```yaml
datasets:
  audio_transcripts:
    type: file
    source: local
    path: "audio_files/audio_paths.json"
    parsing_tools:
      - input_key: audio_path
        function: whisper_speech_to_text
        output_key: transcript
```

In this example, the dataset configuration specifies a JSON file (audio_paths.json) that contains paths to audio files. The parsing_tools section defines how to process these files:

- `input_key`: Specifies which key in the JSON contains the path to the audio file. In this example, each object in the dataset should have a "audio_path" key, that represents a path to an audio file or mp3.
- `function`: Names the parsing function to use (in this case, the built-in whisper_speech_to_text function for audio transcription).
- `output_key`: Defines the key where the processed data (transcript) will be stored. You can access this in the pipeline in any prompts with the `{{ input.transcipt }}` syntax.

This approach allows DocETL to dynamically load and process various file types, extending its capabilities beyond standard JSON or CSV inputs. You can use built-in parsing tools or define custom ones to handle specific file formats or data processing needs. See the [Custom Parsing](../examples/custom-parsing.md) documentation for more details.

!!! note

    Currently, DocETL only supports JSON files or CSV files as input datasets. If you're interested in support for other data types or cloud-based datasets, please reach out to us or join our open-source community and contribute! We welcome new ideas and contributions to expand the capabilities of DocETL.

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
