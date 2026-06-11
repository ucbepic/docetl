# Datasets

A dataset is the input to a pipeline. Each item in it is one row: an object
in a JSON list, a row in a CSV or Parquet file, or one file in a directory.

## Defining a dataset

=== "YAML"

    Datasets are declared at the top level of the config and referenced by
    name in the pipeline's steps:

    ```yaml
    datasets:
      user_logs:
        type: file
        path: "user_logs.json"
    ```

=== "Python"

    A reader loads a dataset and returns a `Frame`:

    ```python
    import docetl

    user_logs = docetl.read_json("user_logs.json")   # or read_csv, read_parquet, read_dir
    in_memory = docetl.from_list([{"text": "..."}])  # from a list of dicts
    ```

    A `Frame` is a lazy pipeline over the dataset: chaining operations like
    `.map()` records them without running anything, and each call returns a
    new immutable `Frame`. Execution happens at a terminal action —
    `.collect()` (rows as a list of dicts), `.to_pandas()` (a DataFrame), or
    `.write_json()`.

## Accepted inputs

- **JSON**: a list of objects.
- **CSV / Parquet**: one row per record.
- **A directory**: every non-hidden file under it (recursively) becomes one
  row with `path`, `filename`, and `text` keys. PDF, Word, PowerPoint, and
  Excel files are converted to text; other files are read as UTF-8; binary
  files with no extractor are skipped with a warning.

Relative paths resolve against the directory you run from, not the location
of the YAML file or Python script.

## Parsing tools (non-standard inputs)

To process file types beyond the above (audio, scanned PDFs, ...), point the
dataset at a JSON file of paths and attach a parsing function:

=== "YAML"

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

=== "Python"

    ```python
    import docetl

    audio_transcripts = docetl.read_json(
        "audio_files/audio_paths.json",
        parsing=[
            {
                "input_key": "audio_path",
                "function": "whisper_speech_to_text",
                "output_key": "transcript",
            }
        ],
    )
    ```

- `input_key`: the key holding the path to the file to parse.
- `function`: the parsing function (built-in or custom).
- `output_key`: the key the parsed content is stored under — accessible in
  prompts as `{{ input.transcript }}`.

See [Custom Parsing](../examples/custom-parsing.md) for the available built-in
tools and how to define your own.
