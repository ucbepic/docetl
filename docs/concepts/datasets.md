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

## Examples

### A JSON file

A list of objects; each object is one row.

```json
// reviews.json
[
  {"id": 1, "product": "headphones", "review": "Battery died after a week."},
  {"id": 2, "product": "keyboard", "review": "Keys feel great, very quiet."}
]
```

=== "YAML"

    ```yaml
    datasets:
      reviews:
        type: file
        path: "reviews.json"
    ```

=== "Python"

    ```python
    reviews = docetl.read_json("reviews.json")
    ```

### A CSV or Parquet file

Each row of the table is one row of the dataset; column names become keys.

```csv
ticket_id,customer,message
101,acme,"Cannot log in since the update"
102,globex,"Invoice total looks wrong"
```

=== "YAML"

    ```yaml
    datasets:
      tickets:
        type: file
        path: "tickets.csv"   # or .parquet
    ```

=== "Python"

    ```python
    tickets = docetl.read_csv("tickets.csv")   # or read_parquet(...)
    ```

### A directory of documents

Every non-hidden file under the directory (recursively) becomes one row:
`text` holds the file's content, with `filename` and `path` alongside. PDF,
Word, PowerPoint, and Excel files are converted to text; other files are read
as UTF-8; binary files with no extractor are skipped with a warning.

```text
contracts/
  acme_msa.pdf
  globex_nda.docx
  notes/renewal_2026.txt
```

=== "YAML"

    ```yaml
    datasets:
      contracts:
        type: file
        path: "contracts"
    ```

=== "Python"

    ```python
    contracts = docetl.read_dir("contracts")
    # one row per file:
    # {
    #     "filename": "acme_msa.pdf",
    #     "path": "contracts/acme_msa.pdf",
    #     "text": "MASTER SERVICE AGREEMENT\nThis Agreement is entered into by...",
    # }
    ```

### An in-memory list (Python only)

```python
docs = docetl.from_list([
    {"speaker": "patient", "utterance": "The headaches started last month."},
    {"speaker": "doctor", "utterance": "Any changes in vision?"},
])
```

Relative paths resolve against the directory you run from, not the location
of the YAML file or Python script.

## Parsing tools (non-standard inputs)

DocETL ships built-in parsing functions for file types beyond the above,
e.g., `whisper_speech_to_text` for audio, and you can register your own.
See [Custom Parsing](../examples/custom-parsing.md) for the available
built-in tools and how to define custom ones.

To use one, point the dataset at a JSON file of paths and attach the
parsing function:

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
- `output_key`: the key the parsed content is stored under, accessible in
  prompts as `{{ input.transcript }}`.
