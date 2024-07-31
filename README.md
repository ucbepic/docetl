# Motion

Motion is a powerful tool for creating and executing data processing pipelines using a custom Domain Specific Language (DSL). It allows you to define complex data operations in a YAML configuration file and execute them efficiently.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Configuration File Structure](#configuration-file-structure)
4. [Operation Types](#operation-types)
   - [Map](#map)
   - [Parallel Map](#parallel-map)
   - [Filter](#filter)
   - [Explode](#explode)
   - [Equijoin](#equijoin)
   - [Split](#split)
   - [Reduce](#reduce)
   - [Resolve](#resolve)
5. [Schema Pass-through](#schema-pass-through)
6. [Validation Rules](#validation-rules)
7. [Example Pipeline](#example-pipeline)

## Installation

To install Motion, clone this repository and install the required dependencies:

```bash
git clone https://github.com/shreyashankar/motion-v3.git
cd motion
pip install -r requirements.txt
```

## Usage

To run a pipeline defined in a YAML file, use the `motion` command:

```bash
motion pipeline.yaml
```

This command will execute the pipeline defined in `pipeline.yaml`.

## Configuration File Structure

The configuration file is a YAML document with the following top-level keys:

- `default_model`: The default language model to use for operations.
- `operations`: Definitions of operations used in the pipeline.
- `datasets`: Input data sources for the pipeline.
- `pipeline`: The sequence of steps to execute, including input and output specifications.

## Operation Types

Motion supports various operation types, each designed for specific data transformation tasks. All prompt templates used in these operations are Jinja2 templates, allowing for the use of loops, conditionals, and other Jinja2 features to create dynamic prompts based on input data.

Here's an overview of the supported operation types:

### Map

The Map operation applies a transformation to each item in the input data.

Required parameters:

- `type`: Must be set to `"map"`.
- `prompt`: The prompt template to use for the transformation.
- `output`: Schema definition for the output from the LLM.
- `model` (optional): The language model to use, falls back to `default_model` if not specified.

Example:

```yaml
map_operation:
  type: map
  prompt: "Analyze the sentiment of the following text: '{{ input.text }}'"
  output:
    schema:
      sentiment: string
  model: gpt-4o-mini
```

### Parallel Map

The Parallel Map operation applies multiple transformations to each item in the input data concurrently.

Required parameters:

- `type`: Must be set to `"parallel_map"`.
- `prompts`: A list of prompt configurations, each containing:
  - `name`: A unique name for the prompt.
  - `prompt`: The prompt template to use for the transformation.
  - `output_keys`: List of keys that this prompt will generate.
  - `model` (optional): The language model to use for this specific prompt.
- `output`: Schema definition for the combined output from all prompts.

Example:

```yaml
parallel_map_operation:
  type: parallel_map
  prompts:
    - name: sentiment
      prompt: "Analyze the sentiment of the following text: '{{ input.text }}'. Classify it as either positive, negative, or neutral."
      output_keys:
        - sentiment
      model: gpt-4o-mini
    - name: word_count
      prompt: "Count the number of words in the following text: '{{ input.text }}'. Return the count as an integer."
      output_keys:
        - word_count
      model: gpt-4o-mini
  output:
    schema:
      sentiment: string
      word_count: integer
```

### Filter

The Filter operation selects items from the input data based on a condition.

Required parameters:

- `type`: Must be set to `"filter"`.
- `prompt`: The prompt template to use for the filtering condition.
- `output`: Schema definition for the output from the LLM. It must include only one field, a boolean field. This field can be named anything, but it must be a boolean field.
- `model` (optional): The language model to use, falls back to `default_model` if not specified.

Example:

```yaml
filter_operation:
  type: filter
  prompt: "Determine if the following text is longer than 5 words: '{{ input.text }}'"
  output:
    schema:
      keep: boolean
  model: gpt-4o-mini
```

### Explode

The Explode operation expands an array field in the input data into multiple items.

Required parameters:

- `type`: Must be set to `"explode"`.
- `explode_key`: The key of the array field to explode.

Example:

```yaml
explode_operation:
  type: explode
  explode_key: tags
```

### Equijoin

The Equijoin operation performs a join between two datasets based on a key, using embedding similarity and a language model for comparison.

Required parameters:

- `type`: Must be set to `"equijoin"`.
- `join_key`: Specification of the join keys for left and right datasets. Both left and right must have at least a `name` field, and may optionally include a `limit` field. The `limit` field specifies that for each tuple from the relevant dataset, there are at most `limit` matching tuples from the other dataset.
- `comparison_model`: The language model to use for comparing join candidates.
- `comparison_prompt`: The prompt template to use for comparing join candidates. This should be designed to elicit a yes or no answer.

Optional parameters:

- `embedding_model`: The model to use for creating embeddings. Only used if blocking threshold is set.
- `blocking_threshold`: Embedding similarity threshold for considering entries as potential matches.
- `blocking_conditions`: List of conditions for initial blocking.

Example:

```yaml
join_book_author:
  type: equijoin
  join_key:
    left:
      name: genre
      limit: 3
    right:
      name: primary_genre
      limit: 3
  embedding_model: "text-embedding-3-small"
  comparison_model: "gpt-4o-mini"
  blocking_threshold: 0.5
  blocking_conditions:
    - "len(left['genre']) > 0 and len(right['primary_genre']) > 0"
  comparison_prompt: |
    Compare the following two genre entries:
    Book Genre: {{ left.genre }}
    Author's Primary Genre: {{ right.primary_genre }}

    Are these genres likely to be the same or closely related?
```

In this example:

- The join is performed on the `genre` field from the left dataset and the `primary_genre` field from the right dataset.
- Each side of the join is limited to 3 matches.
- The `text-embedding-3-small` model is used for creating embeddings for initial similarity comparison.
- The `gpt-4o-mini` model is used for the final comparison.
- Entries with an embedding similarity above 0.5 are considered for comparison.
- An initial blocking condition ensures both genre fields are non-empty.
- The comparison prompt is designed to elicit a yes/no response about genre similarity.

Note that the comparison prompt should be designed to elicit a clear yes or no answer. The equijoin operation will interpret the language model's response and determine if it constitutes a match.

### Split

The Split operation divides long text content into smaller chunks and optionally includes contextual information from surrounding chunks.

Required parameters:

- `type`: Must be set to `"split"`.
- `split_key`: The key of the field containing the text to split.
- `chunk_size`: The maximum size of each chunk in tokens.
- `model` (optional): The language model's tokenizer to use; falls back to `default_model` if not specified. Note that we don't actually run a language model here.

Optional parameters:

- `main_chunk_start`: A string to prefix the main chunk content (default: "<MAIN_CHUNK>"). Only used when there are peripheral chunks.
- `main_chunk_end`: A string to suffix the main chunk content (default: "</MAIN_CHUNK>"). Only used when there are peripheral chunks.
- `peripheral_chunks`: A dictionary specifying how to handle chunks before and after the current chunk.
  - `previous`: Configuration for chunks before the current chunk.
  - `next`: Configuration for chunks after the current chunk.

Both `previous` and `next` can contain the following optional sections:

- `head`: Chunks at the beginning of the document.
- `middle`: Chunks between the head and tail.
- `tail`: Chunks closest to the current chunk.

Each section (`head`, `middle`, `tail`) can have the following properties:

- `type`: Either "full" (include entire chunk) or "summary" (include a summary of the chunk). We default to "full" if the section is specified. If the section is not specified, we will not include any chunks/summaries from that section
- `count`: The number of chunks to include (for `head` and `tail` only). Can be a fractional value.

Example:

```yaml
split_operation:
  type: split
  split_key: content
  model: gpt-4o-mini
  chunk_size: 50
  main_chunk_start: "<MAIN_CHUNK>"
  main_chunk_end: "</MAIN_CHUNK>"
  peripheral_chunks:
    previous:
      head:
        type: full
        count: 2
      middle:
        type: summary
      tail:
        type: full
        count: 1.5
    next:
      head:
        type: full
        count: 1
      tail:
        type: summary
        count: 2
```

In this example:

- The content is split into chunks of 50 tokens each.
- For previous chunks:
  - The first 2 chunks are included in full.
  - All middle chunks are summarized.
  - The 1.5 chunks immediately before the current chunk are included in full.
- For next chunks:
  - The first chunk after the current one is included in full.
  - The last 2 chunks are summarized.

Notes:

- All sections in `peripheral_chunks` are optional. If omitted, no context will be included for that section.
- If `count` is omitted for `head` or `tail`, it defaults to 0 (effectively omitting that section).
- The `middle` section doesn't use a `count` parameter as it covers all chunks between `head` and `tail`.
- Fractional values for `count` will include a partial chunk. For example, `1.5` includes the first chunk and half of the next chunk.
- The split key will get replaced with the chunk content, and the operation acts like a flatmap. In other words, for each input item, it will produce multiple output items, one for each chunk. Each output item will contain all the original key-value pairs from the input item, plus the chunk_id and the chunked content replacing the original content of the split key.
- The operation also adds a `_chunk_intermediates` key to each output item, containing the full chunk content, previous chunks, and next chunks. This can be used for debugging or further processing.

### Reduce

The Reduce operation aggregates data based on a key.

Required parameters:

- `type`: Must be set to `"reduce"`.
- `reduce_key`: The key to use for grouping data.
- `prompt`: The prompt template to use for the reduction operation. This template can access the grouped values using `{{ values }}` (a list of dictionary objects or records) and the reduce key using `{{ reduce_key }}`.
- `output`: Schema definition for the output from the LLM.
- `model` (optional): The language model to use, falls back to `default_model` if not specified.
- `input` (optional): Specifies the schema or keys to subselect from each item or value to pass into the prompt. If omitted, all keys from the input items will be used.
- `pass_through` (optional): Boolean flag. If true, keys (not on input) from the first item in the group will be passed through to the output. Default is false.

Example:

```yaml
reduce_operation:
  type: reduce
  reduce_key: group
  input:
    schema:
      age: integer
  prompt: |
    Analyze the following group of values for the group '{{ reduce_key }}':
    {% for value in values %}
    - {{ value }}
    {% endfor %}

    Based on these values, provide:
    1. The total sum of all numeric values
    2. The average (mean) of all numeric values
  output:
    schema:
      total: number
      avg: number
  model: gpt-4o-mini
```

### Resolve

The Resolve operation identifies and merges duplicate entities in the data.

Required parameters:

- `type`: Must be set to `"resolve"`.
- `comparison_model`: The language model to use for comparing potential matches.
- `comparison_prompt`: The prompt template to use for comparing potential matches.
- `resolution_model`: The language model to use for reducing matched entries.
- `resolution_prompt`: The prompt template to use for reducing matched entries.
- `output`: Schema definition for the output from the LLM. This should include the resolved key.

Optional parameters:

- `embedding_model`: The model to use for creating embeddings. Only used if blocking threshold is set.
- `blocking_keys`: List of keys to use for initial blocking.
- `blocking_threshold`: Embedding similarity threshold for considering entries as potential matches.
- `blocking_conditions`: List of conditions for initial blocking.

Example:

```yaml
resolve_operation:
  type: resolve
  output:
    schema:
      genre: str
  embedding_model: "text-embedding-3-small"
  comparison_model: "gpt-4o-mini"
  resolution_model: "gpt-4o-mini"
  blocking_keys:
    - genre
  blocking_threshold: 0.9
  blocking_conditions:
    - "len(input1['genre']) > 0 and len(input2['genre']) > 0"
  comparison_prompt: |
    Compare the following two genre entries:
    Entry 1:
    Genre: {{ input1.genre }}
    Example Book: {{ input1.title }}
    Theme: {{ input1.theme }}

    Entry 2:
    Genre: {{ input2.genre }}
    Example Book: {{ input2.title }}
    Theme: {{ input2.theme }}

    Are these genres likely to be the same or closely related?
  resolution_prompt: |
    Given the following matched genre entries:
    {% for entry in matched_entries %}
    Entry {{ loop.index }}:
    Genre: {{ entry.genre }}
    Example Book: {{ entry.title }}
    Theme: {{ entry.theme }}
    {% endfor %}

    Determine the best resolved genre for this group of entries. The resolved genre should be a standardized, widely recognized genre category that best represents all matched entries.
```

## Schema Pass-through

It's important to note that all schema items pass through the pipeline. The `output` schema in each operation is ONLY for what is extracted from the LLM. All other fields from the input data are automatically passed through to the next step in the pipeline.

## Validation Rules

You can add validation rules to your operations to ensure the output meets certain criteria. Validation rules are specified using Python expressions.

Example:

```yaml
map_operation:
  type: map
  prompt: "Analyze the following text: '{{ input.text }}'. Provide the word count, theme, and genre."
  output:
    schema:
      word_count: integer
      theme: string
      genre: string
  validate:
    - output["word_count"] > 0
    - len(output["theme"]) > 0
    - len(output["genre"]) > 0
```

In this example, the validation rules ensure that:

1. The word count is greater than zero.
2. The theme is not an empty string.
3. The genre is not an empty string.

If any of these validation rules fail, the output will be discarded and not passed to the next step in the pipeline.

## Example Pipeline

Here's an example of a pipeline that performs sentiment analysis and word counting using a parallel map operation, then filters based on word count:

```yaml
default_model: gpt-4o-mini

operations:
  parallel_map_operation:
    type: parallel_map
    prompts:
      - name: sentiment
        prompt: "Analyze the sentiment of the following text: '{{ input.text }}'. Classify it as either positive, negative, or neutral."
        output_keys: ["sentiment"]
      - name: word_count
        prompt: "Count the number of words in the following text: '{{ input.text }}'. Return the count as an integer."
        output_keys: ["word_count"]
    output:
      schema:
        sentiment: string
        word_count: integer
    validate:
      - output["word_count"] > 0
      - output["sentiment"] in ["positive", "negative", "neutral"]

  filter_operation:
    type: filter
    prompt: "Determine if the word count {{ input.word_count }} is greater than 10. Return true if it is, false otherwise."
    output:
      schema:
        keep: boolean

datasets:
  sample_dataset:
    type: file
    path: "data/sample_data.json"

pipeline:
  steps:
    - name: analyze_text
      input: sample_dataset
      operations:
        - parallel_map_operation
    - name: filter_long_texts
      input: analyze_text
      operations:
        - filter_operation

  output:
    type: file
    path: "output/results.json"
```

To run this pipeline, save it as `pipeline.yaml` and execute:

```bash
motion pipeline.yaml
```

This will process the data in `data/sample_data.json`, perform sentiment analysis and word counting in parallel, validate the results, filter out short texts, and save the results in `output/results.json`.
