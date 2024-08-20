# Motion

Motion is a powerful tool for creating and executing data processing pipelines using LLMs. It allows you to define complex data operations in a YAML configuration file and execute them efficiently.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Configuration File Structure](#configuration-file-structure)
4. [Operation Types](#operation-types)
   - [Map](#map)
   - [Parallel Map](#parallel-map)
   - [Filter](#filter)
   - [Unnest](#unnest)
   - [Equijoin](#equijoin)
   - [Split](#split)
   - [Reduce](#reduce)
   - [Resolve](#resolve)
5. [Schemas](#schemas)
   - [Schema Definition](#schema-definition)
   - [Schema Pass-through](#schema-pass-through)
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
motion run pipeline.yaml
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

### Unnest

The Unnest operation expands an array field in the input data into multiple items.

Required parameters:

- `type`: Must be set to `"unnest"`.
- `unnest_key`: The key of the array field to unnest.

Optional parameters:

- `keep_empty`: Boolean flag. If true, empty arrays being exploded will be kept in the output (with value None). Default is false.
- `expand_fields`: A list of fields to expand from the nested dictionary into the parent dictionary, if unnesting a dictionary.

Example of a list unnest:

```yaml
unnest_operation:
  type: unnest
  unnest_key: people
```

If the input data is a list of strings, the unnest operation will expand each string into its own item in the output list. For example, if the input data is:

```yaml
input_data:
  people:
    - "Alice"
    - "Bob"
```

The output will be:

```yaml
output_data:
  - people:
      - "Alice"
  - people:
      - "Bob"
```

Example of a dictionary unnest:

```yaml
unnest_operation:
  type: unnest
  unnest_key: people
  expand_fields:
    - name
    - age
```

The above example will unnest the `people` field, expanding the `name` and `age` fields from the nested dictionary into the parent dictionary. For example, if the input data is:

```yaml
input_data:
  people:
    - person:
        name: Alice
        age: 30
    - person:
        name: Bob
        age: 25
```

The output will be:

```yaml
output_data:
  - name: Alice
    age: 30
    person:
      name: Alice
      age: 30
  - name: Bob
    age: 25
    person:
      name: Bob
      age: 25
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

- `chunk_group_id_field`: The name of a new field where a randomly generated unique ID for each split record will be stored. All chunks from the same original record will share this generated ID.
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

If you specify a `type` of "summary" for any section in `peripheral_chunks`, you must include a `summary_prompt` field in that section. This `summary_prompt` should be a Jinja2 template with a `{{ chunk_content }}` variable, which will be replaced with the chunk content when summarizing.

Each chunk is summarized independently, and the resulting summaries are concatenated to provide peripheral context. You can optionally specify a `summary_model` for each section with a "summary" type. If not provided, the operation will use the `model` specified for the split operation or fall back to the `default_model`.

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
  summary_prompt: |
    Summarize the following text:
    {{ chunk_content }}
```

In this example:

- The content is split into chunks of 50 tokens each.
- For previous chunks:
  - The first 2 chunks are included in full.
  - Summarized representations of all the middle chunks are concatenated together and included.
  - The 1.5 chunks immediately before the current chunk are included in full.
- For next chunks:
  - The first chunk after the current one is included in full.

Notes:

- All sections in `peripheral_chunks` are optional. If omitted, no context will be included for that section.
- If `count` is omitted for `head` or `tail`, it defaults to 0 (effectively omitting that section).
- The `middle` section doesn't use a `count` parameter as it covers all chunks between `head` and `tail`.
- Fractional values for `count` will include a partial chunk. For example, `1.5` includes the first chunk and half of the next chunk.
- The split key will get replaced with the chunk content, and the operation acts like a flatmap. In other words, for each input item, it will produce multiple output items, one for each chunk. Each output item will contain all the original key-value pairs from the input item, plus the chunk_id and the chunked content replacing the original content of the split key.
- The operation also adds a `_chunk_intermediates` key to each output item, containing the full chunk content, previous chunks, and next chunks. This can be used for debugging or further processing.

### Reduce

The Reduce operation aggregates data based on a key. It supports both batch reduction and incremental folding for large datasets.

Required parameters:

- `type`: Must be set to `"reduce"`.
- `reduce_key`: The key to use for grouping data. This can be a single key (string) or a list of keys.
- `prompt`: The prompt template to use for the reduction operation. This template can access the grouped values using `{{ values }}` (a list of dictionary objects or records) and the reduce key using `{{ reduce_key }}`.
- `output`: Schema definition for the output from the LLM.

Optional parameters:

- `model`: The language model to use, falls back to `default_model` if not specified.
- `input`: Specifies the schema or keys to subselect from each item or value to pass into the prompt. If omitted, all keys from the input items will be used.
- `pass_through`: Boolean flag. If true, keys (not on input) from the first item in the group will be passed through to the output. Default is false.
- `commutative`: Boolean flag. If true, the reduce operation is commutative, meaning the order of operations doesn't matter. This can enable further optimizations. Default is true.
- `fold_prompt`: A prompt template for incremental folding. This enables processing of large groups in smaller batches. The template should access the current reduced values using `{{ output.field_name }}` and the new batch of values using `{{ values }}`.
- `fold_batch_size`: The number of items to process in each fold operation when using incremental folding.
- `merge_prompt`: A prompt template for merging the results of multiple fold operations. This is used when processing large groups in parallel. The template should access the list of intermediate results using `{{ outputs }}`.
- `merge_batch_size`: The number of intermediate results to merge in each merge operation. The optimizers uses a default of 2 if it can find a good merge prompt.
- `value_sampling`: A dictionary specifying the sampling strategy for large groups. This can significantly reduce processing time and costs for very large datasets. The dictionary should contain:
  - `enabled`: Boolean flag to enable or disable value sampling.
  - `method`: The sampling method to use. Options are:
    - `"random"`: Randomly select a subset of values.
    - `"first_n"`: Select the first N values.
    - `"cluster"`: Use K-means clustering to select representative samples.
    - `"sem_sim"`: Use semantic similarity to select the most relevant samples to some query text.
  - `sample_size`: The number of samples to select.
  - `embedding_model`: (Required for "cluster" and "sem_sim" methods) The embedding model to use for generating embeddings.
  - `embedding_keys`: (Required for "cluster" and "sem_sim" methods) The keys from the input data to use for generating embeddings.
  - `query_text`: (Required for "sem_sim" method) The query text to compare against when selecting samples. A jinja template with access to `reduce_key`.

Example of a reduce operation with value sampling:

Example of a basic reduce operation:

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

Example of a reduce operation with incremental folding:

```yaml
reduce_operation:
  type: reduce
  reduce_key: group
  prompt: |
    Analyze the following group of values for the group '{{ reduce_key }}':
    {% for value in values %}
    - {{ value }}
    {% endfor %}

    Based on these values, provide:
    1. The total sum of all numeric values
    2. The average (mean) of all numeric values
  fold_prompt: |
    Current reduced value:
    Total: {{ output.total }}
    Average: {{ output.avg }}

    New values to be folded in:
    {% for value in values %}
    - {{ value }}
    {% endfor %}

    Update the current reduced value by incorporating the new values. Provide:
    1. The updated total sum of all numeric values
    2. The updated average (mean) of all numeric values
  fold_batch_size: 50
  output:
    schema:
      total: number
      avg: number
  model: gpt-4o-mini
```

When `fold_prompt` and `fold_batch_size` are specified, the reduce operation will process the data in batches, using the fold prompt to incrementally update the reduced value. This is particularly useful for large datasets or when working with streaming data.

### Resolve

The Resolve operation identifies and merges duplicate entities in the data. The process works as follows:

1. Initialize each entity as its own cluster.
2. Generate all possible pairs of entities.
3. Apply blocking rules to filter pairs for comparison.
4. Compare filtered pairs in batches, updating clusters in real-time.
5. Process final clusters to generate resolved entities.

Required parameters:

- `type`: Must be set to `"resolve"`.
- `comparison_prompt`: The prompt template to use for comparing potential matches.
- `resolution_prompt`: The prompt template to use for reducing matched entries. The matched entries are accessed via the `matched_entries` variable.
- `output`: Schema definition for the output from the LLM. This should include the resolved key.

Optional parameters:

- `embedding_model`: The model to use for creating embeddings. Only used if blocking threshold is set.
- `resolution_model`: The language model to use for reducing matched entries.
- `comparison_model`: The language model to use for comparing potential matches.
- `blocking_keys`: List of keys to use for initial blocking.
- `blocking_threshold`: Embedding similarity threshold for considering entries as potential matches.
- `blocking_conditions`: List of conditions for initial blocking.
- `input`: Specifies the schema or keys to subselect from each item to pass into the prompts. If omitted, all keys from the input items will be used.
- `embedding_batch_size`: The number of entries to send to the embedding model at a time.
- `compare_batch_size`: The number of entity pairs processed in each batch during the comparison phase. Increasing the batch size may improve overall speed but will increase memory usage. Decreasing it will reduce memory usage but may slightly increase total processing time due to increased overhead.

Example:

```yaml
resolve_operation:
  type: resolve
  comparison_prompt: |
    Compare the following two patient records:

    Patient 1:
    {{ input1 | tojson }}

    Patient 2:
    {{ input2 | tojson }}

    Are these records likely referring to the same patient? Consider name similarity, date of birth, and other identifying information. Respond with "True" if they are likely the same patient, or "False" if they are likely different patients.
  resolution_prompt: |
    Merge the following patient records into a single, consolidated record:

    {% for entry in matched_entries %}
    Patient Record {{ loop.index }}:
    {{ entry | tojson }}

    {% endfor %}

    Provide a single, merged patient record that combines all the information from the matched entries. Resolve any conflicts by choosing the most recent or most complete information.
  output:
    schema:
      record: str
  blocking_keys:
    - record
  blocking_threshold: 0.8
  embedding_model: text-embedding-ada-002
  resolution_model: gpt-4o-mini
  comparison_model: gpt-4o-mini
```

## Schemas

### Schema Definition

Schemas in Motion are defined using a simple key-value structure, where each key represents a field name and the value specifies the data type. The supported data types are:

- `string` (or `str`, `text`, `varchar`): For text data
- `integer` (or `int`): For whole numbers
- `number` (or `float`, `decimal`): For decimal numbers
- `boolean` (or `bool`): For true/false values
- `list`: For arrays or sequences of items

For more complex types like lists of dictionaries, you can use a compact notation:

- `list[{key1: type1, key2: type2, ...}]`: A list of dictionaries with specified key-value pairs

Here's an example of a schema definition that includes various types:

```yaml
schema:
  name: string
  age: integer
  height: number
  is_student: boolean
  hobbies: list[string]
  address: "{street: string, city: string, zip_code: string}"
  grades: "list[{subject: string, score: number}]"
```

This schema definition includes:

- Simple types: string (name), integer (age), number (height), and boolean (is_student)
- A list type for hobbies
- A nested dictionary structure for address, containing street, city, and zip_code
- A complex list type for grades, containing dictionaries with subject and score

When defining schemas, you can use these types to accurately represent your data structure. This helps ensure data consistency and enables proper validation throughout the pipeline.

It's worth noting that the schema definition is flexible and can accommodate various levels of complexity. For instance:

1. Nested structures: As shown in the 'address' field, you can define nested dictionaries to represent complex data types.

2. Arrays of objects: The 'grades' field demonstrates how to define a list of dictionaries, which is useful for representing collections of structured data.

3. Lists must have a subtype specified, e.g. `list[string]` or `list[{subject: string, score: number}]`.

4. When using a dictionary type, you need to use quotes around the type definition---otherwise yaml cannot parse it.

### Schema Pass-through

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
