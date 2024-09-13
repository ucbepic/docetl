# docetl

docetl is a powerful tool for creating and executing data processing pipelines using LLMs. It allows you to define complex data operations in a YAML configuration file and execute them efficiently.

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
   - [Gather](#gather)
   - [Reduce](#reduce)
   - [Resolve](#resolve)
5. [Schemas](#schemas)
   - [Schema Definition](#schema-definition)
   - [Schema Pass-through](#schema-pass-through)
6. [Tool Use](#tool-use)
7. [Validation Rules](#validation-rules)
8. [Example Pipeline](#example-pipeline)

## Installation

To install docetl, clone this repository and install the required dependencies:

```bash
git clone https://github.com/shreyashankar/docetl.git
cd motion-v3
pip install poetry
make install
```

Then set up a .env file in your repository with the following:

```bash
OPENAI_API_KEY=your_openai_api_key
```

or you can set the OPENAI_API_KEY environment variable in your environment.

Then run the basic test suite to ensure everything is working:

```bash
make tests-basic
```

## Usage

To run a pipeline defined in a YAML file, use the `docetl` command:

```bash
docetl run pipeline.yaml
```

This command will execute the pipeline defined in `pipeline.yaml`.

## Configuration File Structure

The configuration file is a YAML document with the following top-level keys:

- `default_model`: The default language model to use for operations.
- `operations`: Definitions of operations used in the pipeline.
- `datasets`: Input data sources for the pipeline.
- `pipeline`: The sequence of steps to execute, including input and output specifications.

## Operation Types

docetl supports various operation types, each designed for specific data transformation tasks. All prompt templates used in these operations are Jinja2 templates, allowing for the use of loops, conditionals, and other Jinja2 features to create dynamic prompts based on input data.

All operations have the following optional parameters:

- `optimize`: Boolean flag. If true, the operation will be optimized. Default is True.
- `recursively_optimize`: Boolean flag. If true, the operation will be recursively optimized (e.g., reduces generated in map operations will be optimized). Default is false. I recommend not settting this to true unless you are willing to babysit the optimizer.
- `sample_size`: Integer. The number of samples to use for the operation, if you want to run it only on a sample of data. (Only applicable at runtime, not in optimization time.)

Here's an overview of the supported operation types:

### Map

The Map operation applies a transformation to each item in the input data.

Required parameters:

- `name`: A unique name for the operation.
- `type`: Must be set to `"map"`.
- `prompt`: The prompt template to use for the transformation. Access variables with `input.keyname`
- `output`: Schema definition for the output from the LLM.
- `model` (optional): The language model to use, falls back to `default_model` if not specified.

Example:

```yaml
- name: sentiment
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

- `name`: A unique name for the operation.
- `type`: Must be set to `"parallel_map"`.
- `prompts`: A list of prompt configurations, each containing:
  - `name`: A unique name for the prompt.
  - `prompt`: The prompt template to use for the transformation.
  - `output_keys`: List of keys that this prompt will generate.
  - `model` (optional): The language model to use for this specific prompt.
- `output`: Schema definition for the combined output from all prompts.

Example:

```yaml
- name: sentiment
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

```yaml
- name: extract_info
  type: map
  model: gpt-4o-mini
  output:
    schema:
      officer_name: string
      suspect_name: string
  prompts:
    - name: officer
      prompt: |
      Infer the officer's name from this police interrogation transcript:
        { { input.transcript } }
      output_keys:
        - officer_name
    - name: suspect
      prompt: |
      Infer the suspect's name from this police interrogation transcript:
        { { input.transcript } }
      output_keys:
        - suspect_name
```

### Filter

The Filter operation selects items from the input data based on a condition.

Required parameters:

- `type`: Must be set to `"filter"`.
- `name`: A unique name for the operation.
- `prompt`: The prompt template to use for the filtering condition.
- `output`: Schema definition for the output from the LLM. It must include only one field, a boolean field. This field can be named anything, but it must be a boolean field.
- `model` (optional): The language model to use, falls back to `default_model` if not specified.

Example:

```yaml
- name: filter
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

- `name`: A unique name for the operation.
- `type`: Must be set to `"unnest"`.
- `unnest_key`: The key of the array field to unnest.

Optional parameters:

- `keep_empty`: Boolean flag. If true, empty arrays being exploded will be kept in the output (with value None). Default is false.
- `expand_fields`: A list of fields to expand from the nested dictionary into the parent dictionary, if unnesting a dictionary.
- `recursive`: Boolean flag. If true, the unnest operation will be applied recursively to the nested arrays. Default is false.

Example of a list unnest:

```yaml
- name: unnest_people
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
- name: unnest_people
  type: unnest
  unnest_key: people
  expand_fields:
    - name
    - age
```

The above example will unnest the `people` field, expanding the `name` and `age` fields from the nested dictionary into the parent dictionary. For example, if the input data is:

```yaml
- name: unnest_people
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

The Equijoin operation performs a join between two datasets using embedding similarity and a language model for comparison.

Required parameters:

- `type`: Must be set to `"equijoin"`.
- `name`: A unique name for the operation.
- `comparison_model`: The language model to use for comparing join candidates.
- `comparison_prompt`: The prompt template to use for comparing join candidates. This should be designed to elicit a yes or no answer.

Optional parameters:

- `blocking_keys`: Specification of the blocking keys for left and right datasets. If provided, both left and right must be specified as lists of key names to use for blocking. We will embed the values of these keys and use the embeddings to filter the potential matches before performing the join.
- `limits`: Specification of the maximum number of matches for each tuple. If provided, both left and right must be specified with integer values.
- `embedding_model`: The model to use for creating embeddings. Only used if blocking threshold is set.
- `blocking_threshold`: Embedding similarity threshold for considering entries as potential matches.
- `blocking_conditions`: List of conditions for initial blocking.

Example:

```yaml
- name: join_book_author
  type: equijoin
  blocking_keys:
    left: ["genre"]
    right: ["primary_genre"]
  limits:
    left: 3
    right: 3
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

The Split operation divides long text content into smaller chunks.

Required parameters:

- `name`: A unique name for the operation.
- `type`: Must be set to "split".
- `split_key`: The key of the field containing the text to split.
- `method`: The method to use for splitting. Options are "delimiter" and "token_count".
- `method_kwargs`: A dictionary of keyword arguments to pass to the splitting method.
  - `delimiter`: The delimiter to use for splitting. Only used if method is "delimiter".
  - `token_count`: The maximum number of tokens to include in each chunk. Only used if method is "token_count".

Optional parameters:

- model: The language model's tokenizer to use; falls back to default_model if not specified. Note that we don't actually run a language model here.
- num_splits_to_group: The number of splits to group together into one chunk. Only used if method is "delimiter".

Example:

```yaml
- name: split_operation
  type: split
  split_key: content
  method: token_count
  method_kwargs:
    token_count: 150
  model: gpt-4o-mini
```

Notes:

- The split operation acts like a flatmap. For each input item, it produces multiple output items, one for each chunk.
- Each output item contains all the original key-value pairs from the input item, plus:
  - {split_key}\_chunk: The content of the split chunk.
  - {name}\_id: A unique identifier for each original document.
  - {name}\_chunk_num: The sequential number of the chunk within its original document.

### Gather

The Gather operation adds contextual information from surrounding chunks to each chunk.

Required parameters:

- `name`: A unique name for the operation.
- `type`: Must be set to "gather".
- `content_key`: The key containing the chunk content.
- `doc_id_key`: The key containing the document ID.
- `order_key`: The key containing the chunk order number.

Optional parameters:

- peripheral_chunks: A dictionary specifying how to handle chunks before and after the current chunk.
  - previous: Configuration for chunks before the current chunk.
  - next: Configuration for chunks after the current chunk.
- main_chunk_start: A string to prefix the main chunk content (default: "--- Begin Main Chunk ---").
- main_chunk_end: A string to suffix the main chunk content (default: "--- End Main Chunk ---").

Both previous and next can contain the following optional sections:

- head: Chunks at the beginning of the document.
- middle: Chunks between the head and tail.
- tail: Chunks closest to the current chunk.

Each section (head, middle, tail) can have a count property specifying the number of chunks to include.

Example:

```yaml
- name: gather_operation
  type: gather
  content_key: content_chunk
  doc_id_key: split_id
  order_key: split_chunk_num
  peripheral_chunks:
    previous:
      tail:
        count: 3
    next:
      head:
        count: 3
  main_chunk_start: "--- Begin Main Chunk ---"
  main_chunk_end: "--- End Main Chunk ---"
```

Notes:

- The gather operation adds a new field to each item: {content_key}\_rendered, which contains the formatted chunk with added context.
- The formatted content includes labels for previous context, main chunk, and next context.
- Skipped chunks are indicated with a "[... X characters skipped ...]" message.

### Reduce

The Reduce operation aggregates data based on a key. It supports both batch reduction and incremental folding for large datasets.

Required parameters:

- `name`: A unique name for the operation.
- `type`: Must be set to `"reduce"`.
- `reduce_key`: The key to use for grouping data. This can be a single key (string) or a list of keys.
- `prompt`: The prompt template to use for the reduction operation. This template can access the grouped values using `{{ inputs }}` (a list of dictionary objects or records) and the reduce key using `{{ reduce_key }}`.
- `output`: Schema definition for the output from the LLM.

Optional parameters:

- `synthesize_resolve`: Boolean flag. If false, we will not synthesize a resolve operation in between a map and a reduce operation. Default is true.
- `synthesize_merge`: Boolean flag. If false, we will not synthesize a merge optimization (we will only rely on folding). Default is true.
- `model`: The language model to use, falls back to `default_model` if not specified.
- `input`: Specifies the schema or keys to subselect from each item or value to pass into the prompt. If omitted, all keys from the input items will be used.
- `pass_through`: Boolean flag. If true, keys (not on input) from the first item in the group will be passed through to the output. Default is false.
- `associative`: Boolean flag. If true, the reduce operation is associative, meaning the order of operations doesn't matter. This can enable further optimizations. Default is true.
- `fold_prompt`: A prompt template for incremental folding. This enables processing of large groups in smaller batches. The template should access the current reduced values using `{{ output.field_name }}` and the new batch of values using `{{ inputs }}`.
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

```yaml
- name: reduce_operation
  type: reduce
  reduce_key: category
  prompt: |
    Analyze the following items in the category '{{ reduce_key.category }}':
    {% for item in inputs %}
    - {{ item.name }}: ${{ item.price }}
    {% endfor %}

    Provide a summary with:
    1. The total number of items
    2. The total value of all items
    3. The average price of items
  output:
    schema:
      item_count: integer
      total_value: number
      average_price: number
  model: gpt-3.5-turbo
  value_sampling:
    enabled: true
    method: cluster
    sample_size: 50
    embedding_model: text-embedding-3-small
    embedding_keys:
      - name
      - price
```

Example of a basic reduce operation:

```yaml
- name: reduce_operation
  type: reduce
  reduce_key: group
  input:
    schema:
      age: integer
  prompt: |
    Analyze the following group of values for the group '{{ reduce_key }}':
    {% for value in inputs %}
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
- name: reduce_operation
  type: reduce
  reduce_key: group
  prompt: |
    Analyze the following group of values for the group '{{ reduce_key }}':
    {% for value in inputs %}
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
    {% for value in inputs %}
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
- `resolution_prompt`: The prompt template to use for reducing matched entries. The matched entries are accessed via the `inputs` variable.
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
- name: resolve_operation
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

    {% for entry in inputs %}
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
  embedding_model: text-embedding-3-small
  resolution_model: gpt-4o-mini
  comparison_model: gpt-4o-mini
```

## Schemas

### Schema Definition

Schemas in docetl are defined using a simple key-value structure, where each key represents a field name and the value specifies the data type. The supported data types are:

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

## Tool Use

docetl supports the use of tools in operations, allowing for more complex and specific data processing tasks. Tools are defined as Python functions that can be called by the language model during execution.

To use tools in an operation, you need to define them in the operation's configuration. Here's an example of how to define and use a tool:

```yaml
operations:
  - name: word_count_analysis
    type: map
    prompt: |
      Count the number of words in the following book title:
      Title: {{ input.title }}
    tools:
      - required: true
        code: |
          def count(iterable):
            return {"word_count": len(iterable)}
        function:
          name: count
          description: Count the number of items in an iterable. Must pass in a list.
          parameters:
            type: object
            properties:
              iterable:
                type: array
                items:
                  type: string
          required:
            - iterable
    output:
      schema:
        word_count: int
```

In this example:

1. The tool is defined within the `tools` list of the operation.
2. The `required` field indicates whether the tool must be used in the operation.
3. The `code` field contains the actual Python function definition. The function must return a dictionary with keys specified in the `output` schema.
4. The `function` field provides metadata about the tool, including its name, description, and parameters.

The language model can then use this tool to count words in the input title. The tool's output will be incorporated into the operation's result according to the defined output schema.

You can define multiple tools for an operation, allowing the model to choose the most appropriate one for the task at hand. Tools can range from simple utility functions to more complex data processing or external API calls, enhancing the capabilities of your docetl pipeline.

Currently, only map and parallel_map operations support tools.

## Validation Rules

You can add validation rules to your operations to ensure the output meets certain criteria. Validation rules are specified using Python expressions.

Example:

```yaml
- name: map_operation
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

Here's an example of a pipeline that extracts themes from student survey responses, unnests the themes, and then summarizes the responses for each theme:

```yaml
default_model: gpt-4o-mini

datasets:
  student_submissions:
    type: file
    path: "data/student_survey_responses.json" # Assuming all items have a "survey_response" attribute

operations:
  - name: extract_themes
    type: map
    prompt: |
      I'm teaching a class on databases. Analyze the following student survey response:

      {{ input.survey_response }}

      Extract 2-3 main themes from this response, each being 1-2 words. Return the themes as a list of strings.
    output:
      schema:
        theme: list[str]
    validate:
      - len(output["theme"]) >= 2)
    num_retries_on_validate_failure: 3

  - name: unnest_themes
    type: unnest
    unnest_key: theme

  - name: resolve_themes
    type: resolve
    embedding_model: text-embedding-3-small
    blocking_threshold: 0.7
    blocking_keys:
      - theme
    limit_comparisons: 1000 # You can change this or remove it entirely

    comparison_prompt: |
      Compare the following two themes extracted from student survey responses about a database class:

      Theme 1: {{ input1.theme }}
      Theme 2: {{ input2.theme }}

      Are these themes similar/should they be merged?
    resolution_prompt: |
      You are merging similar themes from student survey responses about a database class. Here are the themes to merge:

      {% for theme in inputs %}
      Theme {{ loop.index }}: {{ theme.theme }}
      {% endfor %}

      Create a single, concise theme that captures the essence of all these themes.
    output: # Merge prompt output. no need to define schema for comparison prompt output
      schema:
        theme: str
    model: gpt-4o-mini

  - name: summarize_themes
    type: reduce
    reduce_key: theme
    prompt: |
      I am teaching a class on databases. You are helping me analyze student survey responses. Summarize the responses for the theme: {{ inputs[0].theme }}

      Responses:
      {% for item in inputs %}
      Survey {{ loop.index }}:
      - {{ item.survey_response }}
      {% endfor %}

      Summarize the main points from the surveys expressed about this theme. Do not mention any names of students or any other identifying information.
    output:
      schema:
        summary: str

pipeline:
  steps:
    - name: extract_response_themes
      input: student_submissions
      operations:
        - extract_themes
        - unnest_themes
        - summarize_themes

  output:
    type: file
    path: "theme_summaries.json" # Your summaries will be saved to the summary key
```

To run this pipeline, save it as `pipeline.yaml` and execute:

```bash
docetl run pipeline.yaml
```

This will process the student submissions data, extract themes from each response, unnest the themes, summarize the responses for each theme, and save the theme summaries in `output/theme_summaries.json`.
