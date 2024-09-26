# Operators

Operators in DocETL are designed for semantically processing unstructured data. They form the building blocks of data processing pipelines, allowing you to transform, analyze, and manipulate datasets efficiently.

## Overview

- Datasets contain documents, where a document is an object in the JSON list, with fields and values.
- DocETL provides several operators, each tailored for specific unstructured data processing tasks.
- By default, operations are parallelized on your data using multithreading for improved performance.

!!! tip "Caching in DocETL"

    DocETL employs caching for all LLM calls and partially-optimized plans. The cache is stored in the `.docetl/cache` and `.docetl/llm_cache` directories within your home directory. This caching mechanism helps to improve performance and reduce redundant API calls when running similar operations or reprocessing data.

## Common Attributes

All operators share some common attributes:

- `name`: A unique identifier for the operator.
- `type`: Specifies the type of operation (e.g., "map", "reduce", "filter").

LLM-based operators have additional attributes:

- `prompt`: A Jinja2 template that defines the instruction for the language model.
- `output`: Specifies the schema for the output from the LLM call.
- `model` (optional): Allows specifying a different model from the pipeline default.

## Input and Output

Prompts can reference any fields in the data, including:

- Original fields from the input data.
- Fields synthesized by previous operations in the pipeline.

For map operations, you can only reference `input`, but in reduce operations, you can reference `inputs` (since it's a list of inputs).

Example:

```yaml
prompt: |
  Summarize the user behavior insights for the country: {{ inputs[0].country }}

  Insights and supporting actions:
  {% for item in inputs %}
  - Insight: {{ item.insight }}
  Supporting actions:
  {% for action in item.supporting_actions %}
  - {{ action }}
  {% endfor %}
  {% endfor %}
```

!!! question "What happens if the input is too long?"

    When the input data exceeds the token limit of the LLM, DocETL automatically truncates tokens from the middle of the data to make it fit in the prompt. This approach preserves the beginning and end of the input, which often contain crucial context.

    A warning is displayed whenever truncation occurs, alerting you to potential loss of information:

    ```
    WARNING: Input exceeded token limit. Truncated 500 tokens from the middle of the input.
    ```

    If you frequently encounter this warning, consider using DocETL's optimizer or breaking down your input yourself into smaller chunks to handle large inputs more effectively.

## Output Schema

The `output` attribute defines the structure of the LLM's response. It supports various data types:

- `string` (or `str`, `text`, `varchar`): For text data
- `integer` (or `int`): For whole numbers
- `number` (or `float`, `decimal`): For decimal numbers
- `boolean` (or `bool`): For true/false values
- `list`: For arrays or sequences of items
- objects: Using notation `{field: type}`

Example:

```yaml
output:
  schema:
    insights: "list[{insight: string, supporting_actions: string}]"
    detailed_summary: string
```

!!! tip "Keep Output Types Simple"

    It's recommended to keep output types as simple as possible. Complex nested structures may be difficult for the LLM to consistently produce, potentially leading to parsing errors. The structured output feature works best with straightforward schemas. If you need complex data structures, consider breaking them down into multiple simpler operations.

    For example, instead of:
    ```yaml
    output:
      schema:
        insights: "list[{insight: string, supporting_actions: list[{action: string, priority: integer}]}]"
    ```

    Consider:
    ```yaml
    output:
      schema:
        insights: "list[{insight: string, supporting_actions: string}]"
    ```

    And then use a separate operation to further process the supporting actions if needed.

    Read more about schemas in the [schemas](../concepts/schemas.md) section.

## Validation

Validation is a first-class citizen in DocETL, ensuring the quality and correctness of processed data.

### Basic Validation

LLM-based operators can include a `validate` field, which accepts a list of Python statements:

```yaml
validate:
  - len(output["insights"]) >= 2
  - all(len(insight["supporting_actions"]) >= 1 for insight in output["insights"])
```

Access variables using dictionary syntax: `input["field"]` or `output["field"]`.

The `num_retries_on_validate_failure` attribute specifies how many times to retry the LLM if any validation statements fail.

### Advanced Validation: Gleaning

Gleaning is an advanced validation technique that uses LLM-based validators to refine outputs iteratively.

To enable gleaning, specify:

- `validation_prompt`: Instructions for the LLM to evaluate and improve the output.
- `num_rounds`: The maximum number of refinement iterations.

Example:

```yaml
gleaning:
  num_rounds: 1
  validation_prompt: |
    Evaluate the extraction for completeness and relevance:
    1. Are all key user behaviors and pain points from the log addressed in the insights?
    2. Are the supporting actions practical and relevant to the insights?
    3. Is there any important information missing or any irrelevant information included?
```

This approach allows for _context-aware_ validation and refinement of LLM outputs. Note that it is expensive, since it at least doubles the number of LLM calls required for each operator.

### How Gleaning Works

Gleaning is an iterative process that refines LLM outputs using context-aware validation. Here's how it works:

1. **Initial Operation**: The LLM generates an initial output based on the original operation prompt.

2. **Validation**: The validation prompt is appended to the chat thread, along with the original operation prompt and output. This is submitted to the LLM. _Note that the validation prompt doesn't need any variables, since it's appended to the chat thread._

3. **Assessment**: The LLM responds with an assessment of the output according to the validation prompt.

4. **Decision**: The system interprets the assessment:

   - If there's no error or room for improvement, the current output is returned.
   - If improvements are suggested, the process continues.

5. **Refinement**: If improvements are needed:

   - A new prompt is created, including the original operation prompt, the original output, and the validator feedback.
   - This is submitted to the LLM to generate an improved output.

6. **Iteration**: Steps 2-5 are repeated until either:

   - The validator has no more feedback (i.e., the evaluation passes), or
   - The number of iterations exceeds `num_rounds`.

7. **Final Output**: The last refined output is returned.

This process allows for nuanced, context-aware validation and refinement of LLM outputs. It's particularly useful for complex tasks where simple rule-based validation might miss subtleties or context-dependent aspects of the output.

Note that gleaning can significantly increase the number of LLM calls for each operator, potentially doubling it at minimum. While this increases cost and latency, it can lead to higher quality outputs for complex tasks.
``` 


Example:

```yaml
- name: extract_insights
  type: map
  model: gpt-4o
  prompt: |
    Analyze the following user interaction log:
    {{ input.log }}

    Extract 2-3 main insights from this log, each being 1-2 words, to help inform future product development. Consider any difficulties or pain points the user may have had. Also provide 1-2 supporting actions for each insight.
    Return the results as a list of dictionaries, each containing 'insight' and 'supporting_actions' keys.
  output:
    schema:
      insights: "list[{insight: string, supporting_actions: list[string]}]"
```

## Batching in Map Operations

### Batching

Batching in map operations allows you to process multiple input items simultaneously, improving efficiency and potentially reducing costs associated with API calls to language models. By grouping input data into batches, you can optimize the performance of your operations, especially when dealing with large datasets.

### How to Use and Configure Batching

To enable batching in your map operations, you need to specify the `batch_size` and optionally the `clustering_method` in your configuration. The `batch_size` determines how many input items will be processed together in a single call, while the `clustering_method` can be used to group similar items together for more efficient processing.

#### Clustering Methods

- `random`: Groups items into batches randomly.
- `sem_cluster`: Uses semantic similarity measures to cluster similar items together before batching. This method leverages similarity measures to ensure that items within a batch are contextually related, which can enhance the quality and coherence of the output.

### Example Configuration

Here’s an example of how to configure batching in a map operation with semantic clustering:

```yaml
- name: analyze_responses
  type: map
  batch_size: 10
  clustering_method: sem_cluster
  prompt: |
    Analyze the following set of responses:
    {% for response in inputs %}
    - {{ response.text }}
    {% endfor %}

    Provide a summary for each response.
  output:
    schema:
      summaries: "list[string]"
```

### Example Usage

You can use the configured map operation as follows:

```yaml
- name: extract_summaries
  type: map
  batch_size: 5
  clustering_method: random
  prompt: |
    Summarize each of the following items:
    {% for item in inputs %}
    - {{ item.description }}
    {% endfor %}
  output:
    schema:
      summaries: "list[string]"
```

## Trade-offs and Considerations for Batch Size Selection

When selecting a batch size, consider the following trade-offs:

1. **Performance:**
   Larger batch sizes can improve throughput by reducing the number of API calls, but they may also increase latency if the processing time for a single batch is significantly longer.

2. **Cost:**
   Depending on the pricing model of the language model API, larger batches may reduce the cost per item processed. However, if the batch size exceeds the model's limits, it could lead to errors or increased costs.

3. **Memory Usage:**
   Larger batches require more memory to hold the input data and the intermediate results. Ensure that your system has sufficient resources to handle the selected batch size.

4. **Accuracy:**
   When using clustering methods like `sem_cluster`, ensure that the items in a batch are similar enough to maintain the accuracy of the results. Poor clustering can lead to mixed results that may not accurately reflect the sentiment or other metrics being analyzed.

5. **Error Handling:**
   Consider how errors will be handled in batched operations. If one item in a batch fails, you may need to implement logic to retry or log the error without affecting the entire batch.
