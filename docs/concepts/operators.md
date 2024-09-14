# Operators

Operators in docetl are designed for semantically processing unstructured data. They form the building blocks of data processing pipelines, allowing you to transform, analyze, and manipulate datasets efficiently.

## Overview

- Datasets contain documents, where a document is an object in the JSON list, with fields and values.
- docetl provides several operators, each tailored for specific unstructured data processing tasks.
- By default, operations are parallelized on your data using multithreading for improved performance.

## Common Attributes

All operators share some common attributes:

- `name`: A unique identifier for the operator.
- `type`: Specifies the type of operation (e.g., "map", "reduce", "filter").

LLM-based operators have additional attributes:

- `prompt`: A Jinja2 template that defines the instruction for the language model.
- `output`: Specifies the schema for the output from the LLM call.
- `model` (optional): Allows specifying a different model from the pipeline default.

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

    When the input data exceeds the token limit of the LLM, docetl automatically truncates tokens from the middle of the data to make it fit in the prompt. This approach preserves the beginning and end of the input, which often contain crucial context.

    A warning is displayed whenever truncation occurs, alerting you to potential loss of information:

    ```
    WARNING: Input exceeded token limit. Truncated 500 tokens from the middle of the input.
    ```

    If you frequently encounter this warning, consider using docetl's optimizer or breaking down your input yourself into smaller chunks to handle large inputs more effectively.

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

Validation is a first-class citizen in docetl, ensuring the quality and correctness of processed data.

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
