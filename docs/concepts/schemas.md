# Schemas

Every LLM call in DocETL has an output schema specifying the structure and types of its output. DocETL enforces schemas via structured outputs or the tool API (see [How We Enforce Schemas](#how-we-enforce-schemas)).

!!! tip "Schema Simplicity"

    We've observed that **the more complex the output schema is, the worse the quality of the output tends to be**. Keep your schemas as simple as possible for better results.

## Defining Schemas

Schemas are defined in the `output` section of an operator. They support various data types:

| Type      | Aliases                  | Description                                                  |
| --------- | ------------------------ | ------------------------------------------------------------ |
| `string`  | `str`, `text`, `varchar` | For text data                                                |
| `integer` | `int`                    | For whole numbers                                            |
| `number`  | `float`, `decimal`       | For decimal numbers                                          |
| `boolean` | `bool`                   | For true/false values                                        |
| `enum`    | -                        | For a set of possible values                                |
| `list`    | -                        | For arrays or sequences of items (must specify element type) |
| Objects   | -                        | Using notation `{field: type}`                               |

!!! note "Filter Operation Schemas"

    Filter operation schemas must have a boolean output field, which determines whether each item is kept.

## Examples

### Simple Schema

=== "YAML"

    ```yaml
    output:
      schema:
        summary: string
        sentiment: string
        include_item: boolean # For filter operations
    ```

=== "Python"

    ```python
    output={
        "schema": {
            "summary": "string",
            "sentiment": "string",
            "include_item": "boolean",  # For filter operations
        }
    }
    ```

### Complex Schema

=== "YAML"

    ```yaml
    output:
      schema:
        insights: "list[{insight: string, confidence: number}]"
        metadata: "{timestamp: string, source: string}"
    ```

=== "Python"

    ```python
    output={
        "schema": {
            "insights": "list[{insight: string, confidence: number}]",
            "metadata": "{timestamp: string, source: string}",
        }
    }
    ```

## Lists and Objects

Lists in schemas must specify their element type:

- `list[string]`: A list of strings
- `list[int]`: A list of integers
- `list[{name: string, age: integer}]`: A list of objects

Objects are defined using curly braces and must have typed fields:

- `{name: string, age: integer, is_active: boolean}`

!!! example "Complex List Example"

    === "YAML"

        ```yaml
        output:
          schema:
            users: "list[{name: string, age: integer, hobbies: list[string]}]"
        ```

        Make sure that you put the type in quotation marks, if it references an object type (i.e., has curly braces)! Otherwise the yaml won't compile!

    === "Python"

        ```python
        output={
            "schema": {
                "users": "list[{name: string, age: integer, hobbies: list[string]}]"
            }
        }
        ```

## Enum Types

Enum values are validated against the declared set of possible values:

=== "YAML"

    ```yaml
    output:
      schema:
        sentiment: "enum[positive, negative, neutral]"
    ```

=== "Python"

    ```python
    output={"schema": {"sentiment": "enum[positive, negative, neutral]"}}
    ```

Lists of enums also work:

=== "YAML"

    ```yaml
    output:
      schema:
        possible_sentiments: "list[enum[positive, negative, neutral]]"
    ```

=== "Python"

    ```python
    output={"schema": {"possible_sentiments": "list[enum[positive, negative, neutral]]"}}
    ```

## How We Enforce Schemas

DocETL supports two output modes that determine how the LLM generates structured outputs:

### Tools Mode (Default)

Uses the OpenAI tools/function calling API to enforce schema structure.

=== "YAML"

    ```yaml
    output:
      schema:
        summary: string
        sentiment: string
      mode: "tools"  # Optional - this is the default
    ```

=== "Python"

    ```python
    output={
        "schema": {
            "summary": "string",
            "sentiment": "string",
        },
        "mode": "tools",  # Optional - this is the default
    }
    ```

### Structured Output Mode

Uses LiteLLM's structured output feature with JSON schema validation. This mode can provide more reliable schema adherence for complex outputs.

=== "YAML"

    ```yaml
    output:
      schema:
        insights: "list[{insight: string, confidence: number}]"
      mode: "structured_output"
    ```

=== "Python"

    ```python
    output={
        "schema": {
            "insights": "list[{insight: string, confidence: number}]",
        },
        "mode": "structured_output",
    }
    ```

!!! tip "When to Use Structured Output Mode"

    Consider using `structured_output` mode when:

    - You have complex nested schemas with lists and objects
    - You need more consistent schema adherence
    - You're experiencing schema validation issues with tools mode

### Mode Configuration

Set `mode` in the `output` section of any operation:

=== "YAML"

    ```yaml
    operations:
      - name: analyze_text
        type: map
        prompt: "Analyze the following text..."
        output:
          schema:
            topics: "list[{topic: string, relevance: number}]"
          mode: "structured_output"  # or "tools"
        model: gpt-4o-mini
    ```

=== "Python"

    ```python
    pipeline = pipeline.map(
        name="analyze_text",
        prompt="Analyze the following text...",
        output={
            "schema": {"topics": "list[{topic: string, relevance: number}]"},
            "mode": "structured_output",  # or "tools"
        },
        model="gpt-4o-mini",
    )
    ```

## Best Practices

1. Keep output fields simple and use string types whenever possible.
2. Only use structured fields (like lists and objects) when necessary for downstream analysis or reduce operations.
3. If you need to reference structured fields in downstream operations, consider breaking complex structures into multiple simpler operations.

!!! example "Breaking Down Complex Schemas"

    Instead of:

    === "YAML"

        ```yaml
        output:
          schema:
            summary: string
            key_points: "list[{point: string, sentiment: string}]"
        ```

    === "Python"

        ```python
        output={
            "schema": {
                "summary": "string",
                "key_points": "list[{point: string, sentiment: string}]",
            }
        }
        ```

    Consider:

    === "YAML"

        ```yaml
        output:
          schema:
            summary: string
            key_points: "string"
        ```

    === "Python"

        ```python
        output={
            "schema": {
                "summary": "string",
                "key_points": "string",
            }
        }
        ```

    Where in the prompt you can say something like: `In your key points, please include the sentiment of each point.`

    The only reason to use the complex schema is if you need to do an operation at the point level, like resolve them and reduce on them.
