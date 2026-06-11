# Operators

Operators are the building blocks of DocETL pipelines; each semantically processes unstructured data.

- Datasets contain items: objects in the JSON list, with fields and values. An item can be a text chunk or a document reference.
- By default, operations are parallelized over your data using multithreading.

!!! tip "Caching in DocETL"

    DocETL caches all LLM calls and partially-optimized plans, in `.cache/docetl/general` and `.cache/docetl/llm` in your home directory. Rerunning similar operations or reprocessing data avoids redundant API calls.

## Common Attributes

All operators share some common attributes:

- `name`: A unique identifier for the operator.
- `type`: Specifies the type of operation (e.g., "map", "reduce", "filter").

LLM-based operators have additional attributes:

- `prompt`: A Jinja2 template that defines the instruction for the language model.
- `output`: Specifies the schema for the output from the LLM call.
- `model` (optional): Allows specifying a different model from the pipeline default.
- `litellm_completion_kwargs` (optional): Additional parameters to pass to LiteLLM completion calls.

DocETL executes all LLM calls through [LiteLLM](https://docs.litellm.ai), which supports 100+ providers including OpenAI, Anthropic, and Azure.

Example:

=== "YAML"

    ```yaml
    - name: extract_insights
      type: map
      model: gpt-4o-mini
      litellm_completion_kwargs:
        max_tokens: 500          # limit response length
        temperature: 0.7         # control randomness
        top_p: 0.9              # nucleus sampling parameter
      prompt: |
        Analyze the following user interaction log:
        {{ input.log }}

        Extract 2-3 main insights from this log, each being 1-2 words, to help inform future product development. Consider any difficulties or pain points the user may have had. Also provide 1-2 supporting actions for each insight.
        Return the results as a list of dictionaries, each containing 'insight' and 'supporting_actions' keys.
      output:
        schema:
          insights: "list[{insight: string, supporting_actions: list[string]}]"
    ```

=== "Python"

    ```python
    pipeline = pipeline.map(
        name="extract_insights",
        model="gpt-4o-mini",
        litellm_completion_kwargs={
            "max_tokens": 500,   # limit response length
            "temperature": 0.7,  # control randomness
            "top_p": 0.9,        # nucleus sampling parameter
        },
        prompt="""Analyze the following user interaction log:
    {{ input.log }}

    Extract 2-3 main insights from this log, each being 1-2 words, to help inform future product development. Consider any difficulties or pain points the user may have had. Also provide 1-2 supporting actions for each insight.
    Return the results as a list of dictionaries, each containing 'insight' and 'supporting_actions' keys.""",
        output={
            "schema": {
                "insights": "list[{insight: string, supporting_actions: list[string]}]"
            }
        },
    )
    ```

## Input and Output

Prompts can reference any fields in the data, including:

- Original fields from the input data.
- Fields synthesized by previous operations in the pipeline.

For map operations, you can only reference `input`, but in reduce operations, you can reference `inputs` (since it's a list of inputs).

Example:

=== "YAML"

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

=== "Python"

    ```python
    prompt="""Summarize the user behavior insights for the country: {{ inputs[0].country }}

    Insights and supporting actions:
    {% for item in inputs %}
    - Insight: {{ item.insight }}
    Supporting actions:
    {% for action in item.supporting_actions %}
    - {{ action }}
    {% endfor %}
    {% endfor %}"""
    ```

!!! question "What happens if the input is too long?"

    When the input data exceeds the token limit of the LLM, DocETL automatically truncates tokens from the middle of the data to make it fit in the prompt. This approach preserves the beginning and end of the input, which often contain crucial context.

    A warning is displayed whenever truncation occurs, alerting you to potential loss of information:

    ```
    WARNING: Input exceeded token limit. Truncated 500 tokens from the middle of the input.
    ```

    If you frequently encounter this warning, consider using DocETL's optimizer or breaking your input into smaller chunks.

## Output Schema

The `output` attribute defines the structure of the LLM's response. It supports various data types (see [schemas](../concepts/schemas.md) for more details):

- `string` (or `str`, `text`, `varchar`): For text data
- `integer` (or `int`): For whole numbers
- `number` (or `float`, `decimal`): For decimal numbers
- `boolean` (or `bool`): For true/false values
- `list`: For arrays or sequences of items
- objects: Using notation `{field: type}`
- `enum`: For a set of possible values

Example:

=== "YAML"

    ```yaml
    output:
      schema:
        insights: "list[{insight: string, supporting_actions: string}]"
        detailed_summary: string
    ```

=== "Python"

    ```python
    output={
        "schema": {
            "insights": "list[{insight: string, supporting_actions: string}]",
            "detailed_summary": "string",
        }
    }
    ```

!!! tip "Keep Output Types Simple"

    Complex nested structures are harder for the LLM to produce consistently and can cause parsing errors. Break them into multiple simpler operations instead.

    For example, instead of:

    === "YAML"

        ```yaml
        output:
          schema:
            insights: "list[{insight: string, supporting_actions: list[{action: string, priority: integer}]}]"
        ```

    === "Python"

        ```python
        output={
            "schema": {
                "insights": "list[{insight: string, supporting_actions: list[{action: string, priority: integer}]}]"
            }
        }
        ```

    Consider:

    === "YAML"

        ```yaml
        output:
          schema:
            insights: "list[{insight: string, supporting_actions: string}]"
        ```

    === "Python"

        ```python
        output={
            "schema": {
                "insights": "list[{insight: string, supporting_actions: string}]"
            }
        }
        ```

    And then use a separate operation to further process the supporting actions if needed.

## Validation

### Basic Validation

LLM-based operators can include a `validate` field, which accepts a list of Python statements:

=== "YAML"

    ```yaml
    validate:
      - len(output["insights"]) >= 2
      - all(len(insight["supporting_actions"]) >= 1 for insight in output["insights"])
    ```

=== "Python"

    ```python
    validate=[
        lambda output: len(output["insights"]) >= 2,
        lambda output: all(len(i["supporting_actions"]) >= 1 for i in output["insights"]),
    ]
    ```

    Entries may be callables taking the output dict, or the same expression
    strings as YAML. Use strings if you plan to export with `to_yaml()`.

Access variables using dictionary syntax: `output["field"]`. Note that you can't access `input` docs in validation, but the output docs should have all the fields from the input docs (for non-reduce operations), since fields pass through unchanged.

The `num_retries_on_validate_failure` attribute specifies how many times to retry the LLM if any validation statements fail.

### Advanced Validation: Gleaning

Gleaning is an advanced validation technique that uses LLM-based validators to refine outputs iteratively.

To enable gleaning, specify:

- `validation_prompt`: Instructions for the LLM to evaluate and improve the output.
- `num_rounds`: The maximum number of refinement iterations.
- `model` (optional): The model to use for the LLM executing the validation prompt. Defaults to the model specified for this operation; a cheaper model can be used here to reduce validation cost. **Note that if the validator LLM determines the output needs to be improved, the final output will be generated by the model specified for this operation.**
- `if` (optional): A Python boolean expression (evaluated with `safe_eval`) that refers to **fields in the current `output`**. If the expression evaluates to `False`, DocETL skips gleaning entirely. If omitted, gleaning always runs.

Example:

=== "YAML"

    ```yaml
    gleaning:
      num_rounds: 1
      validation_prompt: |
        Evaluate the extraction for completeness and relevance:
        1. Are all key user behaviors and pain points from the log addressed in the insights?
        2. Are the supporting actions practical and relevant to the insights?
        3. Is there any important information missing or any irrelevant information included?
    ```

=== "Python"

    ```python
    gleaning={
        "num_rounds": 1,
        "validation_prompt": """Evaluate the extraction for completeness and relevance:
    1. Are all key user behaviors and pain points from the log addressed in the insights?
    2. Are the supporting actions practical and relevant to the insights?
    3. Is there any important information missing or any irrelevant information included?""",
    }
    ```

Gleaning is expensive: it at least doubles the number of LLM calls for each operator.

Example map operation (with a different model for the validation prompt):

=== "YAML"

    ```yaml
    - name: extract_insights
      type: map
      model: gpt-4o
      prompt: |
        From the user log below, list 2-3 concise insights (1-2 words each) and 1-2 supporting actions per insight.
        Return as a list of dictionaries with 'insight' and 'supporting_actions'.
        Log: {{ input.log }}
      output:
        schema:
          insights_summary: "string"
      gleaning:
        if: "len(output['insights_summary']) < 10"  # Only refine if summary is too short
        num_rounds: 2 # Will refine up to 2 times if needed
        model: gpt-4o-mini
        validation_prompt: |
          There should be at least 2 insights, and each insight should have at least 1 supporting action.
    ```

=== "Python"

    ```python
    pipeline = pipeline.map(
        name="extract_insights",
        model="gpt-4o",
        prompt="""From the user log below, list 2-3 concise insights (1-2 words each) and 1-2 supporting actions per insight.
    Return as a list of dictionaries with 'insight' and 'supporting_actions'.
    Log: {{ input.log }}""",
        output={"schema": {"insights_summary": "string"}},
        gleaning={
            "if": "len(output['insights_summary']) < 10",  # Only refine if summary is too short
            "num_rounds": 2,  # Will refine up to 2 times if needed
            "model": "gpt-4o-mini",
            "validation_prompt": "There should be at least 2 insights, and each insight should have at least 1 supporting action.",
        },
    )
    ```

### How Gleaning Works

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