# Filter Operation

The Filter operation behaves like Map, except items whose boolean output evaluates to false are dropped from the dataset.

## Example: Filtering High-Impact News Articles

=== "YAML"

    ```yaml
    - name: filter_high_impact_articles
      type: filter
      prompt: |
        Analyze the following news article:
        Title: "{{ input.title }}"
        Content: "{{ input.content }}"

        Determine if this article is high-impact based on the following criteria:
        1. Covers a significant global or national event
        2. Has potential long-term consequences
        3. Affects a large number of people
        4. Is from a reputable source

        Respond with 'true' if the article meets at least 3 of these criteria, otherwise respond with 'false'.

      output:
        schema:
          is_high_impact: boolean

      model: gpt-4-turbo
      validate:
        - isinstance(output["is_high_impact"], bool)
    ```

=== "Python"

    ```python
    import docetl

    docetl.default_model = "gpt-4-turbo"

    frame = docetl.read_json("articles.json")
    frame = frame.filter(
        prompt="""Analyze the following news article:
    Title: "{{ input.title }}"
    Content: "{{ input.content }}"

    Determine if this article is high-impact based on the following criteria:
    1. Covers a significant global or national event
    2. Has potential long-term consequences
    3. Affects a large number of people
    4. Is from a reputable source

    Respond with 'true' if the article meets at least 3 of these criteria, otherwise respond with 'false'.""",
        output={"schema": {"is_high_impact": "boolean"}},
        model="gpt-4-turbo",
        validate=["isinstance(output['is_high_impact'], bool)"],
    )
    df = frame.collect()
    ```

??? example "Sample Input and Output"

    **Input:**
    ```json
    [
      {
        "title": "Global Climate Summit Reaches Landmark Agreement",
        "content": "In a historic move, world leaders at the Global Climate Summit have unanimously agreed to reduce carbon emissions by 50% by 2030. This unprecedented agreement involves all major economies and sets binding targets for renewable energy adoption, reforestation, and industrial emissions reduction. Experts hail this as a turning point in the fight against climate change, with potential far-reaching effects on global economies, energy systems, and everyday life for billions of people."
      },
      {
        "title": "Local Bakery Wins Best Croissant Award",
        "content": "Downtown's favorite bakery, 'The Crusty Loaf', has been awarded the title of 'Best Croissant' in the annual City Food Festival. Owner Maria Garcia attributes the win to their use of imported French butter and a secret family recipe. Local food critics praise the bakery's commitment to traditional baking methods."
      }
    ]
    ```

    **Output:**
    ```json
    [
      {
        "title": "Global Climate Summit Reaches Landmark Agreement",
        "content": "In a historic move, world leaders at the Global Climate Summit have unanimously agreed to reduce carbon emissions by 50% by 2030. This unprecedented agreement involves all major economies and sets binding targets for renewable energy adoption, reforestation, and industrial emissions reduction. Experts hail this as a turning point in the fight against climate change, with potential far-reaching effects on global economies, energy systems, and everyday life for billions of people."
      }
    ]
    ```

## Configuration

### Required Parameters

- `name`: A unique name for the operation.
- `type`: Must be set to "filter".
- `prompt`: The prompt template to use for the filtering condition. Access input variables with `input.keyname`.
- `output`: Schema definition for the output from the LLM. It must include only one field, a boolean field.

### Optional Parameters

See [map optional parameters](./map.md#optional-parameters) for additional configuration options, including `batch_prompt` and `max_batch_size`.

### Model Cascade (cost reduction)

A `cascade` block runs a cheap proxy model on all items first and only escalates uncertain cases to the expensive oracle model, with a statistical quality guarantee.

=== "YAML"

    ```yaml
    - name: is_relevant
      type: filter
      model: gpt-4o
      prompt: "Is this document about climate policy? {{ input.text }}"
      output: { schema: { keep: "bool" } }
      cascade:
        proxy_model: gpt-4o-mini
        target: 0.95
    ```

=== "Python"

    ```python
    pipeline = pipeline.filter(
        name="is_relevant",
        model="gpt-4o",
        prompt="Is this document about climate policy? {{ input.text }}",
        output={"schema": {"keep": "bool"}},
        cascade={"proxy_model": "gpt-4o-mini", "target": 0.95},
    )
    ```

| Parameter | Description | Default |
|---|---|---|
| `proxy_model` | The cheap model for the proxy pass (required) | — |
| `guarantee` | `accuracy`, `precision`, `recall`, or `precision+recall` | `recall` |
| `target` | Target value for the guarantee metric, in `(0, 1)` (required) | — |
| `delta` | Failure probability; guarantee holds w.p. `1 - delta` | `0.05` |
| `label_budget` | Max oracle calls spent learning the threshold | `400` |

`proxy_model` can be a chat model (scored by logprobs) or an embedding model
like `text-embedding-3-small` (a logistic head is fitted on an oracle-labeled
slice of the budget — far cheaper per item for high-volume topical filters).

See [Model Cascades with BARGAIN](../optimization/cascades.md) for full details,
guarantee explanations, and examples.

### Limiting filtered outputs

For filter, `limit` counts only retained documents (boolean output `true`). DocETL evaluates inputs until it has collected `limit` passing documents, then stops scheduling LLM calls — so you can request "the first N matches" without scoring the entire dataset.

!!! info "Validation"

    For more details on validation techniques and implementation, see [operators](../concepts/operators.md#validation).

## Best Practices

1. **Boolean Output**: The output schema must have exactly one boolean field; write the prompt so the LLM produces a clear true/false judgment.
2. **Data Flow Awareness**: Unlike Map, Filter reduces the size of your dataset.
