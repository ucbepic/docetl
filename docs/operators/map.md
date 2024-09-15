# Map Operation

The Map operation in DocETL applies a specified transformation to each item in your input data, allowing for complex processing and insight extraction from large, unstructured documents.

## 🚀 Example: Analyzing Long-Form News Articles

Let's see a practical example of using the Map operation to analyze long-form news articles, extracting key information and generating insights.

```yaml
- name: analyze_news_article
  type: map
  prompt: |
    Analyze the following news article:
    "{{ input.article }}"

    Provide the following information:
    1. Main topic (1-3 words)
    2. Summary (2-3 sentences)
    3. Key entities mentioned (list up to 5, with brief descriptions)
    4. Sentiment towards the main topic (positive, negative, or neutral)
    5. Potential biases or slants in reporting (if any)
    6. Relevant categories (e.g., politics, technology, environment; list up to 3)
    7. Credibility score (1-10, where 10 is highly credible)

  output:
    schema:
      main_topic: string
      summary: string
      key_entities: list[object]
      sentiment: string
      biases: list[string]
      categories: list[string]
      credibility_score: integer

  model: gpt-4o-mini
  validate:
    - len(output["main_topic"].split()) <= 3
    - len(output["key_entities"]) <= 5
    - output["sentiment"] in ["positive", "negative", "neutral"]
    - len(output["categories"]) <= 3
    - 1 <= output["credibility_score"] <= 10
  num_retries_on_validate_failure: 2
```

This Map operation processes long-form news articles to extract valuable insights:

1. Identifies the main topic of the article.
2. Generates a concise summary.
3. Extracts key entities (people, organizations, locations) mentioned in the article.
4. Analyzes the overall sentiment towards the main topic.
5. Identifies potential biases or slants in the reporting.
6. Categorizes the article into relevant topics.
7. Assigns a credibility score based on the content and sources.

The operation includes validation to ensure the output meets our expectations and will retry up to 2 times if validation fails.

??? example "Sample Input and Output"

    Input:
    ```json
    [
      {
        "article": "In a groundbreaking move, the European Union announced yesterday a comprehensive plan to transition all member states to 100% renewable energy by 2050. The ambitious proposal, dubbed 'Green Europe 2050', aims to completely phase out fossil fuels and nuclear power across the continent.

        European Commission President Ursula von der Leyen stated, 'This is not just about fighting climate change; it's about securing Europe's energy independence and economic future.' The plan includes massive investments in solar, wind, and hydroelectric power, as well as significant funding for research into new energy storage technologies.

        However, the proposal has faced criticism from several quarters. Some Eastern European countries, particularly Poland and Hungary, argue that the timeline is too aggressive and could damage their economies, which are still heavily reliant on coal. Industry groups have also expressed concern about the potential for job losses in the fossil fuel sector.

        Environmental groups have largely praised the initiative, with Greenpeace calling it 'a beacon of hope in the fight against climate change.' However, some activists argue that the 2050 target is not soon enough, given the urgency of the climate crisis.

        The plan also includes provisions for a 'just transition,' with billions of euros allocated to retraining workers and supporting regions that will be most affected by the shift away from fossil fuels. Additionally, it proposes stricter energy efficiency standards for buildings and appliances, and significant investments in public transportation and electric vehicle infrastructure.

        Experts are divided on the feasibility of the plan. Dr. Maria Schmidt, an energy policy researcher at the University of Berlin, says, 'While ambitious, this plan is achievable with the right political will and technological advancements.' However, Dr. John Smith from the London School of Economics warns, 'The costs and logistical challenges of such a rapid transition should not be underestimated.'

        As the proposal moves forward for debate in the European Parliament, it's clear that 'Green Europe 2050' will be a defining issue for the continent in the coming years, with far-reaching implications for Europe's economy, environment, and global leadership in climate action."
      }
    ]
    ```

    Output:
    ```json
    [
      {
        "main_topic": "EU Renewable Energy",
        "summary": "The European Union has announced a plan called 'Green Europe 2050' to transition all member states to 100% renewable energy by 2050. The ambitious proposal aims to phase out fossil fuels and nuclear power, invest in renewable energy sources, and includes provisions for a 'just transition' to support affected workers and regions.",
        "key_entities": [
          {
            "name": "European Union",
            "description": "Political and economic union of 27 member states"
          },
          {
            "name": "Ursula von der Leyen",
            "description": "European Commission President"
          },
          {
            "name": "Poland",
            "description": "Eastern European country critical of the plan"
          },
          {
            "name": "Hungary",
            "description": "Eastern European country critical of the plan"
          },
          {
            "name": "Greenpeace",
            "description": "Environmental organization supporting the initiative"
          }
        ],
        "sentiment": "positive",
        "biases": [
          "Slight bias towards environmental concerns over economic impacts",
          "More emphasis on supportive voices than critical ones"
        ],
        "categories": [
          "Environment",
          "Politics",
          "Economy"
        ],
        "credibility_score": 8
      }
    ]
    ```

This example demonstrates how the Map operation can transform long, unstructured news articles into structured, actionable insights. These insights can be used for various purposes such as trend analysis, policy impact assessment, and public opinion monitoring.

## Required Parameters

- `name`: A unique name for the operation.
- `type`: Must be set to "map".
- `prompt`: The prompt template to use for the transformation. Access input variables with `input.keyname`.
- `output`: Schema definition for the output from the LLM.

## Optional Parameters

| Parameter                         | Description                                                    | Default                       |
| --------------------------------- | -------------------------------------------------------------- | ----------------------------- |
| `model`                           | The language model to use                                      | Falls back to `default_model` |
| `optimize`                        | Flag to enable operation optimization                          | `True`                        |
| `recursively_optimize`            | Flag to enable recursive optimization                          | `false`                       |
| `sample_size`                     | Number of samples to use for the operation                     | Processes all data            |
| `tools`                           | List of tool definitions for LLM use                           | None                          |
| `validate`                        | List of Python expressions to validate the output              | None                          |
| `num_retries_on_validate_failure` | Number of retry attempts on validation failure                 | 0                             |
| `gleaning`                        | Configuration for advanced validation and LLM-based refinement | None                          |

!!! info "Validation and Gleaning"

    For more details on validation techniques and implementation, see [operators](../concepts/operators.md#validation).

## Advanced Features

### Tool Use

Tools can extend the capabilities of the Map operation. Each tool is a Python function that can be called by the LLM during execution, and follows the [OpenAI Function Calling API](https://platform.openai.com/docs/guides/function-calling).

??? example "Tool Definition Example"

    ```yaml
    tools:
    - required: true
        code: |
        def count_words(text):
            return {"word_count": len(text.split())}
        function:
        name: count_words
        description: Count the number of words in a text string.
        parameters:
            type: object
            properties:
            text:
                type: string
            required:
            - text
    ```

!!! warning

    Tool use and gleaning cannot be used simultaneously.

### Input Truncation

If the input doesn't fit within the token limit, DocETL automatically truncates tokens from the middle of the input data, preserving the beginning and end which often contain more important context. A warning is displayed when truncation occurs.

## Best Practices

1. **Clear Prompts**: Write clear, specific prompts that guide the LLM to produce the desired output.
2. **Robust Validation**: Use validation to ensure output quality and consistency.
3. **Appropriate Model Selection**: Choose the right model for your task, balancing performance and cost.
4. **Optimize for Scale**: For large datasets, consider using `sample_size` to test your operation before running on the full dataset.
5. **Use Tools Wisely**: Leverage tools for complex calculations or operations that the LLM might struggle with. You can write any Python code in the tools, so you can even use tools to call other APIs or search the internet.
