# Tutorial: Mining User Behavior Data with docetl

This tutorial will guide you through the process of using docetl to analyze user behavior data from UI logs. We'll create a simple pipeline that extracts key insights and supporting actions from user logs, then summarizes them by country.

## Installation

First, let's install docetl. Follow the instructions in the [installation guide](installation.md) to set up docetl on your system.

## Setting up API Keys

docetl uses [LiteLLM](https://github.com/BerriAI/litellm) under the hood, which supports various LLM providers. For this tutorial, we'll use OpenAI, as docetl tests and existing pipelines are run with OpenAI.

!!! tip "Setting up API Key"

    Set your OpenAI API key as an environment variable:

    ```bash
    export OPENAI_API_KEY=your_api_key_here
    ```

    Alternatively, you can create a `.env` file in your project directory and add the following line:

    ```
    OPENAI_API_KEY=your_api_key_here
    ```

## Preparing the Data

Organize your user behavior data in a JSON file as a list of objects. Each object should have the following keys: "user_id", "country", and "log". The "log" field contains the user interaction logs.

!!! example "Sample Data Structure"

    ```json
    [
        {
            "user_id": "user123",
            "country": "USA",
            "log": "[2023-06-15 09:15:23] User opened app\n[2023-06-15 09:16:05] User clicked on 'Products' tab\n[2023-06-15 09:16:30] User viewed product 'Laptop X'\n[2023-06-15 09:18:45] User added 'Laptop X' to cart\n[2023-06-15 09:19:10] User proceeded to checkout\n[2023-06-15 09:25:37] User completed purchase\n42333 more tokens..."
        },
        {
            "user_id": "user456",
            "country": "Canada",
            "log": "[2023-06-15 14:30:12] User launched app\n[2023-06-15 14:31:03] User searched for 'wireless headphones'\n[2023-06-15 14:32:18] User applied price filter\n[2023-06-15 14:33:00] User viewed product 'Headphone Y'\n[2023-06-15 14:38:22] User exited app without purchase\n13238 more tokens..."
        }
    ]
    ```

Save this file as `user_logs.json` in your project directory.

## Creating the Pipeline

Now, let's create a docetl pipeline to analyze this data. We'll use a map-reduce-like approach:

1. Map each user log to key insights and supporting actions
2. Unnest the insights
3. Reduce by country to summarize insights and identify common patterns

Create a file named `pipeline.yaml` with the following structure:

!!! abstract "Pipeline Structure"

    1.  **Define the dataset**
        ```yaml
        datasets:
            user_logs:
            type: file
            path: "user_logs.json"
        ```

    2. **Extract insights** (map operation)
        ```yaml
        - name: extract_insights
            type: map
            prompt: |
            Analyze the following user interaction log:
            {{ input.log }}

            Extract 2-3 main insights from this log, each being 1-2 words, to help inform future product development. Consider any difficulties or pain points the user may have had. Also provide 1-2 supporting actions for each insight.
            Return the results as a list of dictionaries, each containing 'insight' and 'supporting_actions' keys.
            output:
            schema:
                insights: "list[{insight: string, supporting_actions: string}]"
        ```

    3. **Unnest insights** (unnest operation)
        ```yaml
        - name: unnest_insights
            type: unnest
            unnest_key: insights
            recursive: true
        ```

    4. **Summarize by country** (reduce operation)
        ```yaml
        - name: summarize_by_country
            type: reduce
            reduce_key: country
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

            Provide a summary of common insights and notable behaviors of users from this country.
            output:
            schema:
                detailed_summary: string
        ```

    5. **Define the pipeline steps**
        ```yaml
        pipeline:
            steps:
            - name: analyze_user_logs
                input: user_logs
                operations:
                - extract_insights
                - unnest_insights
                - summarize_by_country
        ```

    6. **Specify the output**
        ```yaml
        output:
            type: file
            path: "country_summaries.json"
        ```

??? example "Full Pipeline Configuration"

    ```yaml
    default_model: gpt-4o-mini

    datasets:
      user_logs:
        type: file
        path: "user_logs.json"

    operations:
      - name: extract_insights
        type: map
        prompt: |
          Analyze the following user interaction log:
          {{ input.log }}

          Extract 2-3 main insights from this log, each being 1-2 words, to help inform future product development. Consider any difficulties or pain points the user may have had. Also provide 1-2 supporting actions for each insight.
          Return the results as a list of dictionaries, each containing 'insight' and 'supporting_actions' keys.
        output:
          schema:
            insights: "list[{insight: string, supporting_actions: string}]"

      - name: unnest_insights
        type: unnest
        unnest_key: insights
        recursive: true

      - name: summarize_by_country
        type: reduce
        reduce_key: country
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

          Provide a summary of common insights and notable behaviors of users from this country.
        output:
          schema:
            detailed_summary: string

    pipeline:
      steps:
        - name: analyze_user_logs
          input: user_logs
          operations:
            - extract_insights
            - unnest_insights
            - summarize_by_country

      output:
        type: file
        path: "country_summaries.json"
    ```

## Running the Pipeline

To execute the pipeline, run the following command in your terminal:

```bash
docetl run pipeline.yaml
```

This will process the user logs, extract key insights and supporting actions, and generate summaries for each country, saving the results in `country_summaries.json`.

## Further Questions

??? question "What if I want to reduce by insights or an LLM-generated field?"

    You can modify the reduce operation to use any field as the reduce key, including LLM-generated fields from prior operations. Simply change the `reduce_key` in the `summarize_by_country` operation to the desired field. Note that we may need to perform entity resolution on the LLM-generated fields, which docetl can do for you in the optimization process (to be discussed later).

??? question "How do I know what pipeline configuration to write? Can't I do this all in one map operation?"

    While it's possible to perform complex operations in a single map step, breaking down the process into multiple steps often leads to more maintainable and flexible pipelines. To learn more about optimizing your pipeline configuration, read on to discover docetl's optimizer, which can be invoked using `docetl build` instead of `docetl run`.
