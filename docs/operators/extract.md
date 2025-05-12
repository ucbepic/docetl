# Extract Operation

The Extract operation in DocETL identifies and extracts specific sections of text from documents based on provided criteria. It's particularly useful for isolating relevant content from larger documents for further processing or analysis.

## 🚀 Example: Extracting Key Findings from Research Reports

Let's see a practical example of using the Extract operation to pull out key findings from research reports:

```yaml
- name: findings
  type: extract
  prompt: |
    Extract all sections that discuss key findings, results, or conclusions from this research report.
    Focus on paragraphs that:
    - Summarize experimental outcomes
    - Present statistical results
    - Describe discovered insights
    - State conclusions drawn from the research
    
    Only extract the most important and substantive findings.
  document_keys: ["report_text"]
  model: "gpt-4.1-mini"
```

This Extract operation processes each document to identify and extract key findings by:

1. Converting the text into a format with line numbers for precise extraction
2. Using an LLM to identify the line ranges containing key findings
3. Extracting the text from those ranges and adding it to the document with the suffix "_extracted_findings"

??? example "Sample Input and Output"

    Input:
    ```json
    [
      {
        "report_id": "R-2023-001",
        "report_text": "EXPERIMENTAL METHODS\n\nThe study utilized a mixed-methods approach combining quantitative surveys (n=230) and qualitative interviews (n=42) with participants from diverse demographic backgrounds. Data collection occurred between January and March 2023.\n\nRESULTS\n\nThe analysis revealed three primary patterns of user engagement. First, 68% of participants reported daily interaction with the platform, significantly higher than previous industry benchmarks (p<0.01). Second, user retention showed strong correlation with personalization features (r=0.72). Finally, demographic factors such as age and technical proficiency were not significant predictors of engagement, contradicting prior research in this domain.\n\nDISCUSSION\n\nThese findings suggest that platform design priorities should emphasize personalization capabilities over demographic targeting. The high daily engagement rates indicate market readiness for similar applications, while the lack of demographic effects points to broad accessibility across user segments.\n\nLIMITATIONS\n\nThe study was limited by its focus on early adopters, which may not represent the broader potential user base. Additionally, the three-month timeframe may not capture seasonal variations in user behavior."
      }
    ]
    ```

    Output:
    ```json
    [
      {
        "report_id": "R-2023-001",
        "report_text": "EXPERIMENTAL METHODS\n\nThe study utilized a mixed-methods approach combining quantitative surveys (n=230) and qualitative interviews (n=42) with participants from diverse demographic backgrounds. Data collection occurred between January and March 2023.\n\nRESULTS\n\nThe analysis revealed three primary patterns of user engagement. First, 68% of participants reported daily interaction with the platform, significantly higher than previous industry benchmarks (p<0.01). Second, user retention showed strong correlation with personalization features (r=0.72). Finally, demographic factors such as age and technical proficiency were not significant predictors of engagement, contradicting prior research in this domain.\n\nDISCUSSION\n\nThese findings suggest that platform design priorities should emphasize personalization capabilities over demographic targeting. The high daily engagement rates indicate market readiness for similar applications, while the lack of demographic effects points to broad accessibility across user segments.\n\nLIMITATIONS\n\nThe study was limited by its focus on early adopters, which may not represent the broader potential user base. Additionally, the three-month timeframe may not capture seasonal variations in user behavior.",
        "report_text_extracted_findings": "The analysis revealed three primary patterns of user engagement. First, 68% of participants reported daily interaction with the platform, significantly higher than previous industry benchmarks (p<0.01). Second, user retention showed strong correlation with personalization features (r=0.72). Finally, demographic factors such as age and technical proficiency were not significant predictors of engagement, contradicting prior research in this domain.\n\nThese findings suggest that platform design priorities should emphasize personalization capabilities over demographic targeting. The high daily engagement rates indicate market readiness for similar applications, while the lack of demographic effects points to broad accessibility across user segments."
      }
    ]
    ```

This example demonstrates how the Extract operation can identify and isolate specific content based on semantic understanding, providing a focused subset of the original text for further analysis.

## Output Formats

The Extract operation offers two different output formats controlled by the `format_extraction` parameter:

### String Format (`format_extraction: true`)

When `format_extraction` is set to `true` (the default), all extracted text segments are joined together with newlines into a single string:

```yaml
- name: findings
  type: extract
  prompt: "Extract the key findings from this research report."
  document_keys: ["report_text"]
  format_extraction: true  # This is the default
```

With this setting, the output looks like:

```json
{
  "report_id": "R-2023-001",
  "report_text": "... original text ...",
  "report_text_extracted_findings": "Finding 1 about daily engagement rates...\n\nFinding 2 about personalization features..."
}
```

This format is ideal for:
- Human readability
- Further LLM processing of the extracted content
- When the extractions are logically related and should be treated as a unit
- Simpler downstream text processing

### List Format (`format_extraction: false`)

When `format_extraction` is set to `false`, each extracted text segment is kept separate in a list:

```yaml
- name: findings
  type: extract
  prompt: "Extract the key findings from this research report."
  document_keys: ["report_text"]
  format_extraction: false
```

With this setting, the output looks like:

```json
{
  "report_id": "R-2023-001",
  "report_text": "... original text ...",
  "report_text_extracted_findings": [
    "Finding 1 about daily engagement rates...",
    "Finding 2 about personalization features..."
  ]
}
```

This format is particularly useful when:
- You need to process each extraction individually
- The extractions represent distinct items that should be handled separately
- You plan to filter, transform, or analyze individual extractions
- You need to count or quantify how many distinct items were extracted
- The extractions will be used in data processing pipelines

For example, if you're extracting product features from reviews, setting `format_extraction: false` allows you to easily count unique features, apply sentiment analysis to each feature mention separately, or create structured data from each extraction.

## Algorithm and Implementation

The Extract operation has two main extraction strategies:

### Line Number Strategy

1. **Text Reformatting**:
   - The input text is reformatted with line numbers added as prefixes
   - Lines are wrapped to a specified width (default 80 characters) for consistent formatting

2. **LLM Extraction**:
   - The LLM is provided with the formatted text and extraction instructions
   - The LLM identifies specific line ranges (start_line, end_line) containing relevant content
   
3. **Content Extraction**:
   - The system extracts the specified line ranges from the original text
   - Line number prefixes are removed from the extracted content
   - Duplicate extractions are eliminated 

### Regex Strategy

1. **Pattern Generation**:
   - The LLM is provided with the text and extraction instructions
   - The LLM generates regex patterns designed to match the relevant content

2. **Pattern Application**:
   - The system applies the generated regex patterns to the original text
   - Matches are collected and duplicates are removed
   
3. **Output Preparation**:
   - Extracted content is either formatted as a string (with newlines between extractions) or as a list of strings

## Required Parameters

- `name`: A unique name for the operation.
- `type`: Must be set to "extract".
- `prompt`: The prompt specifying what content to extract.
- `document_keys`: List of document field keys containing text to process.

## Optional Parameters

| Parameter | Description | Default |
| --------- | ----------- | ------- |
| `model` | The language model to use for extraction | Falls back to `default_model` |
| `extraction_method` | Method to use for extraction: "line_number" or "regex" | "line_number" |
| `format_extraction` | Whether to join extractions with newlines (`true`) or keep as a list of strings (`false`) | `true` |
| `extraction_key_suffix` | Suffix to add to the original key for storing extractions | "_extracted_{name}" |
| `timeout` | Timeout for each LLM call in seconds | 120 |
| `skip_on_error` | Whether to continue processing if an error occurs | false |
| `litellm_completion_kwargs` | Additional parameters to pass to LiteLLM completion calls | {} |

## Best Practices

1. **Craft Specific Extraction Prompts**: Be detailed about exactly what content should be extracted, including context clues and exclusion criteria.

2. **Choose the Right Extraction Method**:
   - Use `line_number` for structural extraction where content spans multiple lines or paragraphs
   - Use `regex` for pattern-based extraction like dates, identifiers, or specific formatted data

3. **Process Appropriate Document Keys**: Only include document fields that contain text relevant to your extraction needs.

4. **Consider Error Handling**: Enable `skip_on_error` for large batch processing where individual failures shouldn't stop the entire operation.

5. **Format Based on Downstream Needs**:
   - Use `format_extraction: true` (default) when you want the extracted content as a single, readable text block
     - Good for: human review, passing to another LLM, treating extractions as one unit
     - Example: extracting a summary or conclusion that should be read as a whole
   - Use `format_extraction: false` when you need programmatic access to each extracted segment
     - Good for: counting extractions, individual processing, structured data creation
     - Example: extracting individual claims, facts, or data points that need separate treatment
     - Example: extracting product features that will be analyzed individually

6. **Customize Output Keys**: Use `extraction_key_suffix` to create intuitive field names that indicate what was extracted.

## Use Cases

The Extract operation is particularly valuable for:

1. **Document Summarization**: Extracting executive summaries, conclusions, or key findings
2. **Data Mining**: Pulling structured data like dates, measurements, or specific values from unstructured text
3. **Content Filtering**: Isolating relevant sections of long documents for further analysis
4. **Evidence Collection**: Gathering supporting statements or quotes related to specific topics
5. **Preprocessing**: Creating focused inputs for downstream LLM operations 