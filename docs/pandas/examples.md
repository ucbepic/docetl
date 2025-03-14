# Examples

Here are some demonstrating how to use DocETL's pandas integration for various tasks.

Note that caching is enabled, but intermediate outputs are not persisted like in DocETL's YAML interface.

## Example 1: Analyzing Customer Reviews

Extract structured insights from customer reviews:

```python
import pandas as pd
from docetl import SemanticAccessor

# Load customer reviews
df = pd.DataFrame({
    "review": [
        "Great laptop, fast processor but battery life could be better",
        "The camera quality is amazing, especially in low light",
        "Keyboard feels cheap and the screen is too dim"
    ]
})

# Configure semantic accessor
df.semantic.set_config(default_model="gpt-4o-mini")

# Extract structured insights
result = df.semantic.map(
    prompt="""Analyze this product review and extract:
    1. Mentioned features
    2. Sentiment per feature
    3. Overall sentiment
    
    Review: {{input.review}}""",
    output_schema={
        "features": "list[str]",
        "feature_sentiments": "dict[str, str]",
        "overall_sentiment": "str"
    }
)

# Filter for negative reviews
negative_reviews = result.semantic.filter(
    prompt="Is this review predominantly negative? Consider the overall sentiment and feature sentiments.\n{{input}}"
)
```

## Example 2: Deduplicating Customer Records

Identify and merge duplicate customer records using fuzzy matching:

```python
# Customer records from two sources
df1 = pd.DataFrame({
    "name": ["John Smith", "Mary Johnson"],
    "email": ["john@email.com", "mary.j@email.com"],
    "address": ["123 Main St", "456 Oak Ave"]
})

df2 = pd.DataFrame({
    "name": ["John A Smith", "Mary Johnson"],
    "email": ["john@email.com", "mary.johnson@email.com"],
    "address": ["123 Main Street", "456 Oak Avenue"]
})

# Merge records with fuzzy matching
merged = df1.semantic.merge(
    df2,
    comparison_prompt="""Compare these customer records and determine if they represent the same person.
    Consider name variations, email patterns, and address formatting.
    
    Record 1:
    Name: {{input1.name}}
    Email: {{input1.email}}
    Address: {{input1.address}}
    
    Record 2:
    Name: {{input2.name}}
    Email: {{input2.email}}
    Address: {{input2.address}}""",
    fuzzy=True, # This will automatically invoke optimization
)
```

## Example 3: Topic Analysis of News Articles

Group and summarize news articles by topic:

```python
# News articles
df = pd.DataFrame({
    "title": ["Apple's New iPhone Launch", "Tech Giants Face Regulation", "AI Advances in Healthcare"],
    "content": ["Apple announced...", "Lawmakers propose...", "Researchers develop..."]
})

# First, use a semantic map to extract the topic from each article
df = df.semantic.map(
    prompt="Extract the topic from this article: {{input.content}}",
    output_schema={"topic": "str"}
)

# Group similar articles and generate summaries
summaries = df.semantic.agg(
    # First, group similar articles
    fuzzy=True,
    reduce_keys=["topic"],
    comparison_prompt="""Are these articles about the same topic or closely related topics?
    
    Article 1:
    Title: {{input1.title}}
    Content: {{input1.content}}
    
    Article 2:
    Title: {{input2.title}}
    Content: {{input2.content}}""",
    
    # Then, generate a summary for each group
    reduce_prompt="""Summarize these related articles into a comprehensive overview:
    
    Articles:
    {{inputs}}""",
    
    output_schema={
        "summary": "str",
        "key_points": "list[str]"
    }
)

# Summaries will be a df with the following columns:
# - topic: str (because this was the reduce_keys)
# - summary: str
# - key_points: list[str]
```

## Example 4: Multi-step Analysis Pipeline

Combine multiple operations for complex analysis:

```python
# Social media posts
posts = pd.DataFrame({
    "text": ["Just tried the new iPhone 15!", "Having issues with iOS 17", "Android is better"],
    "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"]
})

# 1. Extract structured information
analyzed = posts.semantic.map(
    prompt="""Analyze this social media post and extract:
    1. Product mentioned
    2. Sentiment
    3. Issues/Praise points
    
    Post: {{input.text}}""",
    output_schema={
        "product": "str",
        "sentiment": "str",
        "points": "list[str]"
    }
)

# 2. Filter relevant posts
relevant = analyzed.semantic.filter(
    prompt="Is this post about Apple products? {{input}}"
)


# 3. Group by issue and summarize
summaries = relevant.semantic.agg(
    fuzzy=True,
    reduce_keys=["product"],
    comparison_prompt="Do these posts discuss the same product?",
    reduce_prompt="Summarize the feedback about this product",
    output_schema={
        "summary": "str",
        "frequency": "int",
        "severity": "str"
    }
)

# Summaries will be a df with the following columns:
# - product: str (because this was the reduce_keys)
# - summary: str
# - frequency: int
# - severity: str

# Track total cost
print(f"Total analysis cost: ${summaries.semantic.total_cost}")
```

## Example 5: Error Handling and Validation

Implement robust error handling and validation:

```python
# Product descriptions
df = pd.DataFrame({
    "description": ["High-performance laptop...", "Wireless earbuds...", "Invalid/"]
})

try:
    result = df.semantic.map(
        prompt="Extract product specifications from: {{input.description}}. There should be at least one feature.",
        output_schema={
            "category": "str",
            "features": "list[str]",
            "price_range": "enum[budget, mid-range, premium, luxury]"
        },
        # Validation rules
        validate=[
            "len(output['features']) >= 1",
        ],
        # Retry configuration
        num_retries_on_validate_failure=2,
    )
    
    # Check operation history
    for op in result.semantic.history:
        print(f"Operation: {op.op_type}")
        print(f"Modified columns: {op.output_columns}")
        
except Exception as e:
    print(f"Error during processing: {e}")
``` 

## Example 6: PDF Analysis

DocETL supports native PDF handling with Claude and Gemini, in map and filter operations.
Suppose you have a column in your pandas dataframe with PDF paths (1 path per row), and you want the LLM to do some analysis for each PDF. You can do this by setting the `pdf_url_key` parameter in the map or filter operation.

```python
df = pd.DataFrame({
    "PdfPath": ["https://docetl.blob.core.windows.net/ntsb-reports/Report_N617GC.pdf", "https://docetl.blob.core.windows.net/ntsb-reports/Report_CEN25LA075.pdf"]
})

result_df = df.semantic.map(
    prompt="Summarize the air crash report and determine any contributing factors",
    output_schema={"summary": "str", "contributing_factors": "list[str]"},
    pdf_url_key="PdfPath", # This is the column with the PDF paths
)

print(result_df.head()) # The result will have the same number of rows as the input dataframe, with the summary and contributing factors added
```

## Example 7: Synthetic Data Generation

DocETL supports generating multiple outputs for each input using the `n` parameter in the map operation. This is useful for synthetic data generation, A/B testing content variations, or creating multiple alternatives for each input.

```python
# Starter concepts for content generation
df = pd.DataFrame({
    "product": ["Smart Watch", "Wireless Earbuds", "Home Security Camera"],
    "target_audience": ["Fitness Enthusiasts", "Music Lovers", "Homeowners"]
})

# Generate 5 marketing headlines for each product-audience combination
variations = df.semantic.map(
    prompt="""Generate a compelling marketing headline for the following product and target audience:
    
    Product: {{input.product}}
    Target Audience: {{input.target_audience}}
    
    The headline should:
    - Be attention-grabbing and memorable
    - Speak directly to the target audience's needs or desires
    - Highlight the key benefit of the product
    - Be between 5-10 words
    """,
    output_schema={"headline": "str"},
    n=5  # Generate 5 variations for each input row
)

print(f"Original dataframe rows: {len(df)}")
print(f"Generated variations: {len(variations)}")  # Should be 5x the original count

# Variations dataframe will contain:
# - All original columns (product, target_audience)
# - The new headline column
# - 5 rows for each original row (15 total for this example)

# You can also combine this with other operations
# Filter to only keep the best headlines
best_headlines = variations.semantic.filter(
    prompt="""Is this headline exceptional and likely to drive high engagement?
    
    Product: {{input.product}}
    Target Audience: {{input.target_audience}}
    Headline: {{input.headline}}
    
    Consider catchiness, emotional appeal, and clarity.
    """
)
```

This example will generate 15 total variations (3 products × 5 variations each). You can adjust the `n` parameter to generate more or fewer variations as needed.