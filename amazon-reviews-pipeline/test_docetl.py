import os
import json
from docetl import Dataset, MapOp, ReduceOp, Pipeline
import litellm

# Set OpenAI API key
# IMPORTANT: Set your API key as an environment variable for security
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set!")
    print("Please set it using: export OPENAI_API_KEY='your-api-key-here'")
    exit(1)

# Load the data
with open("data/amazon_reviews.json", "r") as f:
    reviews_data = json.load(f)

# Create a dataset
dataset = Dataset(
    type="file",
    path="data/amazon_reviews.json",
    parsing=[{"name": "json_parser", "type": "json"}]
)

# Define operations
sentiment_op = MapOp(
    name="extract_sentiment",
    type="map",
    prompt="""Analyze this product review and extract sentiment information:

Review: {{ input.review_text }}
Rating: {{ input.rating }} stars

Return a JSON with:
- sentiment: positive, neutral, or negative
- key_points: list of main points from the review (max 3)
- would_recommend: yes, no, or unclear
""",
    output_schema={
        "sentiment": "string",
        "key_points": "list[string]", 
        "would_recommend": "string"
    },
    model="gpt-4o-mini"
)

# Create and run a simple pipeline
print("Running DocETL pipeline on Amazon reviews...")
print(f"Processing {len(reviews_data[:10])} reviews as a test...")

# Process first 10 reviews
results = []
for review in reviews_data[:10]:
    try:
        result = sentiment_op.execute([review])
        if result:
            results.extend(result)
            print(f"✓ Processed review {review['review_id']}")
    except Exception as e:
        print(f"✗ Error processing review {review['review_id']}: {e}")

# Save results
output_path = "output/test_results.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_path}")
print(f"Total processed: {len(results)} reviews")

# Show summary
if results:
    sentiments = [r.get("sentiment", "unknown") for r in results]
    print("\nSentiment Distribution:")
    for sentiment in ["positive", "neutral", "negative"]:
        count = sentiments.count(sentiment)
        print(f"  {sentiment}: {count}")