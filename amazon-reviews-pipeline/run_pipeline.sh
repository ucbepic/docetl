#!/bin/bash

# Amazon Reviews DocETL Pipeline Runner

echo "🚀 Starting Amazon Reviews Analysis Pipeline..."
echo "================================================"

# Set OpenAI API key
export OPENAI_API_KEY="sk-proj-16UaWXgl0AfEef29xu3BT3BlbkFJ3hFc8VSQ9IWRKxvolMeR"

# Check if data exists
if [ ! -f "data/amazon_reviews.json" ]; then
    echo "📊 Generating sample review data..."
    python3 collect_reviews.py
fi

# Install dependencies if needed
if ! python3 -c "import docetl" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip3 install -r requirements.txt
fi

# Run the DocETL pipeline
echo "🔄 Running DocETL pipeline..."
docetl run review_analysis_pipeline.yaml

# Check if output was generated
if [ -f "output/analyzed_reviews.json" ]; then
    echo "✅ Pipeline completed successfully!"
    echo ""
    echo "📈 Generating visualization..."
    python3 visualize_results.py
else
    echo "❌ Pipeline failed to generate output."
    echo "Please check the error messages above."
fi