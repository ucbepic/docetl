# Amazon Product Reviews Analysis Pipeline

A sophisticated DocETL pipeline for analyzing Amazon product reviews using AI-powered natural language processing. This pipeline extracts valuable insights from customer feedback to help businesses understand sentiment patterns, common issues, and market opportunities.

## 🚀 Features

### Data Processing
- **Sentiment Analysis**: Classifies reviews into very positive, positive, neutral, negative, and very negative sentiments
- **Entity Extraction**: Identifies product features, complaints, and praise points mentioned in reviews
- **Issue Categorization**: Automatically categorizes complaints into standard issue types (quality, shipping, performance, etc.)
- **Review Authenticity**: Analyzes reviews for potential authenticity indicators with scoring

### Insights Generation
- **Category-Level Analysis**: Aggregates insights by product category to identify patterns
- **Cross-Category Insights**: Discovers common issues and best practices across all categories
- **Strategic Recommendations**: Generates actionable business insights and market opportunities
- **Review Summarization**: Creates concise summaries of each review for quick scanning

## 📁 Project Structure

```
amazon-reviews-pipeline/
├── data/
│   └── amazon_reviews.json      # Sample review data (200 documents)
├── output/
│   ├── analyzed_reviews.json    # Individual review analysis results
│   ├── category_insights.json   # Category-level aggregated insights
│   └── strategic_insights.json  # Cross-category strategic analysis
├── collect_reviews.py           # Script to generate sample review data
├── review_analysis_pipeline.yaml # DocETL pipeline configuration
├── visualize_results.py         # Results visualization script
├── run_pipeline.sh             # Pipeline execution script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🛠️ Setup & Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd /workspace/amazon-reviews-pipeline
   ```

2. **Install dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```

3. **Set up OpenAI API key** (already included in run_pipeline.sh):
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## 🏃 Running the Pipeline

### Quick Start
Simply run the provided shell script:
```bash
./run_pipeline.sh
```

This script will:
1. Generate sample review data if it doesn't exist
2. Install dependencies if needed
3. Run the DocETL pipeline
4. Display visualization of results

### Manual Execution
If you prefer to run steps individually:

1. **Generate sample data**:
   ```bash
   python3 collect_reviews.py
   ```

2. **Run the DocETL pipeline**:
   ```bash
   docetl run review_analysis_pipeline.yaml
   ```

3. **Visualize results**:
   ```bash
   python3 visualize_results.py
   ```

## 📊 Pipeline Operations

### 1. Extract Sentiment & Entities
- Analyzes each review to extract sentiment and key information
- Identifies product features, complaints, and praise points
- Determines purchase intent and price/value sentiment

### 2. Categorize Issues
- Maps complaints to standard issue categories
- Helps identify systemic problems across products

### 3. Create Review Summaries
- Generates concise 1-2 sentence summaries
- Useful for quick review scanning

### 4. Identify Fake Review Indicators
- Scores reviews on authenticity (1-10 scale)
- Identifies suspicious elements and authenticity indicators

### 5. Extract Review Themes
- Aggregates insights by product category
- Identifies top features, common issues, and patterns

### 6. Generate Cross-Category Insights
- Analyzes patterns across all categories
- Provides strategic business recommendations

## 📈 Output Analysis

The pipeline generates three main output files:

1. **analyzed_reviews.json**: Detailed analysis of each individual review
2. **category_insights.json**: Aggregated insights for each product category
3. **strategic_insights.json**: High-level business insights and recommendations

The visualization script presents:
- Sentiment distribution charts
- Issue category breakdowns
- Authenticity analysis
- Category-specific insights panels
- Strategic business recommendations

## 🎯 Use Cases

- **E-commerce Businesses**: Understand customer satisfaction and improve products
- **Market Research**: Identify trends and opportunities in product categories
- **Quality Assurance**: Detect common product issues and quality problems
- **Customer Service**: Prioritize support based on common complaints
- **Product Development**: Guide feature development based on customer feedback
- **Competitive Analysis**: Understand market standards and customer expectations

## 🔧 Customization

### Modifying the Pipeline
Edit `review_analysis_pipeline.yaml` to:
- Add new operations
- Adjust prompts for different insights
- Change batch sizes for optimization
- Modify output schemas

### Adding New Data Sources
Modify `collect_reviews.py` to:
- Connect to real review APIs
- Import from different file formats
- Add more product categories
- Increase sample size

### Enhancing Visualizations
Update `visualize_results.py` to:
- Add new chart types
- Export results to different formats
- Create interactive dashboards
- Generate PDF reports

## 💡 Tips for Best Results

1. **API Usage**: The pipeline uses OpenAI's GPT-4o-mini model for cost-effectiveness
2. **Batch Processing**: Operations are batched to reduce API calls and costs
3. **Data Quality**: Better quality input data leads to more accurate insights
4. **Scalability**: Can handle thousands of reviews with appropriate API limits

## 🤝 Contributing

Feel free to extend this pipeline with:
- Additional analysis operations
- New visualization options
- Integration with real review sources
- Export capabilities for business intelligence tools

## 📝 License

This is a demonstration project showcasing DocETL capabilities. Feel free to adapt and use for your own purposes.