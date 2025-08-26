# Amazon Reviews DocETL Pipeline Demo

## Overview

This demo showcases a sophisticated data extraction, transformation, and loading (ETL) pipeline for analyzing Amazon product reviews using AI-powered natural language processing. The pipeline processes unstructured text data from 200 synthetic Amazon reviews across 10 product categories.

## Key Features Demonstrated

### 1. **Data Collection & Generation**
- Generated 200 realistic Amazon product reviews across 10 categories
- Includes metadata: ratings, verified purchases, dates, product names
- Balanced distribution: 60% positive, 25% negative, 15% neutral reviews

### 2. **AI-Powered Analysis**
Using OpenAI's GPT-4o-mini model, the pipeline extracts:
- **Sentiment Analysis**: 5-level sentiment classification (very positive to very negative)
- **Feature Extraction**: Key product features mentioned in reviews
- **Complaint Categorization**: Systematic categorization of issues
- **Authenticity Scoring**: 1-10 scale rating for review authenticity
- **Purchase Intent**: Likelihood of repeat purchase/recommendation

### 3. **Insights Generation**
- **Category-Level Analysis**: Aggregated insights by product category
- **Common Patterns**: Top features, complaints, and praise points
- **Strategic Recommendations**: Business-actionable insights

## Results Summary

From analyzing 50 reviews (demo subset):

### Sentiment Distribution
- **Very Positive**: 58.0% (29 reviews)
- **Neutral**: 16.0% (8 reviews)  
- **Negative**: 10.0% (5 reviews)
- **Very Negative**: 16.0% (8 reviews)

### Top Performing Categories
1. **Sports & Outdoors**: 4.2/5.0 average rating
2. **Beauty**: 4.0/5.0 average rating
3. **Automotive**: 4.0/5.0 average rating

### Key Insights
- **Average Authenticity Score**: 8.5/10 (indicating high-quality reviews)
- **Verified Purchase Rate**: 74% (good credibility indicator)
- **Common Positive Features**: customer service, value, quality
- **Common Issues**: quality concerns, shipping problems, expectations mismatch

## Technical Implementation

### Architecture
```
Input Data → AI Analysis → Aggregation → Insights → Visualization
    ↓             ↓            ↓           ↓            ↓
JSON Files   OpenAI API   Category    Strategic    Rich Console
             GPT-4o-mini  Grouping    Analysis     Output
```

### Files Structure
```
amazon-reviews-pipeline/
├── data/
│   └── amazon_reviews.json         # 200 sample reviews
├── output/
│   ├── analyzed_reviews.json       # Detailed AI analysis
│   ├── category_insights.json      # Category aggregations
│   └── summary_report.json         # High-level summary
├── collect_reviews.py              # Data generation script
├── run_analysis.py                 # Main pipeline execution
├── visualize_results.py            # Results visualization
└── working_pipeline.yaml           # DocETL configuration
```

### Key Technologies
- **DocETL**: Pipeline orchestration and configuration
- **OpenAI API**: Natural language processing and analysis
- **Python**: Data processing and orchestration
- **Rich**: Beautiful terminal output and visualization

## Business Value

This pipeline demonstrates how businesses can:

1. **Scale Review Analysis**: Process thousands of reviews automatically
2. **Extract Actionable Insights**: Identify product improvements and market opportunities
3. **Monitor Quality**: Detect authenticity issues and fake reviews
4. **Understand Customers**: Deep insights into customer preferences and pain points
5. **Track Trends**: Monitor sentiment and feature requests over time

## Running the Demo

```bash
# Quick start
./run_pipeline.sh

# Or manually
python3 collect_reviews.py          # Generate sample data
python3 run_analysis.py             # Run AI analysis
python3 visualize_results.py        # Display results
```

## Cost Efficiency

- Uses GPT-4o-mini for cost-effective processing
- Batch processing to minimize API calls
- Processes 50 reviews for ~$0.05 in API costs
- Scalable to thousands of reviews with minimal cost increase

## Future Enhancements

1. **Real-time Processing**: Stream reviews as they arrive
2. **Multi-language Support**: Analyze reviews in different languages
3. **Competitive Analysis**: Compare with competitor products
4. **Trend Detection**: Identify emerging issues or features
5. **Integration**: Connect with e-commerce platforms directly

This demo showcases the power of combining unstructured text processing with AI to generate valuable business insights at scale.