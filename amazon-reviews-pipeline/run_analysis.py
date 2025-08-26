#!/usr/bin/env python3
"""
Amazon Product Reviews Analysis using DocETL-style processing
Demonstrates sentiment analysis, entity extraction, and insights generation
"""

import json
import os
from collections import Counter, defaultdict
from typing import List, Dict
import asyncio
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from rich import box

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use system environment variables

# Initialize console for pretty output
console = Console()

# Set up OpenAI client
# IMPORTANT: Set your API key as an environment variable for security
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    console.print("[red]Error: OPENAI_API_KEY environment variable not set![/red]")
    console.print("Please set it using: export OPENAI_API_KEY='your-api-key-here'")
    exit(1)

client = OpenAI(api_key=api_key)

def load_reviews(path: str, limit: int = None) -> List[Dict]:
    """Load Amazon reviews from JSON file"""
    with open(path, 'r') as f:
        reviews = json.load(f)
    return reviews[:limit] if limit else reviews

async def analyze_review(review: Dict) -> Dict:
    """Analyze a single review using OpenAI"""
    prompt = f"""Analyze this Amazon product review and extract structured information:

Review: {review['review_text']}
Category: {review['category']}
Rating: {review['rating']} stars

Extract and return as JSON:
1. sentiment: very_positive, positive, neutral, negative, or very_negative
2. features_mentioned: list of product features mentioned (max 5)
3. complaints: list of specific complaints (if any)
4. praise_points: list of specific praise points (if any)
5. purchase_intent: high, medium, low, or none
6. authenticity_score: 1-10 (10 being most authentic)
7. key_insights: 1-2 sentence summary of the review
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a product review analyst. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        # Add original review data
        result.update({
            "review_id": review["review_id"],
            "product_name": review["product_name"],
            "category": review["category"],
            "rating": review["rating"],
            "verified_purchase": review["verified_purchase"],
            "review_date": review["review_date"]
        })
        return result
    except Exception as e:
        console.print(f"[red]Error analyzing review {review['review_id']}: {e}[/red]")
        return None

async def batch_analyze_reviews(reviews: List[Dict], batch_size: int = 5) -> List[Dict]:
    """Analyze reviews in batches for efficiency"""
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Analyzing {len(reviews)} reviews...", total=len(reviews))
        
        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i + batch_size]
            batch_results = await asyncio.gather(*[analyze_review(r) for r in batch])
            results.extend([r for r in batch_results if r is not None])
            progress.update(task, advance=len(batch))
    
    return results

def generate_category_insights(analyzed_reviews: List[Dict]) -> Dict[str, Dict]:
    """Generate insights by product category"""
    category_data = defaultdict(list)
    
    for review in analyzed_reviews:
        category_data[review['category']].append(review)
    
    insights = {}
    for category, reviews in category_data.items():
        # Sentiment distribution
        sentiments = [r['sentiment'] for r in reviews]
        sentiment_counts = Counter(sentiments)
        
        # Common features
        all_features = []
        for r in reviews:
            all_features.extend(r.get('features_mentioned', []))
        top_features = Counter(all_features).most_common(5)
        
        # Common complaints and praise
        all_complaints = []
        all_praise = []
        for r in reviews:
            all_complaints.extend(r.get('complaints', []))
            all_praise.extend(r.get('praise_points', []))
        
        top_complaints = Counter(all_complaints).most_common(3)
        top_praise = Counter(all_praise).most_common(3)
        
        # Calculate average authenticity
        avg_authenticity = sum(r.get('authenticity_score', 5) for r in reviews) / len(reviews)
        
        insights[category] = {
            'total_reviews': len(reviews),
            'avg_rating': sum(r['rating'] for r in reviews) / len(reviews),
            'sentiment_distribution': dict(sentiment_counts),
            'top_features': [f[0] for f in top_features],
            'top_complaints': [c[0] for c in top_complaints],
            'top_praise': [p[0] for p in top_praise],
            'avg_authenticity': avg_authenticity,
            'verified_purchase_rate': sum(1 for r in reviews if r['verified_purchase']) / len(reviews)
        }
    
    return insights

def display_results(analyzed_reviews: List[Dict], category_insights: Dict[str, Dict]):
    """Display analysis results in a formatted way"""
    
    # Overall statistics
    console.print(Panel.fit(
        f"[bold cyan]Amazon Reviews Analysis Complete[/bold cyan]\n"
        f"Total Reviews Analyzed: {len(analyzed_reviews)}\n"
        f"Categories Covered: {len(category_insights)}",
        title="[bold]Analysis Summary[/bold]",
        border_style="cyan"
    ))
    
    # Sentiment distribution table
    sentiment_counts = Counter(r['sentiment'] for r in analyzed_reviews)
    
    table = Table(title="Overall Sentiment Distribution", box=box.ROUNDED)
    table.add_column("Sentiment", style="cyan")
    table.add_column("Count", justify="right", style="magenta")
    table.add_column("Percentage", justify="right", style="green")
    
    total = len(analyzed_reviews)
    for sentiment in ['very_positive', 'positive', 'neutral', 'negative', 'very_negative']:
        count = sentiment_counts.get(sentiment, 0)
        percentage = (count / total * 100) if total > 0 else 0
        table.add_row(
            sentiment.replace('_', ' ').title(),
            str(count),
            f"{percentage:.1f}%"
        )
    
    console.print("\n")
    console.print(table)
    
    # Category insights
    console.print("\n[bold]Category-Specific Insights:[/bold]\n")
    
    for category, insights in sorted(category_insights.items(), 
                                   key=lambda x: x[1]['avg_rating'], 
                                   reverse=True)[:5]:  # Top 5 categories
        
        panel_content = [
            f"[cyan]Average Rating:[/cyan] {insights['avg_rating']:.1f}/5.0",
            f"[cyan]Total Reviews:[/cyan] {insights['total_reviews']}",
            f"[cyan]Verified Purchases:[/cyan] {insights['verified_purchase_rate']*100:.0f}%",
            "",
            f"[green]Top Features:[/green]"
        ]
        
        for i, feature in enumerate(insights['top_features'][:3], 1):
            panel_content.append(f"  {i}. {feature}")
        
        if insights['top_complaints']:
            panel_content.append("")
            panel_content.append("[red]Common Issues:[/red]")
            for i, complaint in enumerate(insights['top_complaints'][:2], 1):
                panel_content.append(f"  {i}. {complaint}")
        
        console.print(Panel(
            "\n".join(panel_content),
            title=f"[bold]{category}[/bold]",
            border_style="blue",
            padding=(1, 2)
        ))
        console.print()

def save_results(analyzed_reviews: List[Dict], category_insights: Dict[str, Dict]):
    """Save analysis results to files"""
    # Save detailed review analysis
    with open("output/analyzed_reviews.json", "w") as f:
        json.dump(analyzed_reviews, f, indent=2)
    
    # Save category insights
    with open("output/category_insights.json", "w") as f:
        json.dump(category_insights, f, indent=2)
    
    # Save summary report
    summary = {
        "total_reviews": len(analyzed_reviews),
        "categories_analyzed": len(category_insights),
        "overall_sentiment_distribution": dict(Counter(r['sentiment'] for r in analyzed_reviews)),
        "avg_authenticity_score": sum(r.get('authenticity_score', 5) for r in analyzed_reviews) / len(analyzed_reviews),
        "top_rated_categories": sorted(
            [(cat, data['avg_rating']) for cat, data in category_insights.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
    }
    
    with open("output/summary_report.json", "w") as f:
        json.dump(summary, f, indent=2)

async def main():
    """Main pipeline execution"""
    console.print("[bold cyan]Starting Amazon Reviews Analysis Pipeline[/bold cyan]\n")
    
    # Load reviews
    console.print("📊 Loading reviews data...")
    # Try to load real reviews first, fallback to synthetic if not available
    try:
        reviews = load_reviews("data/real_reviews.json", limit=50)  # Limit for demo
        console.print(f"✓ Loaded {len(reviews)} REAL reviews\n")
    except FileNotFoundError:
        reviews = load_reviews("data/amazon_reviews.json", limit=50)  # Fallback
        console.print(f"✓ Loaded {len(reviews)} reviews\n")
    
    # Analyze reviews
    console.print("🤖 Analyzing reviews with AI...")
    analyzed_reviews = await batch_analyze_reviews(reviews)
    console.print(f"✓ Successfully analyzed {len(analyzed_reviews)} reviews\n")
    
    # Generate insights
    console.print("💡 Generating category insights...")
    category_insights = generate_category_insights(analyzed_reviews)
    console.print("✓ Insights generated\n")
    
    # Display results
    display_results(analyzed_reviews, category_insights)
    
    # Save results
    console.print("\n💾 Saving results...")
    save_results(analyzed_reviews, category_insights)
    console.print("✓ Results saved to output directory")
    
    console.print("\n[bold green]Pipeline completed successfully![/bold green]")
    console.print("\nOutput files:")
    console.print("  - output/analyzed_reviews.json (detailed analysis)")
    console.print("  - output/category_insights.json (category summaries)")
    console.print("  - output/summary_report.json (overall summary)")

if __name__ == "__main__":
    asyncio.run(main())