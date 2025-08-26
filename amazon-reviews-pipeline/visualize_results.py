import json
import os
from collections import Counter, defaultdict
from typing import Dict, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich import box
from rich.text import Text

console = Console()

def load_json_file(filepath: str) -> List[Dict]:
    """Load JSON data from file."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return []

def create_sentiment_distribution_table(reviews: List[Dict]) -> Table:
    """Create a table showing sentiment distribution."""
    sentiment_counts = Counter(review.get('sentiment', 'unknown') for review in reviews)
    
    table = Table(title="Sentiment Distribution", box=box.ROUNDED)
    table.add_column("Sentiment", style="cyan")
    table.add_column("Count", justify="right", style="magenta")
    table.add_column("Percentage", justify="right", style="green")
    
    total = sum(sentiment_counts.values())
    for sentiment in ['very_positive', 'positive', 'neutral', 'negative', 'very_negative']:
        count = sentiment_counts.get(sentiment, 0)
        percentage = (count / total * 100) if total > 0 else 0
        table.add_row(sentiment.replace('_', ' ').title(), str(count), f"{percentage:.1f}%")
    
    return table

def create_category_insights_panels(category_insights: Dict[str, Dict]) -> List[Panel]:
    """Create panels for each category's insights."""
    panels = []
    
    for category, category_data in category_insights.items():
        
        content = []
        
        # Top features
        features = category_data.get('top_features', [])[:3]
        if features:
            content.append("[bold cyan]Top Features:[/bold cyan]")
            for i, feature in enumerate(features, 1):
                content.append(f"  {i}. {feature}")
        
        # Common complaints
        complaints = category_data.get('common_complaints', [])[:3]
        if complaints:
            content.append("\n[bold red]Common Issues:[/bold red]")
            for i, complaint in enumerate(complaints, 1):
                content.append(f"  {i}. {complaint}")
        
        # Key insights
        insights = category_data.get('key_insights', [])[:2]
        if insights:
            content.append("\n[bold yellow]Key Insights:[/bold yellow]")
            for insight in insights:
                content.append(f"  • {insight}")
        
        panel = Panel(
            "\n".join(content),
            title=f"[bold]{category}[/bold]",
            border_style="blue",
            padding=(1, 2)
        )
        panels.append(panel)
    
    return panels

def display_authenticity_analysis(reviews: List[Dict]):
    """Display authenticity analysis results."""
    authentic_scores = [r.get('authenticity_score', 0) for r in reviews if 'authenticity_score' in r]
    
    if not authentic_scores:
        return
    
    avg_score = sum(authentic_scores) / len(authentic_scores)
    suspicious_reviews = [r for r in reviews if r.get('authenticity_score', 10) < 5]
    
    table = Table(title="Review Authenticity Analysis", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="magenta")
    
    table.add_row("Average Authenticity Score", f"{avg_score:.1f}/10")
    table.add_row("Potentially Suspicious Reviews", str(len(suspicious_reviews)))
    table.add_row("Verified Purchase Percentage", 
                  f"{sum(1 for r in reviews if r.get('verified_purchase', False)) / len(reviews) * 100:.1f}%")
    
    console.print(table)

def display_strategic_insights(strategic_data: Dict):
    """Display strategic insights from cross-category analysis."""
    if not strategic_data:
        return
    
    panel_content = []
    
    # Best practices
    practices = strategic_data.get('best_practices', [])[:3]
    if practices:
        panel_content.append("[bold green]Best Practices:[/bold green]")
        for practice in practices:
            panel_content.append(f"  ✓ {practice}")
    
    # Market opportunities
    opportunities = strategic_data.get('market_opportunities', [])[:3]
    if opportunities:
        panel_content.append("\n[bold yellow]Market Opportunities:[/bold yellow]")
        for opp in opportunities:
            panel_content.append(f"  💡 {opp}")
    
    # Strategic recommendations
    recommendations = strategic_data.get('strategic_recommendations', [])[:3]
    if recommendations:
        panel_content.append("\n[bold cyan]Strategic Recommendations:[/bold cyan]")
        for rec in recommendations:
            panel_content.append(f"  → {rec}")
    
    console.print(Panel(
        "\n".join(panel_content),
        title="[bold]Strategic Business Insights[/bold]",
        border_style="green",
        padding=(1, 2)
    ))

def create_issue_category_chart(reviews: List[Dict]):
    """Create a visual representation of issue categories."""
    issue_counts = Counter()
    
    for review in reviews:
        categories = review.get('issue_categories', [])
        issue_counts.update(categories)
    
    if not issue_counts:
        return
    
    table = Table(title="Issue Category Distribution", box=box.ROUNDED)
    table.add_column("Issue Type", style="cyan")
    table.add_column("Frequency", justify="right", style="magenta")
    table.add_column("Visual", style="blue")
    
    max_count = max(issue_counts.values()) if issue_counts else 1
    
    for issue, count in issue_counts.most_common():
        bar_length = int((count / max_count) * 30)
        bar = "█" * bar_length
        table.add_row(
            issue.replace('_', ' ').title(),
            str(count),
            bar
        )
    
    console.print(table)

def main():
    """Main function to visualize all results."""
    console.print(Panel.fit(
        "[bold]Amazon Reviews Analysis Results[/bold]\n"
        "Powered by DocETL Pipeline",
        border_style="bright_blue"
    ))
    
    # Load all result files
    analyzed_reviews = load_json_file("/workspace/amazon-reviews-pipeline/output/analyzed_reviews.json")
    category_insights = load_json_file("/workspace/amazon-reviews-pipeline/output/category_insights.json")
    strategic_insights = load_json_file("/workspace/amazon-reviews-pipeline/output/strategic_insights.json")
    
    # If strategic_insights is not found, try summary_report
    if not strategic_insights:
        strategic_insights = [load_json_file("/workspace/amazon-reviews-pipeline/output/summary_report.json")]
    
    if not analyzed_reviews:
        console.print("[red]No analysis results found. Please run the DocETL pipeline first.[/red]")
        return
    
    # Display various analyses
    console.print("\n")
    
    # Sentiment distribution
    sentiment_table = create_sentiment_distribution_table(analyzed_reviews)
    console.print(sentiment_table)
    console.print("\n")
    
    # Issue categories
    create_issue_category_chart(analyzed_reviews)
    console.print("\n")
    
    # Authenticity analysis
    display_authenticity_analysis(analyzed_reviews)
    console.print("\n")
    
    # Category insights
    if category_insights:
        console.print("[bold]Category-Specific Insights[/bold]")
        panels = create_category_insights_panels(category_insights)
        
        # Display panels in columns
        for i in range(0, len(panels), 2):
            if i + 1 < len(panels):
                console.print(Columns([panels[i], panels[i + 1]], equal=True, padding=1))
            else:
                console.print(panels[i])
        console.print("\n")
    
    # Strategic insights
    if strategic_insights and len(strategic_insights) > 0:
        display_strategic_insights(strategic_insights[0])
    
    # Summary statistics
    console.print("\n")
    stats_panel = Panel(
        f"[cyan]Total Reviews Analyzed:[/cyan] {len(analyzed_reviews)}\n"
        f"[cyan]Categories Covered:[/cyan] {len(set(r.get('category', '') for r in analyzed_reviews))}\n"
        f"[cyan]Average Rating:[/cyan] {sum(r.get('rating', 0) for r in analyzed_reviews) / len(analyzed_reviews):.1f}/5",
        title="Summary Statistics",
        border_style="blue"
    )
    console.print(stats_panel)

if __name__ == "__main__":
    main()