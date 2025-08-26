#!/usr/bin/env python3
"""
Quick demo of the real reviews analysis pipeline
Analyzes a small sample to demonstrate functionality
"""

import json
import os
from run_analysis import analyze_review, load_reviews
from rich.console import Console
import asyncio

console = Console()

async def quick_demo():
    """Run a quick analysis on a few real reviews"""
    console.print("[bold cyan]Real Reviews Analysis Demo[/bold cyan]\n")
    
    # Load real reviews
    reviews = load_reviews("data/real_reviews.json", limit=5)
    console.print(f"Loaded {len(reviews)} real reviews for demo\n")
    
    # Show sample reviews
    console.print("[bold]Sample Real Reviews:[/bold]")
    for i, review in enumerate(reviews[:3], 1):
        console.print(f"\n[cyan]Review {i}:[/cyan]")
        console.print(f"Product: {review.get('product_name', 'Unknown')}")
        console.print(f"Rating: {'⭐' * review.get('rating', 0)}")
        console.print(f"Text: {review['review_text'][:150]}...")
    
    console.print("\n[bold]Analyzing with AI...[/bold]\n")
    
    # Analyze reviews
    results = []
    for review in reviews:
        result = await analyze_review(review)
        if result:
            results.append(result)
            console.print(f"✓ Analyzed: {review['product_name']} - Sentiment: {result['sentiment']}")
    
    # Show insights
    console.print("\n[bold]Analysis Results:[/bold]")
    for result in results[:3]:
        console.print(f"\n[green]Product:[/green] {result['product_name']}")
        console.print(f"[yellow]Sentiment:[/yellow] {result['sentiment']}")
        if result.get('praise_points'):
            console.print(f"[green]Praise:[/green] {', '.join(result['praise_points'][:2])}")
        if result.get('complaints'):
            console.print(f"[red]Issues:[/red] {', '.join(result['complaints'][:2])}")
        console.print(f"[cyan]Authenticity:[/cyan] {result.get('authenticity_score', 'N/A')}/10")
    
    console.print("\n[bold green]✅ Demo complete![/bold green]")
    console.print("This demonstrates the pipeline analyzing real product reviews.")

if __name__ == "__main__":
    # Check API key
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your-key-here":
        console.print("[red]Please set your OpenAI API key first![/red]")
        console.print("export OPENAI_API_KEY='your-actual-key'")
        exit(1)
    
    asyncio.run(quick_demo())