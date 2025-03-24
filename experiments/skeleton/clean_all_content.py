import json
import os
from pathlib import Path
from typing import Dict, List, Any


def clean_article_content(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean an article by truncating the all_content field and keeping only specific fields.
    
    Args:
        article: The original article dictionary
        
    Returns:
        A cleaned article with only uuid, url, and truncated all_content
    """
    return {
        "uuid": article.get("uuid", ""),
        "url": article.get("url", ""),
        "all_content": article.get("all_content", "")[:1000000]
    }


def process_articles_file(input_path: str, output_path: str) -> None:
    """
    Process the articles file by cleaning each article and saving to a new file.
    
    Args:
        input_path: Path to the input JSON file
        output_path: Path to save the output JSON file
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the input file
    with open(input_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    # Clean each article
    cleaned_articles = [clean_article_content(article) for article in articles]
    
    # Save the cleaned articles
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_articles, f, ensure_ascii=False, indent=2)
    
    print(f"Processed {len(cleaned_articles)} articles")
    print(f"Saved cleaned articles to {output_path}")


def main() -> None:
    """Main function to run the cleaning process."""
    input_file = "experiments/skeleton/blackvault_articles_pdfs.json"
    output_file = "experiments/skeleton/blackvault_articles_trunc.json"
    
    process_articles_file(input_file, output_file)


if __name__ == "__main__":
    main()
