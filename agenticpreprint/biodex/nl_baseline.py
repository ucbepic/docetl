import pandas as pd
import os
import json
import time
from typing import List, Dict, Any, Tuple, Set
import multiprocessing as mp
from functools import partial
import re
from collections import Counter


def normalize_text(text: str) -> str:
    """
    Normalize text for basic keyword matching.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    # Replace newlines and tabs with spaces
    text = re.sub(r'[\n\t]+', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # Add spaces around punctuation to ensure proper word boundaries
    text = re.sub(r'([.,!?;:])', r' \1 ', text)
    # Remove any duplicate spaces that might have been introduced
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def create_word_boundary_pattern(term: str) -> str:
    """
    Create a regex pattern that matches the term with word boundaries.
    
    Args:
        term: The term to create a pattern for
        
    Returns:
        Regex pattern string
    """
    # Escape any special regex characters in the term
    escaped_term = re.escape(term)
    # Create pattern with word boundaries
    return r'\b' + escaped_term + r'\b'

def find_keyword_matches(
    article_data: Dict[str, Any], 
    reaction_terms: List[Dict[str, str]],
    article_index: int
) -> List[Dict[str, Any]]:
    """
    Find keyword matches between reaction terms and document text.
    Simply checks if each term exists in the document.
    
    Args:
        article_data: Dictionary containing article data
        reaction_terms: List of reaction term dictionaries
        article_index: Index of article for debugging
        
    Returns:
        List of dictionaries containing match information
    """
    # Get article ID - try multiple possible field names
    article_id = None
    for id_field in ['id', 'paper_id', 'pmid', 'doi']:
        if id_field in article_data and article_data[id_field]:
            article_id = article_data[id_field]
            break
    
    # If no ID found, use index
    if not article_id:
        article_id = f"doc_{article_index}"
    
    fulltext = article_data.get("fulltext_processed", "")
    if not fulltext:
        return []  # Skip empty documents
    
    # Normalize the document text once
    normalized_fulltext = normalize_text(fulltext)
    
    results = []
    
    # For each term, check if it appears in the document with whole word matching
    for term_dict in reaction_terms:
        term = term_dict["reaction"]
        normalized_term = normalize_text(term)
        
        # Multi-word terms
        if ' ' in normalized_term:
            # For multi-word terms, use direct string matching
            if normalized_term in normalized_fulltext:
                results.append({
                    "id": article_id,
                    "fulltext_processed": fulltext[:100] + "...",  # Just keep a snippet for debugging
                    "reaction": term,
                    "_scores": len(term)  # Use length of the term as the score
                })
        # Single word terms - use word boundary matching
        else:
            # Create pattern for whole word matching
            pattern = create_word_boundary_pattern(normalized_term)
            matches = re.findall(pattern, normalized_fulltext)
            
            if matches:
                results.append({
                    "id": article_id,
                    "fulltext_processed": fulltext[:100] + "...",  # Just keep a snippet for debugging
                    "reaction": term,
                    "_scores": len(term)  # Use length of the term as the score
                })
    
    # Sort results by term length (descending) to prioritize longer, more specific matches
    results.sort(key=lambda x: len(x["reaction"]), reverse=True)
    
    return results

def process_batch(
    batch_info: Tuple[int, List[Dict[str, Any]]],
    reaction_terms: List[Dict[str, str]],
    total_articles: int
) -> List[Dict[str, Any]]:
    """Process a batch of articles in parallel."""
    batch_idx, articles_batch = batch_info
    batch_size = len(articles_batch)
    start_idx = batch_idx * batch_size
    
    all_results = []
    for i, article in enumerate(articles_batch):
        overall_idx = start_idx + i + 1  # 1-based indexing for display
        try:
            # Get article ID for logging
            article_id = None
            for id_field in ['id', 'paper_id', 'pmid', 'doi']:
                if id_field in article and article[id_field]:
                    article_id = article[id_field]
                    break
            if not article_id:
                article_id = f"doc_{overall_idx}"
                
            if (overall_idx % 10 == 0) or (overall_idx == total_articles) or (overall_idx == 1):  # Print 1st, every 10th, and last
                print(f"Processing document {overall_idx} of {total_articles} (Batch {batch_idx+1}, ID: {article_id})")
                
            matches = find_keyword_matches(article, reaction_terms, overall_idx)
            all_results.extend(matches)
            
            # Print matching info for debugging
            if matches:
                print(f"  - Found {len(matches)} matches in document {article_id}")
                
        except Exception as e:
            print(f"Error processing article {article_id}: {e}")
    
    return all_results

def main():
    start = time.time()
    
    # Load data
    print("Loading data...")
    try:
        with open("agenticpreprint/biodex/biodex_sample.json", "r") as f:
            articles = json.load(f)
        
        with open("agenticpreprint/biodex/biodex_terms.json", "r") as f:
            reaction_labels = json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Print a sample article to debug structure
    if articles and len(articles) > 0:
        print("Sample article keys:", list(articles[0].keys()))
        

    # Convert to DataFrame for consistency with lotus implementation
    articles_df = pd.DataFrame(articles)
    reaction_labels_df = pd.DataFrame(reaction_labels)
    
    # Sort terms by length (descending) to prioritize longer matches
    reaction_labels_df = reaction_labels_df.sort_values(by="reaction", key=lambda x: x.str.len(), ascending=False)
    
    print(f"There are {len(articles_df)} articles and {len(reaction_labels_df)} reaction labels")
    
    # Set up multiprocessing
    num_cores = mp.cpu_count()
    print(f"Using {num_cores} CPU cores")
    
    # Convert back to lists for multiprocessing
    articles_list = articles_df.to_dict("records")
    reaction_terms_list = reaction_labels_df.to_dict("records")
    
    # Split articles into batches for parallel processing
    batch_size = max(1, len(articles_list) // num_cores)
    batches = [(i, articles_list[i*batch_size:(i+1)*batch_size]) for i in range((len(articles_list) + batch_size - 1) // batch_size)]
    
    total_articles = len(articles_list)
    
    # Process batches in parallel
    print(f"Processing {len(batches)} batches with {total_articles} total articles...")
    start_processing = time.time()
    
    with mp.Pool(processes=num_cores) as pool:
        process_func = partial(process_batch, reaction_terms=reaction_terms_list, total_articles=total_articles)
        results_lists = pool.map(process_func, batches)
    
    processing_time = time.time() - start_processing
    print(f"Processing completed in {processing_time:.2f} seconds")
    
    # Combine results
    all_results = []
    for results in results_lists:
        all_results.extend(results)
    
    # Print unique document IDs for debugging
    unique_docs = set(r["id"] for r in all_results)
    print(f"Found matches in {len(unique_docs)} unique documents")
    if len(unique_docs) < 10:
        print(f"Document IDs with matches: {unique_docs}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Add an estimated article_reaction field to match lotus output structure
    if len(results_df) > 0:
        results_df["article_reaction"] = results_df.apply(
            lambda row: f"The patient experienced {row['reaction']}", axis=1
        )
    
    # Ensure output directory exists
    os.makedirs("agenticpreprint/biodex/results", exist_ok=True)
    
    # Save results
    results_df.to_csv("agenticpreprint/biodex/results/baseline_output.csv", index=False)
    print(f"Results saved to agenticpreprint/biodex/results/baseline_output.csv")
    
    # Print statistics
    end = time.time()
    total_time = end - start
    print(f"Total matches found: {len(results_df)}")
    
    # Count unique articles with matches
    if len(results_df) > 0:
        unique_articles = results_df["id"].nunique()
        print(f"Articles with at least one match: {unique_articles} ({unique_articles/len(articles_df)*100:.1f}%)")
        doc_counts = results_df["id"].value_counts()
        print(f"Top 5 documents by match count: {doc_counts.head()}")
    
    print(f"Total processing time: {total_time:.2f} seconds")
    if len(articles_df) > 0:
        print(f"Average processing time per article: {total_time / len(articles_df):.4f} seconds")
    if len(all_results) > 0 and "id" in results_df.columns:
        matches_by_article = results_df.groupby("id").size()
        print(f"Average matches per matching document: {matches_by_article.mean():.2f}")
        print(f"Max matches in a single document: {matches_by_article.max()}")
        top_terms = Counter(results_df["reaction"]).most_common(10)
        print(f"Top 10 matched terms: {top_terms}")

if __name__ == "__main__":
    main()
