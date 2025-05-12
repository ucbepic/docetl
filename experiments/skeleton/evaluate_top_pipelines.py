import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import ndcg_score

# Import textstat for Flesch-Kincaid readability scores
try:
    import textstat
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False
    print("Warning: textstat not found, will use simplified readability assessment")

# Import the semantic accessor for location detection
try:
    from docetl import SemanticAccessor
    HAS_DOCETL = True
except ImportError:
    HAS_DOCETL = False
    print("Warning: docetl not found, will use simplified location detection")


def calculate_readability_score(text: str) -> float:
    """Calculate the Flesch-Kincaid readability score for the text."""
    if not HAS_TEXTSTAT:
        # Simplified implementation if textstat isn't available
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0
            
        words = text.split()
        if not words:
            return 0
            
        avg_words_per_sentence = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Approximation of Flesch-Kincaid
        score = 100 - (0.39 * avg_words_per_sentence + 11.8 * avg_word_length - 15.59)
        return max(0, min(100, score))  # Clamp between 0-100
    else:
        # Use textstat for proper calculation
        fk_score = textstat.flesch_reading_ease(text)
        return max(0, min(100, fk_score))  # Clamp between 0-100


def load_json_file(file_path: Path) -> Any:
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return None


def detect_months(text: str) -> List[str]:
    """Detect mentions of months in text."""
    months = [
        "january", "february", "march", "april", "may", "june", 
        "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"
    ]
    
    found_months = []
    for month in months:
        # Look for the month as a word boundary
        pattern = r'\b' + month + r'\b'
        if re.search(pattern, text.lower()):
            found_months.append(month)
            break
    
    return found_months


def detect_locations_with_llm_batch(summaries: List[str]) -> List[List[str]]:
    """
    Use DocETL to detect locations in multiple text summaries in parallel.
    
    Args:
        summaries: List of summary texts to analyze
        
    Returns:
        List of location lists, one list per summary
    """
    if not HAS_DOCETL or not summaries:
        # Simple fallback detection without LLM
        common_locations = [
            "united states", "usa", "us", "america", "washington", "new york", "california",
            "russia", "moscow", "china", "beijing", "uk", "london", "paris", "france",
            "germany", "berlin", "tokyo", "japan", "australia", "sydney", "canada", "mexico"
        ]
        results = []
        for text in summaries:
            found_locations = []
            for loc in common_locations:
                if re.search(r'\b' + re.escape(loc) + r'\b', text.lower()):
                    found_locations.append(loc)
            results.append(found_locations)
        return results
    
    # Use DocETL pandas API for batch location detection
    print(f"Processing {len(summaries)} summaries for location detection in parallel...")
    df = pd.DataFrame({"summary": summaries})
    df.semantic.set_config(default_model="gemini/gemini-2.5-pro-preview-03-25")
    
    result = df.semantic.map(
        prompt="""Extract all geographic locations mentioned in the text. 
        Include countries, cities, states, regions, and other geographic areas.
        Return only the list of locations, nothing else.
        
        Text: {{input.summary}}""",
        output_schema={"locations": "list[str]"}
    )
    
    # Extract all location lists from the result dataframe
    location_lists = result["locations"].tolist()
    print(f"Location detection complete: found locations in {sum(1 for locs in location_lists if locs)} summaries")
    return location_lists


def extract_summary_text(data: Any) -> str:
    """Extract summary text from various JSON structures."""
    summary_text = ""
    
    if isinstance(data, list) and data and isinstance(data[0], dict) and 'summary' in data[0]:
        # Combine all summaries from all documents
        for doc in data:
            summary = doc.get('summary', '')
            if isinstance(summary, str):
                summary_text += summary + " "
    elif isinstance(data, dict) and 'summary' in data:
        summary = data.get('summary', '')
        if isinstance(summary, str):
            summary_text = summary
    
    return summary_text.strip()


def calculate_ndcg(actual_ranks: List[int], predicted_ranks: List[int], k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) at k using sklearn.
    
    Args:
        actual_ranks: List of actual ranks (positions) - lower is better
        predicted_ranks: List of predicted ranks (positions) - lower is better
        k: Number of top positions to consider
        
    Returns:
        NDCG score between 0 and 1
    """
    # Convert ranks to relevance scores (higher is better)
    max_rank = max(max(actual_ranks), max(predicted_ranks)) + 1
    
    # Create relevance arrays (sklearn expects 2D arrays)
    # We convert ranks to relevance scores where lower rank = higher relevance
    relevance = [[max_rank - r for r in actual_ranks]]
    
    # For y_score, we need to convert predicted ranks to scores
    # Lower predicted rank should result in higher score
    pred_scores = [[max_rank - r for r in predicted_ranks]]
    
    # Use sklearn's ndcg_score with k
    return ndcg_score(relevance, pred_scores, k=k)


def analyze_pipeline_results(config_name: str) -> None:
    """Analyze pipeline results and create visualization."""
    # Get a display name for the config
    display_name = "Pipeline A" if config_name == "map_test" else "Pipeline B"
    
    # Determine paths
    base_dir = Path(__file__).parent
    output_dir = base_dir / config_name
    stats_path = output_dir / "stats.json"
    
    # Verify the directory exists
    if not output_dir.exists() or not output_dir.is_dir():
        print(f"Error: Output directory {output_dir} not found.")
        return
    
    # Load stats file
    stats_data = load_json_file(stats_path)
    if stats_data is None:
        print(f"Error: Could not load stats from {stats_path}")
        return
    
    print(f"Loaded stats data for {len(stats_data)} pipelines in {display_name}")
    
    # Collect all result files from the directory
    result_files = list(output_dir.glob("*.json"))
    result_files = [f for f in result_files if f.name != "stats.json"]
    
    if not result_files:
        print(f"Error: No result files found in {output_dir}")
        return
    
    print(f"Found {len(result_files)} result files to analyze")
    
    # Pre-process: load all result files and extract necessary data
    file_names = []
    costs = []
    output_data_list = []
    
    for file_path in result_files:
        file_name = file_path.name
        
        # Get cost from stats
        cost = stats_data.get(file_name, {}).get("cost")
        if cost is None:
            print(f"Warning: No cost data found for {file_name} in stats.json")
            continue
            
        # Load the actual output data
        output_data = load_json_file(file_path)
        if output_data is None:
            print(f"Warning: Could not load output data from {file_path}")
            continue
        
        file_names.append(file_name)
        costs.append(cost)
        output_data_list.append(output_data)
    
    # Process based on config_name
    if config_name == "map_test":
        # Process map_test results (just count locations)
        results = []
        for i, (file_name, cost, data) in enumerate(zip(file_names, costs, output_data_list)):
            # Count unique locations
            all_locations = []
            if isinstance(data, list) and data and isinstance(data[0], dict) and 'locations' in data[0]:
                for doc in data:
                    if isinstance(doc.get('locations'), list):
                        all_locations.extend(doc['locations'])
            elif isinstance(data, dict) and 'locations' in data:
                if isinstance(data['locations'], list):
                    all_locations = data['locations']
            
            unique_locations = len(set(all_locations)) if all_locations else 0
            is_original = file_name.startswith("original")
            
            results.append({
                "file_name": file_name,
                "cost": cost,
                "output_size": unique_locations,
                "is_original": is_original
            })
    
    else:  # map_summary_test - more comprehensive analysis with parallel location detection
        # Extract summary texts
        summaries = []
        for data in output_data_list:
            summary_text = extract_summary_text(data)
            summaries.append(summary_text)
        
        # Batch detect locations using LLM
        location_lists = detect_locations_with_llm_batch(summaries)
        
        # First pass: collect all raw metrics
        raw_results = []
        for i, (file_name, cost, data, summary_text, locations) in enumerate(
            zip(file_names, costs, output_data_list, summaries, location_lists)
        ):
            # Skip if no summary text
            if not summary_text:
                print(f"Warning: No summary text found in {file_name}")
                continue
            
            # Detect months
            months = detect_months(summary_text)
            
            # Calculate summary length
            summary_length = len(summary_text)
            
            # Calculate readability score
            readability_score = calculate_readability_score(summary_text)
            
            # Extract pipeline identifier from filename (assuming format like rank_X.json or original.json)
            pipeline_id = "original" if file_name.startswith("original") else file_name.split(".")[0]
            
            # Store raw metrics
            raw_results.append({
                "file_name": file_name,
                "pipeline_id": pipeline_id,
                "cost": cost,
                "raw_length": summary_length,
                "months_count": len(months),
                "locations_count": len(locations),
                "readability_score": readability_score,
                "locations": locations,
                "months": months,
                "is_original": file_name.startswith("original")
            })
        
        # Group results by pipeline and calculate averages
        pipeline_averages = {}
        for result in raw_results:
            pipeline_id = result["pipeline_id"]
            if pipeline_id not in pipeline_averages:
                pipeline_averages[pipeline_id] = {
                    "count": 0,
                    "total_length": 0,
                    "total_months": 0,
                    "total_locations": 0,
                    "total_readability": 0,
                    "is_original": result["is_original"],
                    "cost": result["cost"],
                    "file_name": result["file_name"],
                    "all_months": set(),  # Store unique months as a set
                    "all_locations": set()  # Store unique locations as a set
                }
            
            avg = pipeline_averages[pipeline_id]
            avg["count"] += 1
            avg["total_length"] += result["raw_length"]
            avg["total_months"] += result["months_count"]
            avg["total_locations"] += min(result["locations_count"], 3)  # Cap at 3 as in original code
            avg["total_readability"] += result["readability_score"]
            
            # Add all months and locations to the sets
            if result["months"]:
                avg["all_months"].update(result["months"])
            if result["locations"]:
                avg["all_locations"].update(result["locations"])
        
        # Calculate final averages for each pipeline
        pipeline_metrics = []
        for pipeline_id, totals in pipeline_averages.items():
            count = totals["count"]
            if count > 0:
                pipeline_metrics.append({
                    "pipeline_id": pipeline_id,
                    "file_name": totals["file_name"],
                    "cost": totals["cost"],
                    "avg_length": totals["total_length"] / count,
                    "avg_months": totals["total_months"] / count,
                    "avg_locations": totals["total_locations"] / count,
                    "avg_readability": totals["total_readability"] / count,
                    "is_original": totals["is_original"],
                    "months": list(totals["all_months"]),  # Convert set to list
                    "locations": list(totals["all_locations"])  # Convert set to list
                })
        
        # Extract values for normalization across pipelines
        all_lengths = [p["avg_length"] for p in pipeline_metrics]
        all_months = [p["avg_months"] for p in pipeline_metrics]
        all_locations = [p["avg_locations"] for p in pipeline_metrics]
        all_readability = [p["avg_readability"] for p in pipeline_metrics]
        
        # Calculate normalization factors (avoid division by zero)
        length_range = max(all_lengths) - min(all_lengths) if len(set(all_lengths)) > 1 else 1
        months_range = max(all_months) - min(all_months) if len(set(all_months)) > 1 else 1
        locations_range = max(all_locations) - min(all_locations) if len(set(all_locations)) > 1 else 1
        readability_range = max(all_readability) - min(all_readability) if len(set(all_readability)) > 1 else 1
        
        # Normalize and combine
        results = []
        for p in pipeline_metrics:
            # Normalized scores (0-1 scale)
            if length_range > 0:
                length_score = (p["avg_length"] - min(all_lengths)) / length_range
            else:
                length_score = 1.0 if p["avg_length"] > 0 else 0.0
                
            # For months, use normalized value instead of binary existence
            if months_range > 0:
                months_score = (p["avg_months"] - min(all_months)) / months_range
            else:
                months_score = 1.0 if p["avg_months"] > 0 else 0.0
                
            if locations_range > 0:
                locations_score = (p["avg_locations"] - min(all_locations)) / locations_range
            else:
                locations_score = 1.0 if p["avg_locations"] > 0 else 0.0
                
            if readability_range > 0:
                readability_score = (p["avg_readability"] - min(all_readability)) / readability_range
            else:
                readability_score = 1.0 if p["avg_readability"] > 0 else 0.0
            
            # Weighted composite score - remove length factor and redistribute weights
            # Scale to 0-10 for easier interpretation
            composite_score = 10 * (
                (0.3 * months_score) + 
                (0.3 * locations_score) + 
                (0.4 * readability_score)
            )
            
            # Create result entry with normalized score and average metrics
            results.append({
                "file_name": p["file_name"],
                "pipeline_id": p["pipeline_id"],
                "cost": p["cost"],
                "output_size": composite_score,
                "raw_length": p["avg_length"],
                "months_count": p["avg_months"],
                "has_month": p["avg_months"] > 0,
                "locations_count": p["avg_locations"],
                "has_location": p["avg_locations"] > 0,
                "readability_score": p["avg_readability"],
                "is_original": p["is_original"],
                "months": p["months"],  # Include the months list
                "locations": p["locations"],  # Include the locations list
                # Store normalized scores for reference
                "normalized_length": length_score,
                "normalized_months": months_score,
                "normalized_locations": locations_score,
                "normalized_readability": readability_score
            })
    
    if not results:
        print("Error: No valid results to analyze")
        return
    
    # After processing results and before visualizing, add ranking evaluation
    if results:
        # Filter out the original pipeline for rank comparison
        rank_results = [r for r in results if not r["is_original"]]
        
        if len(rank_results) > 1:  # Need at least 2 items to compare rankings
            # Extract proposed ranks from filenames
            proposed_ranks = []
            for r in rank_results:
                filename = r["file_name"]
                if filename.startswith("rank_"):
                    try:
                        rank = int(filename.split("_")[1].split(".")[0])
                        proposed_ranks.append((rank, r))
                    except (IndexError, ValueError):
                        # Skip if rank cannot be parsed
                        pass
            
            if proposed_ranks:
                print("\nRanking Evaluation:")
                print("-" * 80)
                
                # Sort by proposed rank
                proposed_ranks.sort(key=lambda x: x[0])
                
                # Extract items in proposed rank order
                items_by_proposed_rank = [item for _, item in proposed_ranks]
                proposed_rank_values = [rank for rank, _ in proposed_ranks]
                
                # Calculate actual rank based on output_size (higher is better)
                items_by_actual_rank = sorted(
                    items_by_proposed_rank, 
                    key=lambda x: -x["output_size"]  # Negative to sort in descending order
                )
                
                # Create mapping from item to its rank position
                actual_rank_map = {item["file_name"]: rank for rank, item in enumerate(items_by_actual_rank)}
                
                # Get actual ranks in the same order as proposed ranks
                actual_rank_values = [actual_rank_map[item["file_name"]] for item in items_by_proposed_rank]
                
                # Print comparison table
                print(f"{'Proposed Rank':<15} {'Filename':<20} {'Actual Rank':<15} {'Score':<10}")
                print("-" * 80)
                for prop_rank, item, act_rank in zip(
                    proposed_rank_values, items_by_proposed_rank, actual_rank_values
                ):
                    print(f"{prop_rank:<15} {item['file_name']:<20} {act_rank:<15} {item['output_size']:.2f}")
                
                # Calculate Kendall's tau
                tau, p_value = kendalltau(proposed_rank_values, actual_rank_values)
                print(f"\nKendall's tau: {tau:.4f} (p-value: {p_value:.4f})")
                
                # Calculate NDCG@5
                ndcg_score = calculate_ndcg(actual_rank_values, proposed_rank_values, k=5)
                print(f"NDCG@5: {ndcg_score:.4f}")
                
                # Interpret results
                print("\nInterpretation:")
                if tau > 0.6:
                    print("- Strong positive correlation between proposed and actual rankings")
                elif tau > 0.3:
                    print("- Moderate positive correlation between proposed and actual rankings")
                elif tau > 0:
                    print("- Weak positive correlation between proposed and actual rankings")
                elif tau == 0:
                    print("- No correlation between proposed and actual rankings")
                else:
                    print("- Negative correlation between proposed and actual rankings")
                
                if ndcg_score > 0.9:
                    print("- Excellent top-10 ranking accuracy")
                elif ndcg_score > 0.7:
                    print("- Good top-10 ranking accuracy")
                elif ndcg_score > 0.5:
                    print("- Fair top-10 ranking accuracy")
                else:
                    print("- Poor top-10 ranking accuracy")

    # Extract data for plotting
    plot_costs = [r["cost"] for r in results]
    plot_sizes = [r["output_size"] for r in results]
    plot_labels = [r["file_name"] for r in results]
    plot_is_original = [r["is_original"] for r in results]
    
    # Set y-axis label based on config
    if config_name == "map_summary_test":
        y_axis_label = "Normalized Quality Score (0-10)"
    else:
        y_axis_label = "Number of Unique Locations"
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    # Plot non-original points
    for i, (cost, size, label, orig) in enumerate(zip(plot_costs, plot_sizes, plot_labels, plot_is_original)):
        if not orig:
            plt.scatter(cost, size, marker='o', s=80, alpha=0.7)
            # Add rank number as label
            if label.startswith("rank_"):
                rank = label.split("_")[1].split(".")[0]
                plt.annotate(f"Rank {rank}", (cost, size), xytext=(5, 5), textcoords="offset points")
    
    # Plot original point as a star
    for i, (cost, size, label, orig) in enumerate(zip(plot_costs, plot_sizes, plot_labels, plot_is_original)):
        if orig:
            plt.scatter(cost, size, marker='*', color='red', s=200, label="Original Pipeline")
            plt.annotate("Original", (cost, size), xytext=(5, 5), textcoords="offset points")
    
    # Add title and labels
    plt.title(f"Pipeline Performance: {display_name}")
    plt.xlabel("Cost ($)")
    plt.ylabel(y_axis_label)
    plt.grid(True, alpha=0.3)
    
    # Add legend if the original pipeline is in the results
    if any(plot_is_original):
        plt.legend()
    
    # Determine y-axis limits with some padding
    if plot_sizes:
        y_min, y_max = min(plot_sizes), max(plot_sizes)
        y_range = y_max - y_min
        plt.ylim(max(0, y_min - 0.1 * y_range), y_max + 0.1 * y_range)
    
    # Save the plot
    output_plot_path = output_dir / f"{config_name}_evaluation.png"
    plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_plot_path}")
    
    # Print summary statistics
    print(f"\nSummary Statistics for {display_name}:")
    print("-" * 80)
    
    if config_name == "map_test":
        print(f"{'Pipeline':<20} {'Cost ($)':<12} {'Unique Locations':<15}")
        print("-" * 80)
        
        # Sort by number of locations for the table
        sorted_results = sorted(results, key=lambda x: (x["output_size"] * -1, x["cost"]))
        
        for r in sorted_results:
            cost_str = f"{r['cost']:.6f}" if r['cost'] is not None else "N/A"
            print(f"{r['file_name']:<20} {cost_str:<12} {r['output_size']:<15}")
    else:
        # For map_summary_test, print more detailed stats with normalized scores
        print(f"{'Pipeline':<15} {'Cost ($)':<10} {'Length':<8} {'Has Month':<10} {'Locations':<8} {'Readability':<10} {'Quality':<8}")
        print("-" * 90)
        
        # Sort by composite score for the table
        sorted_results = sorted(results, key=lambda x: (x["output_size"] * -1, x["cost"]))
        
        for r in sorted_results:
            cost_str = f"{r['cost']:.4f}" if r['cost'] is not None else "N/A"
            has_month = "Yes" if r["months_count"] > 0 else "No"
            print(f"{r['file_name']:<15} {cost_str:<10} {r['raw_length']:<8} {has_month:<10} {r['locations_count']:<8} {r['readability_score']:<10.1f} {r['output_size']:.2f}")
            
            # Print normalized scores
            norm_str = (f"  Normalized scores - Length: {r['normalized_length']:.2f}, "
                       f"Month: {r['normalized_months']:.2f}, "
                       f"Locations: {r['normalized_locations']:.2f}, "
                       f"Readability: {r['normalized_readability']:.2f}")
            print(norm_str)
            
            # If there are months and locations, print them
            if r['months_count'] > 0:
                print(f"  Months: {', '.join(r['months'])}")
            if r['locations_count'] > 0:
                print(f"  Locations: {', '.join(r['locations'])}")
    
    # Find the best pipelines (different metrics for each type)
    if config_name == "map_test":
        valid_results = [r for r in results if r["cost"] is not None and r["cost"] > 0 and r["output_size"] > 0]
        if valid_results:
            for r in valid_results:
                r["efficiency"] = r["output_size"] / r["cost"] if r["cost"] > 0 else 0
            
            best_efficiency = max(valid_results, key=lambda x: x["efficiency"])
            print("\nMost Efficient Pipeline (locations per $):")
            print(f"- {best_efficiency['file_name']}: {best_efficiency['efficiency']:.2f} locations/$")
            
            largest_output = max(valid_results, key=lambda x: x["output_size"])
            print("\nPipeline with Most Locations:")
            print(f"- {largest_output['file_name']}: {largest_output['output_size']} unique locations")
    else:
        valid_results = [r for r in results if r["cost"] is not None and r["cost"] > 0 and r["output_size"] > 0]
        if valid_results:
            for r in valid_results:
                r["efficiency"] = r["output_size"] / r["cost"] if r["cost"] > 0 else 0
            
            best_efficiency = max(valid_results, key=lambda x: x["efficiency"])
            print("\nMost Efficient Pipeline (quality per $):")
            print(f"- {best_efficiency['file_name']}: {best_efficiency['efficiency']:.2f} quality/$")
            
            best_quality = max(valid_results, key=lambda x: x["output_size"])
            print("\nPipeline with Highest Quality Score:")
            print(f"- {best_quality['file_name']}: {best_quality['output_size']:.2f}")
            
            most_readable = max(valid_results, key=lambda x: x["readability_score"])
            print("\nMost Readable Pipeline:")
            print(f"- {most_readable['file_name']}: {most_readable['readability_score']:.1f} readability score")
            
            most_locations = max(valid_results, key=lambda x: x["locations_count"])
            print("\nPipeline with Most Locations:")
            print(f"- {most_locations['file_name']}: {most_locations['locations_count']} locations")
            
            most_months = max(valid_results, key=lambda x: x["months_count"])
            print("\nPipeline with Most Month References:")
            print(f"- {most_months['file_name']}: {most_months['months_count']} months")
    
    # Print cheapest pipeline for both types
    cheapest = min(results, key=lambda x: x["cost"] if x["cost"] is not None else float('inf'))
    print("\nCheapest Pipeline:")
    print(f"- {cheapest['file_name']}: ${cheapest['cost']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate and visualize pipeline results.")
    parser.add_argument(
        "--config-name",
        required=True,
        choices=["map_test", "map_summary_test"],
        help="Name of the configuration (e.g., 'map_test')."
    )
    args = parser.parse_args()
    
    analyze_pipeline_results(args.config_name)


if __name__ == "__main__":
    main() 