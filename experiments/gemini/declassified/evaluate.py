import json
from collections import Counter
from pathlib import Path
import pandas as pd
import requests
import time
from typing import Dict, List, Optional
from docetl import SemanticAccessor
import matplotlib.pyplot as plt

# Hardcoded costs in dollars for each plan index
PLAN_COSTS = [0.27, 0.23, 0.18, 0.17, 0.25, 0.23, 0.23, 0.18, 0.22]
BASELINE_COST = 0.27

# Hardcoded runtimes in seconds for each plan index
PLAN_RUNTIMES = [155.94, 138.32, 109.94, 82.45, 154.80, 121.91, 133.16, 104.83, 115.63]
BASELINE_RUNTIME = 171.46

def load_json_file(filepath):
    """Load and parse a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
        return data

def normalize_text(text):
    """Normalize text by removing commas and converting to lowercase."""
    return text.lower().replace(',', '')

def get_location_validity(locations: list[str], source_text: str) -> Dict[str, Dict[str, bool]]:
    """
    Check both if locations are valid real-world locations and if they exist in the source text.
    
    Args:
        locations: List of location strings to validate
        source_text: Source text to check for location presence
        
    Returns:
        Dictionary mapping each location to a dict with 'is_valid_location' and 'exists_in_text' boolean values
    """
    # Convert source_text to normalized form for comparison
    normalized_source_text = normalize_text(source_text)
    
    # First check which locations exist in the text
    location_exists = {}
    for location in locations:
        location_words = normalize_text(location).split()
        # Check if location appears in source text
        exists_in_text = False
        if location_words and location_words[0] in normalized_source_text:
            exists_in_text = True
        location_exists[location] = exists_in_text
    
    # Use LLM to check if the locations are valid
    location_df = pd.DataFrame(locations, columns=['location'])
    location_df.semantic.set_config(default_model="gpt-4o-mini")
    result_df = location_df.semantic.map(prompt="""
    You are a helpful assistant that validates location names.
    Given a location name, you will determine if it is a valid location on Earth.
    If it is, return True. If it is not, return False.
    Location: {{ input.location }}""", output_schema={"valid_location": "bool"})
    
    location_validity = dict(zip(result_df['location'], result_df['valid_location']))
    
    # Combine both validations
    result = {}
    for location in locations:
        result[location] = {
            "is_valid_location": location_validity.get(location, False),
            "exists_in_text": location_exists.get(location, False)
        }
    
    return result

def count_locations_per_event(data):
    """Count the number of unique locations for each event type."""
    event_counts = Counter()
    
    for event in data:
        event_type = event['event_type']
        locations = event.get('locations', [])
        
        # If locations is a string with ```tool_code\nprint(default_api.send_output(locations=, let's strip and eval
        if isinstance(locations, str) and locations.startswith("```tool_code\nprint(default_api.send_output(locations="):
            locations_str  = locations.strip("```tool_code\nprint(default_api.send_output(locations=\n").strip()
            # IF locations_str ends with , remove it
            if locations_str.endswith(","):
                locations_str = locations_str[:-1]
            # Add ' if it's not there
            if not locations_str.endswith("'") and locations_str[1] == "'":
                locations_str += "'"
            elif not locations_str.endswith('"') and locations_str[1] == '"':
                locations_str += '"'
                
            # If it ends with , ' then remove the last character
            if locations_str.endswith(", '"):
                locations_str = locations_str[:-2]
                
            # Add '] if it's not there
            if not locations_str.endswith("]"):
                locations_str += "]"
                
            try:
                locations = eval(locations_str)
            except Exception as e:
                raise e
        elif locations is None:
            print(f"Locations are None for event type: {event_type}")
            locations = []
        
        unique_locations = set(locations)  # Deduplicate locations
        event_counts[event_type] += len(unique_locations)
    
    return dict(event_counts)

def load_all_datasets(base_path: Path, file_pattern: str, indices: List[int]) -> Dict[int, List]:
    """Load multiple datasets based on file pattern and indices."""
    datasets = {}
    for idx in indices:
        file_path = base_path / file_pattern.format(idx)
        if file_path.exists():
            datasets[idx] = load_json_file(file_path)
        else:
            print(f"Warning: File {file_path} not found.")
    return datasets

def plot_cost_vs_metrics(costs, distinct_locations, valid_percentages, baseline=None, output_path=None):
    """
    Create a scatter plot with cost on x-axis and metrics on y-axis.
    
    Args:
        costs: List of costs for each plan
        distinct_locations: List of distinct location counts for each plan
        valid_percentages: List of valid location percentages for each plan
        baseline: Optional tuple with (cost, distinct_count, valid_pct) for baseline
        output_path: Optional path to save the plot image
    """
    fig, ax1 = plt.figure(figsize=(12, 8)), plt.gca()
    
    # Map original indices to costs for labeling
    plan_indices = list(range(len(costs)))
    cost_to_plan = {cost: idx for cost, idx in zip(costs, plan_indices)}
    
    # Sort data by cost
    data = sorted(zip(costs, distinct_locations, valid_percentages, plan_indices))
    costs_sorted = [item[0] for item in data]
    distinct_sorted = [item[1] for item in data]
    valid_pct_sorted = [item[2] for item in data]
    plan_indices_sorted = [item[3] for item in data]
    
    # Plot distinct locations (left y-axis)
    color1 = 'tab:blue'
    ax1.set_xlabel('Cost ($)', fontsize=14)
    ax1.set_ylabel('Number of Distinct Locations', color=color1, fontsize=14)
    scatter1 = ax1.scatter(costs_sorted, distinct_sorted, color=color1, s=100, marker='o', 
                          label='Distinct Locations')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Create second y-axis for percentages
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Valid Locations (%)', color=color2, fontsize=14)
    scatter2 = ax2.scatter(costs_sorted, valid_pct_sorted, color=color2, s=100, marker='s',
                          label='Valid Locations %')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add baseline if provided
    if baseline:
        baseline_cost, baseline_distinct, baseline_valid_pct = baseline
        # Add baseline points with same colors but star markers
        ax1.scatter([baseline_cost], [baseline_distinct], color=color1, s=150, marker='*',
                   label='Baseline Distinct Locations', zorder=5)
        ax2.scatter([baseline_cost], [baseline_valid_pct], color=color2, s=150, marker='*',
                   label='Baseline Valid Locations %', zorder=5)
        
        # Add annotations for baseline
        ax1.annotate('Baseline', 
                    (baseline_cost, baseline_distinct),
                    textcoords="offset points",
                    xytext=(0, 15),
                    ha='center',
                    fontsize=12,
                    fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color1, alpha=0.8))
        
        ax2.annotate('Baseline', 
                    (baseline_cost, baseline_valid_pct),
                    textcoords="offset points",
                    xytext=(0, 15),
                    ha='center', 
                    fontsize=12,
                    fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color2, alpha=0.8))
    
    # Add a grid and title
    ax1.grid(True, linestyle='--', alpha=0.7)
    plt.title('Cost vs. Location Metrics', fontsize=16)
    
    # Add annotations for each point on both axes
    for i, (cost, distinct, valid, plan_idx) in enumerate(zip(costs_sorted, distinct_sorted, valid_pct_sorted, plan_indices_sorted)):
        # Label the distinct locations points (blue)
        ax1.annotate(f"Plan {plan_idx}", 
                    (cost, distinct),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.7))
        
        # Label the valid percentage points (red)
        ax2.annotate(f"Plan {plan_idx}", 
                    (cost, valid),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center', 
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.7))
    
    # Create custom handlers and legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color1, markersize=10, label='Distinct Locations'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color2, markersize=10, label='Valid Locations %'),
    ]
    
    # Add baseline elements to legend if baseline exists
    if baseline:
        legend_elements.extend([
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=color1, markersize=12, label='Baseline Distinct'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=color2, markersize=12, label='Baseline Valid %')
        ])
    
    ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12),
              shadow=True, ncol=2, fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    plt.show()

def main():
    # Define file paths
    base_path = Path(__file__).parent
    plans_path = base_path / "plans"
    source_path = Path("experiments/gemini/declassified/blackvault_articles_pdfs.json")
    
    # Path to baseline data
    baseline_path = base_path / "event_locations_unoptimized.json"
    
    # Load source text
    print(f"Loading source text from {source_path}")
    source_text = ""
    try:
        with open(source_path, 'r') as f:
            source_data = json.load(f)
            # Combine all text content from the source data
            # Assuming the file contains a list or dictionary with text fields
            if isinstance(source_data, list):
                # If it's a list, extract text from each item
                for item in source_data:
                    if isinstance(item, dict):
                        # Extract text fields - adjust keys as needed based on actual structure
                        for value in item.values():
                            if isinstance(value, str):
                                source_text += " " + value
            elif isinstance(source_data, dict):
                # If it's a dictionary, extract text from values
                for value in source_data.values():
                    if isinstance(value, str):
                        source_text += " " + value
            
            # Normalize the combined source text
            source_text = normalize_text(source_text)
            print(f"Loaded source text: {len(source_text)} characters")
    except Exception as e:
        print(f"Warning: Could not load source text: {e}")
        print("Proceeding with empty source text")
    
    # Indices to evaluate (0 through 8)
    indices = list(range(9))
    
    # Load all datasets
    datasets = load_all_datasets(plans_path, "event_locations_{}.json", indices)
    
    # Load baseline dataset
    baseline_data = None
    if baseline_path.exists():
        try:
            baseline_data = load_json_file(baseline_path)
            print(f"Loaded baseline data from {baseline_path}")
        except Exception as e:
            print(f"Error loading baseline data: {e}")
    else:
        print(f"Warning: Baseline file {baseline_path} not found.")
    
    # Get all unique locations across all datasets
    all_unique_locations = set()
    dataset_locations = {}  # To store unique locations for each dataset
    
    # Process regular datasets
    for idx, data in datasets.items():
        locations_in_dataset = set()
        for event in data:
            locations = event.get('locations', [])
            
            # Handle special case of locations being a string with code
            if isinstance(locations, str) and locations.startswith("```tool_code\nprint(default_api.send_output(locations="):
                locations_str = locations.strip("```tool_code\nprint(default_api.send_output(locations=\n").strip()
                # IF locations_str ends with , remove it
                if locations_str.endswith(","):
                    locations_str = locations_str[:-1]
                # Add ' if it's not there
                if not locations_str.endswith("'") and locations_str[1] == "'":
                    locations_str += "'"
                elif not locations_str.endswith('"') and locations_str[1] == '"':
                    locations_str += '"'
                    
                # If it ends with , ' then remove the last character
                if locations_str.endswith(", '"):
                    locations_str = locations_str[:-2]
                    
                # Add '] if it's not there
                if not locations_str.endswith("]"):
                    locations_str += "]"
                    
                try:
                    locations = eval(locations_str)
                except Exception as e:
                    print(f"Error evaluating locations: {e}")
                    locations = []
            elif locations is None:
                locations = []
            
            locations_in_dataset.update(locations)
        
        all_unique_locations.update(locations_in_dataset)
        dataset_locations[idx] = locations_in_dataset
    
    # Process baseline data
    baseline_locations = set()
    if baseline_data:
        for event in baseline_data:
            locations = event.get('locations', [])
            
            # Apply the same processing as for regular datasets
            if isinstance(locations, str) and locations.startswith("```tool_code\nprint(default_api.send_output(locations="):
                locations_str = locations.strip("```tool_code\nprint(default_api.send_output(locations=\n").strip()
                if locations_str.endswith(","):
                    locations_str = locations_str[:-1]
                if not locations_str.endswith("'") and locations_str[1] == "'":
                    locations_str += "'"
                elif not locations_str.endswith('"') and locations_str[1] == '"':
                    locations_str += '"'
                if locations_str.endswith(", '"):
                    locations_str = locations_str[:-2]
                if not locations_str.endswith("]"):
                    locations_str += "]"
                    
                try:
                    locations = eval(locations_str)
                except Exception as e:
                    print(f"Error evaluating baseline locations: {e}")
                    locations = []
            elif locations is None:
                locations = []
            
            baseline_locations.update(locations)
        
        all_unique_locations.update(baseline_locations)
        dataset_locations['baseline'] = baseline_locations
    
    # Validate all unique locations
    print(f"Validating {len(all_unique_locations)} unique locations...")
    location_validity = {}
    
    # Process in batches to avoid potential API limitations
    batch_size = 500
    location_list = list(all_unique_locations)
    
    for i in range(0, len(location_list), batch_size):
        batch = location_list[i:i+batch_size]
        try:
            batch_validity = get_location_validity(batch, source_text)
            location_validity.update(batch_validity)
            print(f"Validated batch {i//batch_size + 1}/{(len(location_list) + batch_size - 1)//batch_size}")
        except Exception as e:
            print(f"Error validating locations batch {i//batch_size + 1}: {e}")
            # If batch fails, try one by one
            for loc in batch:
                try:
                    result = get_location_validity([loc], source_text)
                    location_validity.update(result)
                except:
                    print(f"Failed to validate: {loc}")
                    location_validity[loc] = {"is_valid_location": False, "exists_in_text": False}
    
    # Count locations per event type for all datasets
    event_counts = {}
    for idx, data in datasets.items():
        event_counts[idx] = count_locations_per_event(data)
    
    # Count locations per event type for baseline
    if baseline_data:
        event_counts['baseline'] = count_locations_per_event(baseline_data)
    
    # Get all unique event types across all datasets
    all_event_types = set()
    for counts in event_counts.values():
        all_event_types.update(counts.keys())
    all_event_types = sorted(all_event_types)
    
    # Create comparison DataFrame
    comparison_data = {
        'Event Type': all_event_types
    }
    
    # Add columns for each dataset
    for idx in indices:
        if idx in event_counts:
            comparison_data[f'Version {idx}'] = [event_counts[idx].get(et, 0) for et in all_event_types]
    
    # Add baseline column if available
    if 'baseline' in event_counts:
        comparison_data['Baseline'] = [event_counts['baseline'].get(et, 0) for et in all_event_types]
    
    df = pd.DataFrame(comparison_data)
    
    # Add total row
    totals = {'Event Type': 'TOTAL'}
    for idx in indices:
        if idx in event_counts:
            column_name = f'Version {idx}'
            if column_name in df.columns:
                totals[column_name] = df[column_name].sum()
    
    # Add baseline total if available
    if 'baseline' in event_counts and 'Baseline' in df.columns:
        totals['Baseline'] = df['Baseline'].sum()
    
    df_totals = pd.DataFrame([totals])
    df = pd.concat([df, df_totals], ignore_index=True)
    
    # Print results
    print("\nLocation Count Comparison Across Versions:")
    print("=" * 100)
    print(df.to_string(index=False))
    
    # Create a summary table with totals, valid locations, and improvements
    version_totals = {idx: totals[f'Version {idx}'] for idx in indices if f'Version {idx}' in totals}
    
    # Add baseline to version_totals if available
    baseline_total = None
    if 'baseline' in event_counts and 'Baseline' in totals:
        baseline_total = totals['Baseline']
        version_totals['baseline'] = baseline_total
    
    if version_totals:
        # Only consider regular versions (not baseline) for determining min_total
        regular_versions = {k: v for k, v in version_totals.items() if k != 'baseline'}
        min_total = min(regular_versions.values()) if regular_versions else 0
        
        # Count locations with different validity status for each dataset
        valid_locations = {}
        in_text_locations = {}
        both_valid_locations = {}
        
        for idx, locations in dataset_locations.items():
            valid_count = sum(1 for loc in locations if location_validity.get(loc, {"is_valid_location": False})["is_valid_location"])
            in_text_count = sum(1 for loc in locations if location_validity.get(loc, {"exists_in_text": False})["exists_in_text"])
            both_valid_count = sum(1 for loc in locations if 
                                  location_validity.get(loc, {"is_valid_location": False, "exists_in_text": False})["is_valid_location"] and
                                  location_validity.get(loc, {"is_valid_location": False, "exists_in_text": False})["exists_in_text"])
            
            valid_locations[idx] = valid_count
            in_text_locations[idx] = in_text_count
            both_valid_locations[idx] = both_valid_count
        
        # Create the summary dataframe with expanded metrics
        summary_data = {
            'Version': [],
            'Distinct Locations': [],
            'Valid Locations': [],
            'Valid %': [],
            'In Text': [],
            'In Text %': [],
            'Both Valid': [],
            'Both Valid %': [],
            '% Improvement Over Min': []
        }
        
        # Add regular versions first
        for idx in [i for i in version_totals.keys() if i != 'baseline']:
            summary_data['Version'].append(idx)
            summary_data['Distinct Locations'].append(version_totals[idx])
            summary_data['Valid Locations'].append(valid_locations.get(idx, 0))
            summary_data['Valid %'].append(round(valid_locations.get(idx, 0) / version_totals[idx] * 100, 2) if version_totals[idx] > 0 else 0)
            summary_data['In Text'].append(in_text_locations.get(idx, 0))
            summary_data['In Text %'].append(round(in_text_locations.get(idx, 0) / version_totals[idx] * 100, 2) if version_totals[idx] > 0 else 0)
            summary_data['Both Valid'].append(both_valid_locations.get(idx, 0))
            summary_data['Both Valid %'].append(round(both_valid_locations.get(idx, 0) / version_totals[idx] * 100, 2) if version_totals[idx] > 0 else 0)
            summary_data['% Improvement Over Min'].append(round((version_totals[idx] - min_total) / min_total * 100, 2) if min_total > 0 else 0)
        
        # Add baseline if available
        if 'baseline' in version_totals:
            summary_data['Version'].append('baseline')
            summary_data['Distinct Locations'].append(version_totals['baseline'])
            summary_data['Valid Locations'].append(valid_locations.get('baseline', 0))
            summary_data['Valid %'].append(round(valid_locations.get('baseline', 0) / version_totals['baseline'] * 100, 2) if version_totals['baseline'] > 0 else 0)
            summary_data['In Text'].append(in_text_locations.get('baseline', 0))
            summary_data['In Text %'].append(round(in_text_locations.get('baseline', 0) / version_totals['baseline'] * 100, 2) if version_totals['baseline'] > 0 else 0)
            summary_data['Both Valid'].append(both_valid_locations.get('baseline', 0))
            summary_data['Both Valid %'].append(round(both_valid_locations.get('baseline', 0) / version_totals['baseline'] * 100, 2) if version_totals['baseline'] > 0 else 0)
            # Calculate improvement comparing to min_total from regular versions
            summary_data['% Improvement Over Min'].append(round((version_totals['baseline'] - min_total) / min_total * 100, 2) if min_total > 0 else 0)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by Version but keep 'baseline' at the end
        if 'baseline' in version_totals:
            # Separate baseline row
            baseline_row = summary_df[summary_df['Version'] == 'baseline']
            # Sort other rows
            other_rows = summary_df[summary_df['Version'] != 'baseline'].sort_values('Version')
            # Combine with baseline at the end
            summary_df = pd.concat([other_rows, baseline_row], ignore_index=True)
        else:
            summary_df = summary_df.sort_values('Version')
        
        print("\nSummary of Distinct Locations and Improvements:")
        print("=" * 120)
        print(summary_df.to_string(index=False))
    
        # Prepare data for plotting
        baseline_plot_data = None
        if 'baseline' in version_totals:
            # Extract baseline row for special plotting
            baseline_row = summary_df[summary_df['Version'] == 'baseline'].iloc[0]
            baseline_plot_data = (BASELINE_COST, baseline_row['Distinct Locations'], baseline_row['Valid %'])
            
            # Remove baseline from regular plot data
            plot_df = summary_df[summary_df['Version'] != 'baseline']
        else:
            plot_df = summary_df
        
        # Get plot data for regular versions
        plot_costs = [PLAN_COSTS[int(idx)] for idx in plot_df['Version']]
        plot_distinct = plot_df['Distinct Locations'].tolist()
        plot_valid_pct = plot_df['Valid %'].tolist()
        
        # Create and save plot
        plot_path = base_path / "plans" / "cost_vs_locations_plot.png"
        plot_cost_vs_metrics(plot_costs, plot_distinct, plot_valid_pct, baseline=baseline_plot_data, output_path=plot_path)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 120)
    print(f"Number of Distinct Event Types: {len(all_event_types)}")
    print(f"Total Unique Locations Across All Versions: {len(all_unique_locations)}")
    
    valid_count = sum(1 for loc, validity in location_validity.items() if validity["is_valid_location"])
    in_text_count = sum(1 for loc, validity in location_validity.items() if validity["exists_in_text"])
    both_valid_count = sum(1 for loc, validity in location_validity.items() 
                         if validity["is_valid_location"] and validity["exists_in_text"])
    
    print(f"Valid Locations: {valid_count} ({valid_count/len(all_unique_locations)*100:.2f}%)")
    print(f"Locations Found in Text: {in_text_count} ({in_text_count/len(all_unique_locations)*100:.2f}%)")
    print(f"Locations Both Valid and Found in Text: {both_valid_count} ({both_valid_count/len(all_unique_locations)*100:.2f}%)")

if __name__ == "__main__":      
    main()
