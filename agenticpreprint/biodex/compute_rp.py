import json
import random
import pandas as pd
from typing import List, Dict, Tuple
from tabulate import tabulate


def load_json(file_path: str) -> List[Dict]:
    with open(file_path, "r") as f:
        return json.load(f)


def calculate_rp_at_k(extracted: List[str], ground_truth: List[str], k: int) -> float:
    ground_truth = [gt.lower() for gt in ground_truth]
    relevant = sum(1 for item in extracted[:k] if item.lower() in ground_truth)
    return relevant / min(k, len(ground_truth))


def process_lotus_data(csv_path: str, labels_path: str, k_values: List[int]) -> List[Dict]:
    # Load data
    df = pd.read_csv(csv_path)
    labels = load_json(labels_path)
    labels_df = pd.DataFrame(labels)
    
    # Process each group
    results = []
    for id_, group in df.groupby('id'):
        # Sort by scores and get top reactions
        sorted_group = group.sort_values('_scores', ascending=False)
        extracted = sorted_group['reaction'].tolist()
        
        # Get ground truth for this ID
        try:
            ground_truth = labels_df[labels_df['id'] == id_]['ground_truth_reactions'].iloc[0]
            
            # Calculate RP@k
            result = {"id": id_}
            for k in k_values:
                rp_at_k = calculate_rp_at_k(extracted, ground_truth, k)
                result[f"RP@{k}"] = rp_at_k
            results.append(result)
        except (IndexError, KeyError) as e:
            print(f"Warning: Could not find ground truth for ID {id_}")
            continue
    
    return results


def process_docetl_file(file_path: str, labels_path: str, k_values: List[int]) -> List[Dict]:
    """Process a single DocETL file and calculate RP@k values"""
    # Load data
    data = load_json(file_path)
    labels = load_json(labels_path)

    # Convert to DataFrame
    df = pd.DataFrame(data)
    labels_df = pd.DataFrame(labels)
    df = df.merge(labels_df, on="id")

    # Calculate RP@k for each group
    results = []
    for _, row in df.iterrows():
        id_ = row["id"]
        extracted = row["ranked_conditions"]
        ground_truth = row["ground_truth_reactions"]

        result = {"id": id_}
        for k in k_values:
            rp_at_k = calculate_rp_at_k(extracted, ground_truth, k)
            result[f"RP@{k}"] = rp_at_k
        results.append(result)
    
    return results


def main():
    k_values = [5, 10, 25]
    ground_truth_path = "agenticpreprint/biodex/biodex_ground_truth.json"
    
    # Initialize a list to collect all results for table display
    table_data = []
    
    # Process multiple pipeline results
    print("Processing pipeline results...")
    docetl_files = [
        "agenticpreprint/biodex/results/extracted_reactions_docetl_0.json",
        "agenticpreprint/biodex/results/extracted_reactions_docetl_1.json",
        "agenticpreprint/biodex/results/extracted_reactions_docetl_2.json",
        "agenticpreprint/biodex/results/extracted_reactions_docetl_3.json",
        "agenticpreprint/biodex/results/extracted_reactions_docetl_4.json",
    ]
    
    all_pipeline_results = []
    for file_path in docetl_files:
        print(f"Processing {file_path}...")
        results = process_docetl_file(file_path, ground_truth_path, k_values)
        all_pipeline_results.append(results)
    
    # Process additional historical and recent files (but don't include in averages)
    print("\nProcessing additional files (not included in averages)...")
    additional_files = [
        "agenticpreprint/biodex/results/extracted_reactions_docetl_aug2024.json",
        "agenticpreprint/biodex/results/extracted_reactions_docetl_jan2025.json",
    ]
    
    print("\nAdditional File Results (not included in averages):")
    print("-" * 40)
    for file_path in additional_files:
        try:
            print(f"Processing {file_path}...")
            results = process_docetl_file(file_path, ground_truth_path, k_values)
            
            file_name = file_path.split("/")[-1].replace("extracted_reactions_", "").replace(".json", "")
            print(f"{file_name} Results:")
            
            # Calculate averages for table
            row_data = {"Model": file_name}
            for k in k_values:
                avg_rp_at_k = sum(item[f"RP@{k}"] for item in results) / len(results)
                row_data[f"RP@{k}"] = avg_rp_at_k
                print(f"  Average RP@{k}: {avg_rp_at_k:.4f}")
            table_data.append(row_data)
            print("-" * 40)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            print("-" * 40)
    
    # Calculate and display averages for each pipeline
    print("\nIndividual DocETL Results:")
    print("-" * 40)
    for i, results in enumerate(all_pipeline_results):
        print(f"DocETL_{i} Results:")
        
        # Calculate averages for table
        row_data = {"Model": f"DocETL_{i}"}
        for k in k_values:
            avg_rp_at_k = sum(item[f"RP@{k}"] for item in results) / len(results)
            row_data[f"RP@{k}"] = avg_rp_at_k
            print(f"  Average RP@{k}: {avg_rp_at_k:.4f}")
        table_data.append(row_data)
        print("-" * 40)
    
    # Calculate and display combined averages across all pipelines
    print("\nCombined DocETL Results:")
    print("=" * 40)
    combined_results = [item for sublist in all_pipeline_results for item in sublist]
    
    # Calculate averages for table
    row_data = {"Model": "Combined DocETL (3-14-2025 plans only)"}
    for k in k_values:
        avg_rp_at_k = sum(item[f"RP@{k}"] for item in combined_results) / len(combined_results)
        row_data[f"RP@{k}"] = avg_rp_at_k
        print(f"Average RP@{k}: {avg_rp_at_k:.4f}")
    table_data.append(row_data)
    print("=" * 40)

    # Process Lotus results
    print("\nProcessing Lotus results...")
    try:
        lotus_results = process_lotus_data(
            "agenticpreprint/biodex/results/lotus_output.csv",
            ground_truth_path,
            k_values
        )

        # Print Lotus results
        print("\nLotus Results:")
        
        # Calculate averages for table
        row_data = {"Model": "Lotus"}
        for k in k_values:
            avg_rp_at_k = sum(item[f"RP@{k}"] for item in lotus_results) / len(lotus_results)
            row_data[f"RP@{k}"] = avg_rp_at_k
            print(f"Average RP@{k}: {avg_rp_at_k:.4f}")
        table_data.append(row_data)
    except Exception as e:
        print(f"Error processing Lotus results: {e}")
        print(f"Error details: {str(e)}")
    
    # Process Baseline results
    print("\nProcessing Baseline results...")
    try:
        baseline_results = process_lotus_data(
            "agenticpreprint/biodex/results/baseline_output.csv",
            ground_truth_path,
            k_values
        )

        # Print Baseline results
        print("\nBaseline Results:")
        
        # Calculate averages for table
        row_data = {"Model": "Baseline (Keyword Matching)"}
        for k in k_values:
            if baseline_results:
                avg_rp_at_k = sum(item[f"RP@{k}"] for item in baseline_results) / len(baseline_results)
                row_data[f"RP@{k}"] = avg_rp_at_k
                print(f"Average RP@{k}: {avg_rp_at_k:.4f}")
            else:
                row_data[f"RP@{k}"] = 0.0
                print(f"Average RP@{k}: 0.0000 (No valid results)")
        table_data.append(row_data)
    except Exception as e:
        print(f"Error processing Baseline results: {e}")
        print(f"Error details: {str(e)}")

    # Print pretty table with all results
    print("\n\n")
    print("=" * 80)
    print("SUMMARY OF ALL RESULTS")
    print("=" * 80)
    
    # Sort table data for better readability
    sorted_table_data = sorted(table_data, key=lambda x: x["Model"])
    
    # Group data into categories
    docetl_base = [row for row in sorted_table_data if row["Model"].startswith("DocETL_")]
    docetl_combined = [row for row in sorted_table_data if "Combined" in row["Model"]]
    lotus = [row for row in sorted_table_data if row["Model"] == "Lotus"]
    baseline = [row for row in sorted_table_data if "Baseline" in row["Model"]]
    historical = [row for row in sorted_table_data if "docetl_aug2024" in row["Model"] or "docetl_jan2025" in row["Model"]]
    optimized_new = [row for row in sorted_table_data if "docet_base_opt" in row["Model"]]
    
    # Reorder table data by category
    ordered_table_data = docetl_base + docetl_combined + lotus + baseline + historical + optimized_new
    
    try:
        # Try using tabulate for a nice table
        headers = {"Model": "Model"}
        for k in k_values:
            headers[f"RP@{k}"] = f"RP@{k}"
        
        print(tabulate(ordered_table_data, headers=headers, tablefmt="grid", floatfmt=".4f"))
        
        # Add clarification note
        print("\nNote: 'Combined DocETL' average is calculated only from the DocETL_0 through DocETL_3 plans from 3-14-2025.")
        print("The pipeline_aug2024, pipeline_jan2025, and docet_base_opt_* plans are shown for comparison but not included in the average.")
    except ImportError:
        # Fallback if tabulate is not available
        print(f"{'Model':<40} " + " ".join([f"RP@{k:<8}" for k in k_values]))
        print("-" * 70)
        for row in ordered_table_data:
            model_name = row["Model"]
            metrics = " ".join([f"{row[f'RP@{k}']:.4f}  " for k in k_values])
            print(f"{model_name:<40} {metrics}")
        
        # Add clarification note
        print("\nNote: 'Combined DocETL' average is calculated only from the DocETL_0 through DocETL_3 plans from 3-14-2025.")
        print("The pipeline_aug2024, pipeline_jan2025, and docet_base_opt_* plans are shown for comparison but not included in the average.")
    
    print("=" * 80)


if __name__ == "__main__":
    main()