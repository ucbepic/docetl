import json
import pandas as pd
from typing import List, Dict


def load_json(file_path: str) -> List[Dict]:
    with open(file_path, "r") as f:
        return json.load(f)


def calculate_rp_at_k(extracted: List[str], ground_truth: List[str], k: int) -> float:
    ground_truth = [gt.lower() for gt in ground_truth]
    relevant = sum(1 for item in extracted[:k] if item.lower() in ground_truth)
    return relevant / min(k, len(ground_truth))


def main():
    # Load extracted reactions
    data = load_json(
        "/Users/shreyashankar/Documents/hacking/motion-v3/paper_workloads/biodex/extracted_reactions_pipeline.json"
    )
    labels = load_json(
        "/Users/shreyashankar/Documents/hacking/motion-v3/paper_workloads/biodex/biodex_ground_truth.json"
    )

    # Convert to DataFrame
    df = pd.DataFrame(data)
    labels_df = pd.DataFrame(labels)
    df = df.merge(labels_df, on="id")
    k_values = [5, 10]

    # Calculate RP@5 and RP@10 for each group
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

    # Calculate average RP@k for each k
    for k in k_values:
        avg_rp_at_k = sum(item[f"RP@{k}"] for item in results) / len(results)
        print(f"Average RP@{k}: {avg_rp_at_k}")


if __name__ == "__main__":
    main()
