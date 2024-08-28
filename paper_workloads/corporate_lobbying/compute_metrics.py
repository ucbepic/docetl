import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def compute_metrics(ground_truth_path, predictions_path):
    ground_truth = load_json(ground_truth_path)
    predictions = load_json(predictions_path)

    # Create a dictionary of ground truth items for easy lookup
    ground_truth_dict = {item["id"]: item for item in ground_truth}

    # Filter ground truth to match predictions and ensure order
    filtered_ground_truth = [
        ground_truth_dict[pred["id"]]
        for pred in predictions
        if pred["id"] in ground_truth_dict
    ]

    y_true = [1 if item["answer"] == "Yes" else 0 for item in filtered_ground_truth]
    y_pred = [1 if item["is_relevant"] else 0 for item in predictions]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


# Compute metrics for baseline
baseline_metrics = compute_metrics(
    "/Users/shreyashankar/Documents/hacking/motion-v3/paper_workloads/corporate_lobbying/ground_truth.json",
    "/Users/shreyashankar/Documents/hacking/motion-v3/paper_workloads/corporate_lobbying/relevance_assessment_baseline.json",
)

# Compute metrics for tool
tool_metrics = compute_metrics(
    "/Users/shreyashankar/Documents/hacking/motion-v3/paper_workloads/corporate_lobbying/ground_truth.json",
    "/Users/shreyashankar/Documents/hacking/motion-v3/paper_workloads/corporate_lobbying/relevance_assessment_tool.json",
)

# Prepare data for the DataFrame
data = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Baseline": [
        f"{baseline_metrics['accuracy']:.4f}",
        f"{baseline_metrics['precision']:.4f}",
        f"{baseline_metrics['recall']:.4f}",
        f"{baseline_metrics['f1_score']:.4f}",
    ],
    "Tool": [
        f"{tool_metrics['accuracy']:.4f}",
        f"{tool_metrics['precision']:.4f}",
        f"{tool_metrics['recall']:.4f}",
        f"{tool_metrics['f1_score']:.4f}",
    ],
}

# Create DataFrame
df = pd.DataFrame(data)

# Print the DataFrame
print(df.to_string(index=False))
