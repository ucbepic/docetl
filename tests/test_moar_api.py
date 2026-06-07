"""
Test for Pipeline.moar_optimize() with a toy sentiment analysis task.

Run with:
    python tests/test_moar_api.py
"""

import json
import os
import tempfile

from docetl.api import (
    Dataset,
    MapOp,
    Pipeline,
    PipelineOutput,
    PipelineStep,
)

# --- Toy dataset with ground-truth labels ---
TOY_DATA = [
    {"text": "I absolutely love this product, it changed my life!", "GT sentiment": "positive"},
    {"text": "Terrible experience, would not recommend to anyone.", "GT sentiment": "negative"},
    {"text": "It was okay, nothing special.", "GT sentiment": "neutral"},
    {"text": "Best purchase I have ever made, truly amazing!", "GT sentiment": "positive"},
    {"text": "Awful quality, broke after one day.", "GT sentiment": "negative"},
    {"text": "The item works as described.", "GT sentiment": "neutral"},
    {"text": "Wonderful! Exceeded all my expectations.", "GT sentiment": "positive"},
    {"text": "Complete waste of money.", "GT sentiment": "negative"},
    {"text": "It does the job, not great not terrible.", "GT sentiment": "neutral"},
    {"text": "So happy with this, highly recommend!", "GT sentiment": "positive"},
]


def evaluate_sentiment(dataset_file_path: str, results_file_path: str) -> dict:
    """Simple evaluation: compare predicted sentiment to ground truth."""
    with open(dataset_file_path, "r") as f:
        original = json.load(f)
    with open(results_file_path, "r") as f:
        results = json.load(f)

    correct = 0
    total = min(len(original), len(results))
    for orig, res in zip(original[:total], results[:total]):
        gt = orig.get("GT sentiment", "").strip().lower()
        pred = res.get("sentiment", "").strip().lower()
        if gt == pred:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total}


def main():
    # Write toy dataset to a temp file
    tmp_dir = tempfile.mkdtemp(prefix="moar_test_")
    dataset_path = os.path.join(tmp_dir, "toy_sentiment.json")
    output_path = os.path.join(tmp_dir, "output.json")
    save_dir = os.path.join(tmp_dir, "moar_results")

    with open(dataset_path, "w") as f:
        json.dump(TOY_DATA, f, indent=2)

    print(f"Dataset: {dataset_path}")
    print(f"Save dir: {save_dir}")

    # Build pipeline
    pipeline = Pipeline(
        name="toy_sentiment",
        datasets={
            "reviews": Dataset(type="file", path=dataset_path),
        },
        operations=[
            MapOp(
                name="classify_sentiment",
                type="map",
                prompt=(
                    "Classify the sentiment of the following text as "
                    "'positive', 'negative', or 'neutral'.\n\n"
                    "Text: {{ input.text }}"
                ),
                output={"schema": {"sentiment": "string"}},
            ),
        ],
        steps=[
            PipelineStep(
                name="classify_step",
                input="reviews",
                operations=["classify_sentiment"],
            ),
        ],
        output=PipelineOutput(
            type="file",
            path=output_path,
            intermediate_dir=os.path.join(tmp_dir, "intermediate"),
        ),
        default_model="azure/gpt-4o-mini",
    )

    # Run MOAR optimization
    results = pipeline.moar_optimize(
        evaluate_func=evaluate_sentiment,
        metric_key="accuracy",
        available_models=["azure/gpt-4o-mini", "azure/gpt-4o"],
        max_iterations=5,
        model="azure/gpt-4o-mini",
        save_dir=save_dir,
    )

    print("\n=== MOAR Results ===")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
