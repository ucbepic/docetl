#!/usr/bin/env python3
"""
Script to compute CUAD accuracy for baseline.json output
"""

from experiments.reasoning.evaluation.cuad import evaluate_results as cuad_evaluate
from pathlib import Path

# Path to the baseline results
baseline_results = "experiments/reasoning/outputs/cuad_baseline/original_output.json"

# Path to ground truth CSV
ground_truth_path = "experiments/reasoning/data/CUAD-master_clauses.csv"

# Compute metrics
print("Computing CUAD accuracy metrics...")
metrics = cuad_evaluate("docetl_baseline", baseline_results, ground_truth_path)

print("\nResults:")
print(f"Average Precision: {metrics['avg_precision']:.4f}")
print(f"Average Recall: {metrics['avg_recall']:.4f}")
print(f"Average F1: {metrics['avg_f1']:.4f}")
print(f"NaN Fraction: {metrics['nan_fraction']:.4f}")
print(f"Average Clause Length: {metrics['avg_clause_length']:.2f}")

print("\nPer-metric breakdown:")
for metric, values in metrics['per_metric'].items():
    if not (values['precision'] != values['precision'] or values['recall'] != values['recall']):  # Check for NaN
        f1 = 2 * values['precision'] * values['recall'] / (values['precision'] + values['recall']) if (values['precision'] + values['recall']) > 0 else 0
        print(f"{metric}: P={values['precision']:.3f}, R={values['recall']:.3f}, F1={f1:.3f}")

# -----------------------------------------------------------------------------
# Evaluate CUAD metrics for document compression output
# -----------------------------------------------------------------------------

# Path to document compression results
doc_compression_results = "experiments/reasoning/outputs/cuad/cuad_1.json"

print("\n\nComputing CUAD accuracy metrics for document compression output...")
doc_metrics = cuad_evaluate("docetl_doc_compression", doc_compression_results, ground_truth_path)

print("\nResults (Doc Compression):")
print(f"Average Precision: {doc_metrics['avg_precision']:.4f}")
print(f"Average Recall: {doc_metrics['avg_recall']:.4f}")
print(f"Average F1: {doc_metrics['avg_f1']:.4f}")
print(f"NaN Fraction: {doc_metrics['nan_fraction']:.4f}")
print(f"Average Clause Length: {doc_metrics['avg_clause_length']:.2f}")

print("\nPer-metric breakdown (Doc Compression):")
for metric, values in doc_metrics['per_metric'].items():
    # Skip metrics with NaN precision/recall
    if not (values['precision'] != values['precision'] or values['recall'] != values['recall']):
        f1 = 2 * values['precision'] * values['recall'] / (values['precision'] + values['recall']) if (values['precision'] + values['recall']) > 0 else 0
        print(f"{metric}: P={values['precision']:.3f}, R={values['recall']:.3f}, F1={f1:.3f}")
