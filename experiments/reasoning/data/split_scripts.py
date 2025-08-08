from __future__ import annotations

"""Utility to create disjoint train/test splits (25/100) for multiple datasets.

For every dataset we:
1. Load the raw data file (JSON or CSV).
2. Rank records by a length heuristic (character count of a chosen text field).
3. Keep the longest 50 %.
4. Randomly choose *train_n* and *test_n* disjoint items.
5. Write the splits to ``data/train/<dataset>.json`` and ``data/test/<dataset>.json``.

This script is intentionally self-contained and has no CLI arguments –
executing ``python split_scripts.py`` will generate all splits.
"""

import csv
import json
import random
from pathlib import Path
from typing import Any, Callable, Sequence

import datasets
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # <repo_root>/…/split_scripts.py → 3 levels up
DATA_DIR = PROJECT_ROOT / "experiments/reasoning/data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
TRAIN_N = 50
TEST_N = 100
SAMPLE_FRAC = 0.75
SEED = 42

# Ensure deterministic sampling
random.seed(SEED)


# ---------------------------------------------------------------------------
# Helper functions (≤20 LOC each, no blank lines inside)
# ---------------------------------------------------------------------------

def _ensure_dirs() -> None:
    for d in (TRAIN_DIR, TEST_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _rank_and_sample(records: list[dict[str, Any]], length_fn: Callable[[dict[str, Any]], int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scored = [(length_fn(r), idx) for idx, r in enumerate(records)]
    scored.sort(key=lambda t: t[0], reverse=True)
    cut = int(len(scored) * SAMPLE_FRAC)
    top_idxs = [idx for _, idx in scored[:cut]]
    if len(top_idxs) < TRAIN_N + TEST_N:
        raise ValueError(f"Not enough records in top {SAMPLE_FRAC:.0%} to sample {TRAIN_N + TEST_N} items (got {len(top_idxs)})")
    chosen = random.sample(top_idxs, k=TRAIN_N + TEST_N)
    train_records = [records[i] for i in chosen[:TRAIN_N]]
    test_records = [records[i] for i in chosen[TRAIN_N:]]
    return train_records, test_records


def _write_json(path: Path, data: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(data), f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Dataset-specific splitters
# ---------------------------------------------------------------------------

def _split_blackvault() -> None:
    src = PROJECT_ROOT / "experiments/reasoning/data/blackvault_articles_pdfs_full.json"
    with src.open("r", encoding="utf-8") as f:
        records: list[dict[str, Any]] = json.load(f)
    length_fn = lambda r: len(str(r.get("article_text") or r.get("text") or ""))
    train, test = _rank_and_sample(records, length_fn)
    _write_json(TRAIN_DIR / "blackvault.json", train)
    _write_json(TEST_DIR / "blackvault.json", test)


def _split_reviews() -> None:
    src = PROJECT_ROOT / "experiments/reasoning/data/reviews_full.json"
    with src.open("r", encoding="utf-8") as f:
        records: list[dict[str, Any]] = json.load(f)
    length_fn = lambda r: len(str(r.get("review_text") or r.get("text") or r.get("content") or ""))
    train, test = _rank_and_sample(records, length_fn)
    _write_json(TRAIN_DIR / "reviews.json", train)
    _write_json(TEST_DIR / "reviews.json", test)


def _split_sustainability() -> None:
    from workloads.sustainability.clean_data import clean_dataframe  # Local import to avoid overhead if unused
    src = PROJECT_ROOT / "workloads/sustainability/company_reports_2024_01_22.json"
    df_raw = pd.read_json(src)
    df_clean = clean_dataframe(df_raw)
    records = df_clean.to_dict(orient="records")
    length_fn = lambda r: len(str(r.get("tot_text_raw") or ""))
    train, test = _rank_and_sample(records, length_fn)
    _write_json(TRAIN_DIR / "sustainability.json", train)
    _write_json(TEST_DIR / "sustainability.json", test)


def _split_medec() -> None:
    src = PROJECT_ROOT / "workloads/medec/MEDEC-MS-ValidationSet-with-GroundTruth-and-ErrorType.csv"
    df = pd.read_csv(src, dtype=str, quoting=csv.QUOTE_MINIMAL)
    records = df.to_dict(orient="records")
    length_fn = lambda r: len(str(r.get("Sentences") or r.get("Text") or ""))
    train, test = _rank_and_sample(records, length_fn)
    _write_json(TRAIN_DIR / "medec.json", train)
    _write_json(TEST_DIR / "medec.json", test)


def _split_cuad() -> None:
    src = PROJECT_ROOT / "preprint_workloads/stylized/raw.json"
    with src.open("r", encoding="utf-8") as f:
        records: list[dict[str, Any]] = json.load(f)
    length_fn = lambda r: len(str(r.get("text") or r.get("content") or ""))
    train, test = _rank_and_sample(records, length_fn)
    _write_json(TRAIN_DIR / "cuad.json", train)
    _write_json(TEST_DIR / "cuad.json", test)


def _split_biodex() -> None:
    # Load labels from file
    labels_file = PROJECT_ROOT / "experiments/reasoning/data/biodex_labels.txt"
    with labels_file.open("r", encoding="utf-8") as f:
        all_labels = f.read()
    # Load BioDEX dataset from HuggingFace
    test_dataset = datasets.load_dataset("BioDEX/BioDEX-Reactions", split="test").to_pandas()
    # Convert to records
    records = test_dataset.to_dict(orient="records")
    # Process each record to have the required format
    processed_records = []
    for r in records:
        reactions_lst = [
            reaction.strip().lower().replace("'", "").replace("^", "")
            for reaction in r.get("reactions", "").split(",")
        ]
        processed_record = {
            "id": r.get("pmid"),
            "fulltext_processed": r.get("fulltext_processed", ""),
            "ground_truth_reactions": reactions_lst,
            "possible_labels": all_labels
        }
        processed_records.append(processed_record)
    # Use fulltext_processed for length-based ranking
    length_fn = lambda r: len(str(r.get("fulltext_processed") or ""))
    train, test = _rank_and_sample(processed_records, length_fn)
    _write_json(TRAIN_DIR / "biodex.json", train)
    _write_json(TEST_DIR / "biodex.json", test)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _ensure_dirs()
    _split_blackvault()
    _split_reviews()
    _split_sustainability()
    _split_medec()
    _split_cuad()
    _split_biodex()
    print(f"Done. Splits written to {TRAIN_DIR} and {TEST_DIR}.")


if __name__ == "__main__":
    main()
