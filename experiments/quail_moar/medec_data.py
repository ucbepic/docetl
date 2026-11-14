"""Build MEDEC train and test files for the Quail MOAR experiment.

MEDEC (https://github.com/abachaa/MEDEC) holds clinical notes, some with one
medical error. Each row keeps its ground truth (Error Flag, Error Sentence,
Corrected Sentence), which experiments/reasoning/evaluation/medec.py reads
from the pipeline's output rows.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

REPO = "https://github.com/abachaa/MEDEC"
COLUMNS = ["Text ID", "Text", "Error Flag", "Error Sentence", "Corrected Sentence"]


def _rows(frame: pd.DataFrame) -> list[dict]:
    frame = frame[COLUMNS].fillna("")
    frame["Error Flag"] = frame["Error Flag"].astype(int)
    return frame.to_dict(orient="records")


def build(out_dir: str, n_train: int = 100, n_test: int = 200, seed: int = 0) -> tuple[str, str]:
    """Write medec_train.json and medec_test.json into ``out_dir``; return their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "-q", "--depth", "1", REPO, tmp], check=True)
        base = Path(tmp) / "MEDEC-MS"
        train = pd.read_csv(base / "MEDEC-Full-TrainingSet-with-ErrorType.csv")
        test = pd.read_csv(base / "MEDEC-MS-TestSet-with-GroundTruth-and-ErrorType.csv")
    train = train[train["Text ID"].str.startswith("ms-")]
    # Error Flag is "0"/"1" text in some files; drop rows without one.
    for frame in (train, test):
        frame["Error Flag"] = pd.to_numeric(frame["Error Flag"], errors="coerce")
    train = train.dropna(subset=["Error Flag", "Text"])
    test = test.dropna(subset=["Error Flag", "Text"])
    # Balanced samples: about half the notes have an error.
    def balanced(frame, n):
        flags = frame["Error Flag"].astype(int)
        half = n // 2
        return pd.concat(
            [frame[flags == 1].sample(n - half, random_state=seed), frame[flags == 0].sample(half, random_state=seed)]
        ).sample(frac=1, random_state=seed)
    paths = []
    for name, frame, n in (("medec_train.json", train, n_train), ("medec_test.json", test, n_test)):
        path = out / name
        path.write_text(json.dumps(_rows(balanced(frame, n)), indent=1))
        paths.append(str(path))
    return paths[0], paths[1]


if __name__ == "__main__":
    import sys

    print(build(sys.argv[1] if len(sys.argv) > 1 else "data"))
