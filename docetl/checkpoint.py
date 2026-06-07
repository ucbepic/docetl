"""Checkpoint store for intermediate pipeline results.

Handles saving, loading, and validating operation checkpoints so that
pipeline reruns can skip already-completed operations.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any


class CheckpointStore:
    """Manages intermediate checkpoints for pipeline operations.

    Each checkpoint is a JSON file at ``<base_dir>/<step>/<op>.json``.
    A config file at ``<base_dir>/.docetl_intermediate_config.json`` maps
    ``step → op → hash`` so stale checkpoints (from changed configs) are
    not reused.
    """

    def __init__(self, base_dir: str, op_hashes: dict[str, dict[str, str]],
                 bypass: bool = False):
        self.base_dir = base_dir
        self.op_hashes = op_hashes
        self.bypass = bypass

    @property
    def _config_path(self) -> str:
        return os.path.join(self.base_dir, ".docetl_intermediate_config.json")

    def load(self, step_name: str, op_name: str) -> list[dict] | None:
        """Return cached output for *step_name/op_name*, or ``None``."""
        if self.bypass:
            return None

        if not os.path.exists(self._config_path):
            return None

        if (
            step_name not in self.op_hashes
            or op_name not in self.op_hashes[step_name]
        ):
            return None

        with open(self._config_path, "r") as f:
            saved = json.load(f)

        if (
            saved.get(step_name, {}).get(op_name, "")
            != self.op_hashes[step_name][op_name]
        ):
            return None

        path = os.path.join(self.base_dir, step_name, f"{op_name}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return None

    def save(self, step_name: str, op_name: str, data: list[dict]) -> str:
        """Write checkpoint and update the config. Returns the file path."""
        path = os.path.join(self.base_dir, step_name, f"{op_name}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

        # Update config file with the current hash
        if os.path.exists(self._config_path):
            try:
                with open(self._config_path, "r") as f:
                    cfg: dict[str, dict[str, str]] = json.load(f)
            except json.JSONDecodeError:
                cfg = {}
        else:
            cfg = {}

        cfg.setdefault(step_name, {})[op_name] = (
            self.op_hashes[step_name][op_name]
        )
        with open(self._config_path, "w") as f:
            json.dump(cfg, f, indent=2)

        return path

    def clear_stale(self, step_name: str, op_name: str) -> None:
        """Remove a checkpoint file if it exists (hash mismatch)."""
        path = os.path.join(self.base_dir, step_name, f"{op_name}.json")
        if os.path.exists(path):
            os.remove(path)

    def clear_all(self) -> None:
        """Remove the entire intermediate directory."""
        shutil.rmtree(self.base_dir)

    def flush_batch(self, op_name: str, batch_index: int,
                    data: list[dict]) -> str:
        """Save a partial batch result. Returns the file path."""
        batch_dir = os.path.join(self.base_dir, f"{op_name}_batches")
        os.makedirs(batch_dir, exist_ok=True)
        path = os.path.join(batch_dir, f"batch_{batch_index}.json")
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def has_hash(self, step_name: str, op_name: str) -> bool:
        return op_name in self.op_hashes.get(step_name, {})
