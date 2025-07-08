"""
Flexible checkpoint manager for DocETL pipelines.

This module provides storage and retrieval of intermediate datasets
using either JSON or PyArrow format. PyArrow offers better compression
and faster I/O for large datasets, while JSON provides human-readable
checkpoints and simpler debugging.
"""

import json
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class CheckpointManager:
    """
    Manages checkpoints for DocETL pipeline operations using JSON or PyArrow format.

    This class provides flexible storage and retrieval of intermediate datasets,
    supporting both JSON (human-readable, default) and PyArrow (efficient, compressed)
    storage formats. Users can choose the format based on their needs.
    """

    def __init__(self, intermediate_dir: str, console=None, storage_type: str = "json"):
        """
        Initialize the checkpoint manager.

        Args:
            intermediate_dir: Directory to store checkpoint files
            console: Rich console for logging (optional)
            storage_type: Storage format - "json" (default) or "arrow"
        """
        self.intermediate_dir = intermediate_dir
        self.console = console
        self.storage_type = storage_type.lower()

        if self.storage_type not in ["json", "arrow"]:
            raise ValueError(
                f"Invalid storage_type '{storage_type}'. Must be 'json' or 'arrow'"
            )

        self.config_path = (
            os.path.join(intermediate_dir, ".docetl_intermediate_config.json")
            if intermediate_dir
            else None
        )

        # Ensure the intermediate directory exists
        if intermediate_dir:
            os.makedirs(intermediate_dir, exist_ok=True)

    @classmethod
    def from_intermediate_dir(
        cls, intermediate_dir: str, console=None, storage_type: Optional[str] = None
    ):
        """
        Create a CheckpointManager from an intermediate directory path.

        If storage_type is not specified, automatically detects the most common format
        in the directory (prefers arrow if both formats exist equally).

        Args:
            intermediate_dir: Path to the intermediate directory containing checkpoints
            console: Rich console for logging (optional)
            storage_type: Storage format - "json", "arrow", or None for auto-detection

        Returns:
            CheckpointManager instance
        """
        if storage_type is None:
            storage_type = cls._detect_storage_type(intermediate_dir)

        return cls(intermediate_dir, console=console, storage_type=storage_type)

    @staticmethod
    def _detect_storage_type(intermediate_dir: str) -> str:
        """
        Detect the primary storage type used in an intermediate directory.

        Args:
            intermediate_dir: Path to the intermediate directory

        Returns:
            Detected storage type ("json" or "arrow"), defaults to "json" if unclear
        """
        if not os.path.exists(intermediate_dir):
            return "json"  # Default for new directories

        json_count = 0
        parquet_count = 0

        # Count checkpoint files of each type
        for root, dirs, files in os.walk(intermediate_dir):
            for file in files:
                if file.endswith(".json") and not file.startswith("."):
                    json_count += 1
                elif file.endswith(".parquet"):
                    parquet_count += 1

        # Prefer arrow if more parquet files, or if equal and both exist
        if parquet_count > json_count or (
            parquet_count > 0 and parquet_count == json_count
        ):
            return "arrow"
        else:
            return "json"

    def _get_checkpoint_path(
        self, step_name: str, operation_name: str, storage_type: Optional[str] = None
    ) -> Optional[str]:
        """Get the file path for a checkpoint."""
        if not self.intermediate_dir:
            return None

        storage = storage_type or self.storage_type
        extension = "parquet" if storage == "arrow" else "json"

        return os.path.join(
            self.intermediate_dir, step_name, f"{operation_name}.{extension}"
        )

    def _find_existing_checkpoint(
        self, step_name: str, operation_name: str
    ) -> Optional[Tuple[str, str]]:
        """Find existing checkpoint, checking both JSON and Parquet formats.

        Returns:
            Tuple of (file_path, format) if found, None otherwise
        """
        # Check current storage type first
        current_path = self._get_checkpoint_path(step_name, operation_name)
        if current_path and os.path.exists(current_path):
            return current_path, self.storage_type

        # Check the other format for backward compatibility
        other_type = "json" if self.storage_type == "arrow" else "arrow"
        other_path = self._get_checkpoint_path(step_name, operation_name, other_type)
        if other_path and os.path.exists(other_path):
            return other_path, other_type

        return None

    def _log(self, message: str) -> None:
        """Log a message if console is available."""
        if self.console:
            self.console.log(message)

    def save_checkpoint(
        self, step_name: str, operation_name: str, data: List[Dict], operation_hash: str
    ) -> None:
        """
        Save a checkpoint using the configured storage format.

        Args:
            step_name: Name of the pipeline step
            operation_name: Name of the operation
            data: Data to checkpoint
            operation_hash: Hash of the operation configuration
        """
        if not self.intermediate_dir:
            return

        checkpoint_path = self._get_checkpoint_path(step_name, operation_name)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        # Save based on storage type
        if self.storage_type == "arrow":
            self._save_as_parquet(checkpoint_path, data)
        else:  # json
            self._save_as_json(checkpoint_path, data)

        # Update the configuration file with the hash
        self._update_config(step_name, operation_name, operation_hash)

        format_name = "PyArrow" if self.storage_type == "arrow" else "JSON"
        self._log(
            f"[green]✓ [italic]Checkpoint saved ({format_name}) for operation '{operation_name}' "
            f"in step '{step_name}' at {checkpoint_path}[/italic][/green]"
        )

    def _save_as_json(self, checkpoint_path: str, data: List[Dict]) -> None:
        """Save checkpoint data as JSON."""
        with open(checkpoint_path, "w") as f:
            json.dump(data, f)

    def _sanitize_for_parquet(self, data: List[Dict]) -> List[Dict]:
        """Sanitize data to make it compatible with PyArrow/Parquet serialization."""
        import json

        def sanitize_value(value):
            """Recursively sanitize a value for PyArrow compatibility."""
            if isinstance(value, dict):
                if not value:  # Empty dict
                    return {"__empty_dict__": True}
                return {k: sanitize_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                if not value:  # Empty list
                    return ["__empty_list__"]
                # Check if list has mixed types or contains None - serialize as JSON string if so
                has_none = any(item is None for item in value)
                if len(value) > 1:
                    types = set(
                        type(item).__name__ for item in value if item is not None
                    )
                    if len(types) > 1 or (
                        has_none and len(types) >= 1
                    ):  # Mixed types or has None with other types
                        return f"__mixed_list_json__:{json.dumps(value)}"
                return [sanitize_value(item) for item in value]
            elif value is None:
                return "__null__"
            else:
                return value

        def sanitize_record(record):
            """Sanitize a single record."""
            if not isinstance(record, dict):
                return record
            return {k: sanitize_value(v) for k, v in record.items()}

        return [sanitize_record(record) for record in data]

    def _desanitize_from_parquet(self, data: List[Dict]) -> List[Dict]:
        """Restore original data structure from sanitized Parquet data."""
        import json

        import numpy as np

        def desanitize_value(value):
            """Recursively restore original value structure."""
            # Handle numpy arrays (from pandas conversion)
            if isinstance(value, np.ndarray):
                # Convert to list first, then check for empty list markers
                value_list = value.tolist()
                if value_list == ["__empty_list__"]:
                    return []
                return [desanitize_value(item) for item in value_list]
            elif isinstance(value, str) and value.startswith("__mixed_list_json__:"):
                # Restore mixed-type list from JSON
                json_str = value[len("__mixed_list_json__:") :]
                return json.loads(json_str)
            elif isinstance(value, dict):
                if value == {"__empty_dict__": True}:
                    return {}
                return {k: desanitize_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                if value == ["__empty_list__"]:
                    return []
                return [desanitize_value(item) for item in value]
            elif value == "__null__":
                return None
            elif pd.isna(value):  # Handle pandas NaN values
                return None
            else:
                return value

        def desanitize_record(record):
            """Desanitize a single record."""
            if not isinstance(record, dict):
                return record
            return {k: desanitize_value(v) for k, v in record.items()}

        return [desanitize_record(record) for record in data]

    def _save_as_parquet(self, checkpoint_path: str, data: List[Dict]) -> None:
        """Save checkpoint data as Parquet with data sanitization."""
        if not data:
            # Handle empty data case
            empty_table = pa.Table.from_arrays([], names=[])
            pq.write_table(empty_table, checkpoint_path, compression="snappy")
            return

        # Sanitize data to make it PyArrow-compatible
        sanitized_data = self._sanitize_for_parquet(data)

        try:
            df = pd.DataFrame(sanitized_data)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, checkpoint_path, compression="snappy")
        except Exception as e:
            # If sanitization still doesn't work, raise the error
            raise RuntimeError(
                f"Failed to serialize data to Parquet format even after sanitization. "
                f"This indicates a more fundamental incompatibility. Original error: {str(e)}"
            )

    def load_checkpoint(
        self, step_name: str, operation_name: str, operation_hash: str
    ) -> Optional[List[Dict]]:
        """
        Load a checkpoint if it exists and is valid.

        Args:
            step_name: Name of the pipeline step
            operation_name: Name of the operation
            operation_hash: Expected hash of the operation configuration

        Returns:
            List of dictionaries if checkpoint exists and is valid, None otherwise
        """
        if not self.intermediate_dir:
            return None

        # Check if config file exists
        if not self.config_path or not os.path.exists(self.config_path):
            return None

        # Load and validate configuration
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

        # Check if the hash matches
        if config.get(step_name, {}).get(operation_name) != operation_hash:
            return None

        # Find existing checkpoint (checks both formats)
        checkpoint_info = self._find_existing_checkpoint(step_name, operation_name)
        if not checkpoint_info:
            return None

        checkpoint_path, format_type = checkpoint_info

        try:
            # Load based on the format of the existing file
            if format_type == "arrow":
                data = self._load_from_parquet(checkpoint_path)
            else:  # json
                data = self._load_from_json(checkpoint_path)

            format_name = "PyArrow" if format_type == "arrow" else "JSON"
            self._log(
                f"[green]✓[/green] [italic]Loaded checkpoint ({format_name}) for operation '{operation_name}' "
                f"in step '{step_name}' from {checkpoint_path}[/italic]"
            )

            return data

        except Exception as e:
            self._log(f"[red]Failed to load checkpoint: {e}[/red]")
            return None

    def _load_from_json(self, checkpoint_path: str) -> List[Dict]:
        """Load checkpoint data from JSON."""
        with open(checkpoint_path, "r") as f:
            return json.load(f)

    def _load_from_parquet(self, checkpoint_path: str) -> List[Dict]:
        """Load checkpoint data from Parquet and desanitize."""
        table = pq.read_table(checkpoint_path)
        df = table.to_pandas()
        data = df.to_dict("records")
        # Restore original data structure from sanitized data
        return self._desanitize_from_parquet(data)

    def _update_config(
        self, step_name: str, operation_name: str, operation_hash: str
    ) -> None:
        """Update the checkpoint configuration file."""
        if not self.config_path:
            return

        # Load existing config or create new one
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    config = json.load(f)
            except (json.JSONDecodeError, IOError):
                config = {}
        else:
            config = {}

        # Ensure nested structure exists
        if step_name not in config:
            config[step_name] = {}

        # Update the hash
        config[step_name][operation_name] = operation_hash

        # Save updated config
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

    def load_output_by_step_and_op(
        self, step_name: str, operation_name: str
    ) -> Optional[List[Dict]]:
        """
        Load output data for a specific step and operation.

        Args:
            step_name: Name of the pipeline step
            operation_name: Name of the operation

        Returns:
            List of dictionaries if data exists, None otherwise
        """
        # Find existing checkpoint (checks both formats)
        checkpoint_info = self._find_existing_checkpoint(step_name, operation_name)
        if not checkpoint_info:
            return None

        checkpoint_path, format_type = checkpoint_info

        try:
            # Load based on the format of the existing file
            if format_type == "arrow":
                return self._load_from_parquet(checkpoint_path)
            else:  # json
                return self._load_from_json(checkpoint_path)
        except Exception as e:
            self._log(f"[red]Failed to load output: {e}[/red]")
            return None

    def load_output_as_dataframe(
        self, step_name: str, operation_name: str
    ) -> Optional[pd.DataFrame]:
        """
        Load output data as a pandas DataFrame.

        Args:
            step_name: Name of the pipeline step
            operation_name: Name of the operation

        Returns:
            DataFrame if data exists, None otherwise
        """
        # Find existing checkpoint (checks both formats)
        checkpoint_info = self._find_existing_checkpoint(step_name, operation_name)
        if not checkpoint_info:
            return None

        checkpoint_path, format_type = checkpoint_info

        try:
            # Load based on the format of the existing file
            if format_type == "arrow":
                table = pq.read_table(checkpoint_path)
                return table.to_pandas()
            else:  # json
                data = self._load_from_json(checkpoint_path)
                return pd.DataFrame(data) if data else pd.DataFrame()
        except Exception as e:
            self._log(f"[red]Failed to load output as DataFrame: {e}[/red]")
            return None

    def list_outputs(self) -> List[Tuple[str, str]]:
        """
        List all available outputs (step_name, operation_name pairs).

        Returns:
            List of tuples containing (step_name, operation_name)
        """
        outputs = []

        if not self.intermediate_dir or not os.path.exists(self.intermediate_dir):
            return outputs

        # Walk through the directory structure
        for step_name in os.listdir(self.intermediate_dir):
            step_path = os.path.join(self.intermediate_dir, step_name)

            # Skip files and hidden directories
            if not os.path.isdir(step_path) or step_name.startswith("."):
                continue

            # Look for checkpoint files in the step directory (both formats)
            for filename in os.listdir(step_path):
                if filename.endswith(".parquet"):
                    operation_name = filename[:-8]  # Remove .parquet extension
                    outputs.append((step_name, operation_name))
                elif filename.endswith(".json") and not filename.startswith("."):
                    operation_name = filename[:-5]  # Remove .json extension
                    # Avoid duplicates if both formats exist
                    if (step_name, operation_name) not in outputs:
                        outputs.append((step_name, operation_name))

        return sorted(outputs)

    def clear_all_checkpoints(self) -> None:
        """Clear all checkpoints and configuration."""
        if self.intermediate_dir and os.path.exists(self.intermediate_dir):
            shutil.rmtree(self.intermediate_dir)
            os.makedirs(self.intermediate_dir, exist_ok=True)
            self._log("[green]✓ All checkpoints cleared[/green]")

    def clear_step_checkpoints(self, step_name: str) -> None:
        """
        Clear all checkpoints for a specific step.

        Args:
            step_name: Name of the step to clear
        """
        step_path = os.path.join(self.intermediate_dir, step_name)
        if os.path.exists(step_path):
            shutil.rmtree(step_path)

            # Remove from config
            if self.config_path and os.path.exists(self.config_path):
                try:
                    with open(self.config_path, "r") as f:
                        config = json.load(f)

                    if step_name in config:
                        del config[step_name]

                    with open(self.config_path, "w") as f:
                        json.dump(config, f, indent=2)

                except (json.JSONDecodeError, IOError):
                    pass

            self._log(f"[green]✓ Cleared checkpoints for step '{step_name}'[/green]")

    def get_checkpoint_size(self, step_name: str, operation_name: str) -> Optional[int]:
        """
        Get the size of a checkpoint file in bytes.

        Args:
            step_name: Name of the pipeline step
            operation_name: Name of the operation

        Returns:
            Size in bytes if file exists, None otherwise
        """
        # Find existing checkpoint (checks both formats)
        checkpoint_info = self._find_existing_checkpoint(step_name, operation_name)
        if not checkpoint_info:
            return None

        checkpoint_path, _ = checkpoint_info
        return os.path.getsize(checkpoint_path)

    def get_total_checkpoint_size(self) -> int:
        """
        Get the total size of all checkpoints in bytes.

        Returns:
            Total size in bytes
        """
        total_size = 0

        if not self.intermediate_dir or not os.path.exists(self.intermediate_dir):
            return total_size

        for root, dirs, files in os.walk(self.intermediate_dir):
            for file in files:
                if file.endswith((".parquet", ".json")) and not file.startswith("."):
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)

        return total_size

    def save_incremental_checkpoint(
        self,
        step_name: str,
        operation_name: str,
        data: List[Dict],
        operation_hash: str,
        input_hashes: Optional[List[str]] = None,
    ) -> None:
        """
        Save checkpoint with incremental processing capabilities.

        This method can detect which records have changed based on input hashes
        and potentially avoid reprocessing unchanged records in future runs.

        Args:
            step_name: Name of the pipeline step
            operation_name: Name of the operation
            data: Data to checkpoint
            operation_hash: Hash of the operation configuration
            input_hashes: Optional list of hashes for input records to enable change detection
        """
        # For now, delegate to regular save_checkpoint
        # Future enhancement: store input_hashes for change detection
        self.save_checkpoint(step_name, operation_name, data, operation_hash)

        # Store input hashes for future incremental processing
        if input_hashes and self.intermediate_dir:
            hash_path = self._get_hash_tracking_path(step_name, operation_name)
            if hash_path:
                try:
                    hash_data = {
                        "operation_hash": operation_hash,
                        "input_hashes": input_hashes,
                        "record_count": len(data),
                    }
                    os.makedirs(os.path.dirname(hash_path), exist_ok=True)
                    with open(hash_path, "w") as f:
                        json.dump(hash_data, f)
                except Exception as e:
                    self._log(
                        f"[yellow]Warning: Could not save hash tracking data: {e}[/yellow]"
                    )

    def _get_hash_tracking_path(
        self, step_name: str, operation_name: str
    ) -> Optional[str]:
        """Get the path for storing input hash tracking data."""
        if not self.intermediate_dir:
            return None
        return os.path.join(
            self.intermediate_dir, step_name, f"{operation_name}_input_hashes.json"
        )

    def get_incremental_processing_info(
        self, step_name: str, operation_name: str, current_input_hashes: List[str]
    ) -> Dict[str, Any]:
        """
        Get information about what records need reprocessing for incremental updates.

        Args:
            step_name: Name of the pipeline step
            operation_name: Name of the operation
            current_input_hashes: Hashes of current input records

        Returns:
            Dictionary with incremental processing information:
            - 'needs_full_reprocess': Boolean indicating if full reprocessing is needed
            - 'changed_indices': List of indices that have changed
            - 'unchanged_indices': List of indices that haven't changed
            - 'new_indices': List of indices for new records
            - 'removed_count': Number of records that were removed
        """
        if not self.intermediate_dir:
            return {"needs_full_reprocess": True, "reason": "No intermediate directory"}

        hash_path = self._get_hash_tracking_path(step_name, operation_name)
        if not hash_path or not os.path.exists(hash_path):
            return {
                "needs_full_reprocess": True,
                "reason": "No previous hash tracking data",
            }

        try:
            with open(hash_path, "r") as f:
                previous_data = json.load(f)

            previous_hashes = previous_data.get("input_hashes", [])

            # Compare current vs previous hashes
            changed_indices = []
            unchanged_indices = []
            new_indices = []

            min_len = min(len(current_input_hashes), len(previous_hashes))

            # Check existing records for changes
            for i in range(min_len):
                if current_input_hashes[i] != previous_hashes[i]:
                    changed_indices.append(i)
                else:
                    unchanged_indices.append(i)

            # Check for new records
            if len(current_input_hashes) > len(previous_hashes):
                new_indices = list(
                    range(len(previous_hashes), len(current_input_hashes))
                )

            removed_count = max(0, len(previous_hashes) - len(current_input_hashes))

            return {
                "needs_full_reprocess": False,
                "changed_indices": changed_indices,
                "unchanged_indices": unchanged_indices,
                "new_indices": new_indices,
                "removed_count": removed_count,
                "total_changes": len(changed_indices)
                + len(new_indices)
                + removed_count,
            }

        except (json.JSONDecodeError, IOError, KeyError) as e:
            return {
                "needs_full_reprocess": True,
                "reason": f"Error reading hash data: {e}",
            }

    def load_incremental_checkpoint(
        self,
        step_name: str,
        operation_name: str,
        operation_hash: str,
        unchanged_indices: Optional[List[int]] = None,
    ) -> Optional[List[Dict]]:
        """
        Load checkpoint data, optionally filtered to unchanged records only.

        Args:
            step_name: Name of the pipeline step
            operation_name: Name of the operation
            operation_hash: Expected hash of the operation configuration
            unchanged_indices: Optional list of indices to load (for incremental processing)

        Returns:
            List of dictionaries if checkpoint exists, None otherwise
        """
        data = self.load_checkpoint(step_name, operation_name, operation_hash)

        if data is None or unchanged_indices is None:
            return data

        # Filter to only unchanged records
        try:
            return [data[i] for i in unchanged_indices if i < len(data)]
        except (IndexError, TypeError):
            self._log(
                "[yellow]Warning: Could not filter incremental data, returning full dataset[/yellow]"
            )
            return data
