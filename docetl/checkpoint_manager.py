"""
PyArrow-based checkpoint manager for DocETL pipelines.

This module provides efficient storage and retrieval of intermediate datasets
using PyArrow format, which offers better compression and faster I/O compared
to JSON storage.
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
    Manages checkpoints for DocETL pipeline operations using PyArrow format.

    This class provides efficient storage and retrieval of intermediate datasets,
    replacing the JSON-based approach with PyArrow for better performance and
    smaller file sizes.
    """

    def __init__(self, intermediate_dir: str, console=None):
        """
        Initialize the checkpoint manager.

        Args:
            intermediate_dir: Directory to store checkpoint files
            console: Rich console for logging (optional)
        """
        self.intermediate_dir = intermediate_dir
        self.console = console
        self.config_path = (
            os.path.join(intermediate_dir, ".docetl_intermediate_config.json")
            if intermediate_dir
            else None
        )

        # Ensure the intermediate directory exists
        if intermediate_dir:
            os.makedirs(intermediate_dir, exist_ok=True)

    def _get_checkpoint_path(
        self, step_name: str, operation_name: str
    ) -> Optional[str]:
        """Get the file path for a checkpoint."""
        if not self.intermediate_dir:
            return None
        return os.path.join(
            self.intermediate_dir, step_name, f"{operation_name}.parquet"
        )

    def _log(self, message: str) -> None:
        """Log a message if console is available."""
        if self.console:
            self.console.log(message)

    def save_checkpoint(
        self, step_name: str, operation_name: str, data: List[Dict], operation_hash: str
    ) -> None:
        """
        Save a checkpoint using PyArrow format.

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

        # Convert data to PyArrow table and save as Parquet
        if data:
            df = pd.DataFrame(data)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, checkpoint_path, compression="snappy")
        else:
            # Handle empty data case
            empty_table = pa.Table.from_arrays([], names=[])
            pq.write_table(empty_table, checkpoint_path, compression="snappy")

        # Update the configuration file with the hash
        self._update_config(step_name, operation_name, operation_hash)

        self._log(
            f"[green]✓ [italic]Checkpoint saved for operation '{operation_name}' "
            f"in step '{step_name}' at {checkpoint_path}[/italic][/green]"
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

        # Check if checkpoint file exists
        checkpoint_path = self._get_checkpoint_path(step_name, operation_name)
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            return None

        try:
            # Load the parquet file
            table = pq.read_table(checkpoint_path)
            df = table.to_pandas()

            # Convert back to list of dictionaries
            data = df.to_dict("records")

            self._log(
                f"[green]✓[/green] [italic]Loaded checkpoint for operation '{operation_name}' "
                f"in step '{step_name}' from {checkpoint_path}[/italic]"
            )

            return data

        except Exception as e:
            self._log(f"[red]Failed to load checkpoint: {e}[/red]")
            return None

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
        checkpoint_path = self._get_checkpoint_path(step_name, operation_name)
        if not checkpoint_path:
            return None

        if not os.path.exists(checkpoint_path):
            return None

        try:
            table = pq.read_table(checkpoint_path)
            df = table.to_pandas()
            return df.to_dict("records")
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
        checkpoint_path = self._get_checkpoint_path(step_name, operation_name)
        if not checkpoint_path:
            return None

        if not os.path.exists(checkpoint_path):
            return None

        try:
            table = pq.read_table(checkpoint_path)
            return table.to_pandas()
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

            # Look for parquet files in the step directory
            for filename in os.listdir(step_path):
                if filename.endswith(".parquet"):
                    operation_name = filename[:-8]  # Remove .parquet extension
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
        checkpoint_path = self._get_checkpoint_path(step_name, operation_name)
        if not checkpoint_path:
            return None

        if os.path.exists(checkpoint_path):
            return os.path.getsize(checkpoint_path)
        return None

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
                if file.endswith(".parquet"):
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
