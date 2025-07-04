"""
Checkpoint Manager for efficient storage of intermediate datasets using PyArrow.

This module provides utilities for saving and loading checkpoint data using PyArrow's 
Parquet format with compression, while maintaining backward compatibility with 
existing JSON checkpoints.

Key benefits over JSON:
- Significantly smaller file sizes due to columnar storage and compression
- Faster loading and saving for large datasets
- Better memory efficiency
- Type preservation
"""

import json
import os
import hashlib
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    pa = None  # type: ignore
    pq = None  # type: ignore

try:
    import pandas as pd  # type: ignore
    PANDAS_AVAILABLE = True
except ImportError:
    # Pandas is required by pyarrow for some operations
    pd = None  # type: ignore
    PANDAS_AVAILABLE = False


class CheckpointManager:
    """
    Manages checkpoint storage and retrieval using PyArrow for efficiency.
    
    Features:
    - Automatic compression using Snappy
    - Backward compatibility with JSON checkpoints
    - Efficient columnar storage for structured data
    - Data integrity verification
    """
    
    def __init__(self, intermediate_dir: str, enable_arrow: bool = True):
        """
        Initialize the checkpoint manager.
        
        Args:
            intermediate_dir: Directory for storing checkpoints
            enable_arrow: Whether to use PyArrow format (falls back to JSON if not available)
        """
        self.intermediate_dir = intermediate_dir
        self.enable_arrow = enable_arrow and PYARROW_AVAILABLE and PANDAS_AVAILABLE
        
    def _get_checkpoint_path(self, step_name: str, operation_name: str, use_arrow: Optional[bool] = None) -> str:
        """Get the checkpoint file path for given step and operation."""
        if use_arrow is None:
            use_arrow = self.enable_arrow
            
        extension = "parquet" if use_arrow else "json"
        return os.path.join(self.intermediate_dir, step_name, f"{operation_name}.{extension}")
    
    def _get_batch_checkpoint_path(self, operation_name: str, batch_index: int, use_arrow: Optional[bool] = None) -> str:
        """Get the batch checkpoint file path."""
        if use_arrow is None:
            use_arrow = self.enable_arrow
            
        extension = "parquet" if use_arrow else "json"
        batch_dir = os.path.join(self.intermediate_dir, f"{operation_name}_batches")
        return os.path.join(batch_dir, f"batch_{batch_index}.{extension}")
    
    def _convert_to_arrow_table(self, data: List[Dict[str, Any]]) -> Any:  # pa.Table
        """
        Convert list of dictionaries to PyArrow Table.
        
        Args:
            data: List of dictionaries representing records
            
        Returns:
            PyArrow Table with optimized schema
        """
        if not PYARROW_AVAILABLE or pa is None:
            raise RuntimeError("PyArrow is not available")
            
        if not data:
            # Return empty table with basic schema
            schema = pa.schema([
                ('_placeholder', pa.string())
            ])
            return pa.table([["empty"]], schema=schema)
        
        # Convert to PyArrow table, letting it infer the schema
        try:
            table = pa.table(data)
            return table
        except Exception as e:
            # Fallback: convert all values to strings to handle mixed types
            sanitized_data = []
            for record in data:
                sanitized_record = {}
                for k, v in record.items():
                    if isinstance(v, (dict, list)):
                        # Serialize complex objects as JSON strings
                        sanitized_record[k] = json.dumps(v)
                    else:
                        sanitized_record[k] = str(v) if v is not None else None
                sanitized_data.append(sanitized_record)
            
            return pa.table(sanitized_data)
    
    def _convert_from_arrow_table(self, table: Any) -> List[Dict[str, Any]]:  # pa.Table
        """
        Convert PyArrow Table back to list of dictionaries.
        
        Args:
            table: PyArrow Table
            
        Returns:
            List of dictionaries
        """
        if not PANDAS_AVAILABLE or pd is None:
            raise RuntimeError("Pandas is not available")
            
        if table.num_rows == 0:
            return []
        
        # Convert to pandas first for easier handling, then to records
        df = table.to_pandas()
        
        # Convert back to original format, attempting to deserialize JSON strings
        records = []
        for _, row in df.iterrows():
            record = {}
            for col, val in row.items():
                if col == '_placeholder':  # Skip placeholder column
                    continue
                    
                if pd.isna(val):
                    record[col] = None
                elif isinstance(val, str):
                    # Try to deserialize JSON strings back to objects
                    try:
                        # Check if it looks like JSON
                        if val.startswith(('{', '[')):
                            record[col] = json.loads(val)
                        else:
                            record[col] = val
                    except (json.JSONDecodeError, ValueError):
                        record[col] = val
                else:
                    record[col] = val
            records.append(record)
        
        return records
    
    def save_checkpoint(self, step_name: str, operation_name: str, data: List[Dict[str, Any]]) -> str:
        """
        Save checkpoint data using the most efficient available format.
        
        Args:
            step_name: Name of the pipeline step
            operation_name: Name of the operation
            data: Data to checkpoint
            
        Returns:
            Path where the checkpoint was saved
        """
        checkpoint_path = self._get_checkpoint_path(step_name, operation_name)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        if self.enable_arrow and PYARROW_AVAILABLE and pq is not None:
            try:
                # Use PyArrow for efficient storage
                table = self._convert_to_arrow_table(data)
                
                # Save with compression
                pq.write_table(
                    table, 
                    checkpoint_path,
                    compression='snappy',  # Good balance of speed and compression
                    use_dictionary=True,   # Optimize for repeated strings
                    write_statistics=True  # Enable column statistics
                )
                
                return checkpoint_path
            except Exception as e:
                # Fallback to JSON if Arrow fails
                print(f"Warning: Failed to save Arrow checkpoint, falling back to JSON: {e}")
                checkpoint_path = self._get_checkpoint_path(step_name, operation_name, use_arrow=False)
        
        # JSON fallback
        with open(checkpoint_path, 'w') as f:
            json.dump(data, f)
        
        return checkpoint_path
    
    def load_checkpoint(self, step_name: str, operation_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        Load checkpoint data, trying Arrow format first, then JSON fallback.
        
        Args:
            step_name: Name of the pipeline step
            operation_name: Name of the operation
            
        Returns:
            Loaded data or None if checkpoint doesn't exist
        """
        # Try Arrow format first
        if self.enable_arrow and PYARROW_AVAILABLE and pq is not None:
            arrow_path = self._get_checkpoint_path(step_name, operation_name, use_arrow=True)
            if os.path.exists(arrow_path):
                try:
                    table = pq.read_table(arrow_path)
                    return self._convert_from_arrow_table(table)
                except Exception as e:
                    print(f"Warning: Failed to load Arrow checkpoint, trying JSON: {e}")
        
        # Try JSON format
        json_path = self._get_checkpoint_path(step_name, operation_name, use_arrow=False)
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load JSON checkpoint: {e}")
                return None
        
        return None
    
    def save_batch_checkpoint(self, operation_name: str, batch_index: int, data: List[Dict[str, Any]]) -> str:
        """
        Save batch checkpoint data.
        
        Args:
            operation_name: Name of the operation
            batch_index: Index of the batch
            data: Batch data to save
            
        Returns:
            Path where the batch checkpoint was saved
        """
        checkpoint_path = self._get_batch_checkpoint_path(operation_name, batch_index)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        if self.enable_arrow and PYARROW_AVAILABLE and pq is not None:
            try:
                # Use PyArrow for efficient storage
                table = self._convert_to_arrow_table(data)
                pq.write_table(
                    table, 
                    checkpoint_path,
                    compression='snappy',
                    use_dictionary=True,
                    write_statistics=True
                )
                return checkpoint_path
            except Exception as e:
                print(f"Warning: Failed to save Arrow batch checkpoint, falling back to JSON: {e}")
                checkpoint_path = self._get_batch_checkpoint_path(operation_name, batch_index, use_arrow=False)
        
        # JSON fallback
        with open(checkpoint_path, 'w') as f:
            json.dump(data, f)
        
        return checkpoint_path
    
    def get_checkpoint_info(self, step_name: str, operation_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a checkpoint file (size, format, etc.).
        
        Args:
            step_name: Name of the pipeline step
            operation_name: Name of the operation
            
        Returns:
            Dictionary with checkpoint information or None if not found
        """
        arrow_path = self._get_checkpoint_path(step_name, operation_name, use_arrow=True)
        json_path = self._get_checkpoint_path(step_name, operation_name, use_arrow=False)
        
        info = {}
        
        if os.path.exists(arrow_path):
            stat = os.stat(arrow_path)
            info['arrow'] = {
                'path': arrow_path,
                'size_bytes': stat.st_size,
                'format': 'parquet',
                'exists': True
            }
        
        if os.path.exists(json_path):
            stat = os.stat(json_path)
            info['json'] = {
                'path': json_path,
                'size_bytes': stat.st_size,
                'format': 'json',
                'exists': True
            }
        
        return info if info else None
    
    def migrate_json_to_arrow(self, step_name: str, operation_name: str) -> bool:
        """
        Migrate existing JSON checkpoint to Arrow format.
        
        Args:
            step_name: Name of the pipeline step
            operation_name: Name of the operation
            
        Returns:
            True if migration was successful, False otherwise
        """
        if not (self.enable_arrow and PYARROW_AVAILABLE):
            return False
        
        json_path = self._get_checkpoint_path(step_name, operation_name, use_arrow=False)
        if not os.path.exists(json_path):
            return False
        
        try:
            # Load JSON data
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Save as Arrow
            arrow_path = self.save_checkpoint(step_name, operation_name, data)
            
            # Verify the migration worked
            loaded_data = self.load_checkpoint(step_name, operation_name)
            if loaded_data == data:
                # Migration successful, optionally remove JSON file
                # os.remove(json_path)  # Uncomment to remove JSON after migration
                return True
            else:
                # Remove failed Arrow file
                if os.path.exists(arrow_path):
                    os.remove(arrow_path)
                return False
                
        except Exception as e:
            print(f"Migration failed: {e}")
            return False