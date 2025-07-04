"""
Checkpoint Manager for efficient storage of intermediate datasets using PyArrow.

This module provides utilities for saving and loading checkpoint data using PyArrow's 
Parquet format with compression.

Key benefits:
- Significantly smaller file sizes due to columnar storage and compression
- Faster loading and saving for large datasets
- Better memory efficiency
- Type preservation
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
    import pandas as pd  # type: ignore
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    pa = None  # type: ignore
    pq = None  # type: ignore
    pd = None  # type: ignore


class CheckpointManager:
    """
    Manages checkpoint storage and retrieval using PyArrow for efficiency.
    
    Features:
    - Automatic compression using Snappy
    - Efficient columnar storage for structured data
    - Data inspection utilities for notebooks
    """
    
    def __init__(self, intermediate_dir: str):
        """
        Initialize the checkpoint manager.
        
        Args:
            intermediate_dir: Directory for storing checkpoints
            
        Raises:
            RuntimeError: If PyArrow is not available
        """
        if not PYARROW_AVAILABLE:
            raise RuntimeError(
                "PyArrow is required for checkpoint management. "
                "Install with: pip install pyarrow pandas"
            )
            
        self.intermediate_dir = intermediate_dir
        
    def _get_checkpoint_path(self, step_name: str, operation_name: str) -> str:
        """Get the checkpoint file path for given step and operation."""
        return os.path.join(self.intermediate_dir, step_name, f"{operation_name}.parquet")
    
    def _get_batch_checkpoint_path(self, operation_name: str, batch_index: int) -> str:
        """Get the batch checkpoint file path."""
        batch_dir = os.path.join(self.intermediate_dir, f"{operation_name}_batches")
        return os.path.join(batch_dir, f"batch_{batch_index}.parquet")
    
    def _convert_to_arrow_table(self, data: List[Dict[str, Any]]) -> Any:  # pa.Table
        """
        Convert list of dictionaries to PyArrow Table.
        
        Args:
            data: List of dictionaries representing records
            
        Returns:
            PyArrow Table with optimized schema
        """
        assert pa is not None  # PyArrow is checked in __init__
        
        if not data:
            # Return empty table with basic schema
            schema = pa.schema([('_placeholder', pa.string())])
            return pa.table([["empty"]], schema=schema)
        
        # Convert to PyArrow table, letting it infer the schema
        try:
            table = pa.table(data)
            return table
        except Exception:
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
        assert pd is not None  # Pandas is checked in __init__
        
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
        Save checkpoint data using PyArrow format.
        
        Args:
            step_name: Name of the pipeline step
            operation_name: Name of the operation
            data: Data to checkpoint
            
        Returns:
            Path where the checkpoint was saved
        """
        assert pq is not None  # PyArrow is checked in __init__
        
        checkpoint_path = self._get_checkpoint_path(step_name, operation_name)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
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
    
    def load_checkpoint(self, step_name: str, operation_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        Load checkpoint data.
        
        Args:
            step_name: Name of the pipeline step
            operation_name: Name of the operation
            
        Returns:
            Loaded data or None if checkpoint doesn't exist
        """
        assert pq is not None  # PyArrow is checked in __init__
        
        checkpoint_path = self._get_checkpoint_path(step_name, operation_name)
        if not os.path.exists(checkpoint_path):
            return None
            
        try:
            table = pq.read_table(checkpoint_path)
            return self._convert_from_arrow_table(table)
        except Exception:
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
        assert pq is not None  # PyArrow is checked in __init__
        
        checkpoint_path = self._get_batch_checkpoint_path(operation_name, batch_index)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
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
    
    def list_checkpoints(self) -> List[Tuple[str, str]]:
        """
        List all available checkpoints.
        
        Returns:
            List of (step_name, operation_name) tuples for available checkpoints
        """
        checkpoints = []
        intermediate_path = Path(self.intermediate_dir)
        
        if not intermediate_path.exists():
            return checkpoints
        
        # Find all .parquet files that match the checkpoint pattern
        for parquet_file in intermediate_path.rglob("*.parquet"):
            # Skip batch files
            if "_batches" in str(parquet_file.parent):
                continue
                
            # Extract step and operation names from path
            relative_path = parquet_file.relative_to(intermediate_path)
            if len(relative_path.parts) >= 2:
                step_name = relative_path.parts[0]
                operation_name = parquet_file.stem  # filename without extension
                checkpoints.append((step_name, operation_name))
        
        return sorted(checkpoints)
    
    def load_checkpoint_as_dataframe(self, step_name: str, operation_name: str) -> Optional[Any]:  # pd.DataFrame
        """
        Load checkpoint data as a pandas DataFrame for analysis.
        
        Args:
            step_name: Name of the pipeline step
            operation_name: Name of the operation
            
        Returns:
            DataFrame or None if checkpoint doesn't exist
        """
        assert pq is not None  # PyArrow is checked in __init__
        
        checkpoint_path = self._get_checkpoint_path(step_name, operation_name)
        if not os.path.exists(checkpoint_path):
            return None
            
        try:
            # Load directly as DataFrame without JSON deserialization
            table = pq.read_table(checkpoint_path)
            return table.to_pandas()
        except Exception:
            return None
    
    def get_checkpoint_info(self, step_name: str, operation_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a checkpoint file (size, schema, row count, etc.).
        
        Args:
            step_name: Name of the pipeline step
            operation_name: Name of the operation
            
        Returns:
            Dictionary with checkpoint information or None if not found
        """
        checkpoint_path = self._get_checkpoint_path(step_name, operation_name)
        
        if not os.path.exists(checkpoint_path):
            return None
        
        try:
            # Get file stats
            stat = os.stat(checkpoint_path)
            
            # Get Arrow table info
            assert pq is not None  # PyArrow is checked in __init__
            table = pq.read_table(checkpoint_path)
            
            return {
                'path': checkpoint_path,
                'size_bytes': stat.st_size,
                'num_rows': table.num_rows,
                'num_columns': table.num_columns,
                'column_names': table.column_names,
                'schema': str(table.schema),
                'format': 'parquet'
            }
        except Exception:
            return {
                'path': checkpoint_path,
                'size_bytes': os.path.getsize(checkpoint_path),
                'error': 'Could not read parquet metadata'
            }
    
    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all checkpoints in the intermediate directory.
        
        Returns:
            Dictionary with summary statistics
        """
        checkpoints = self.list_checkpoints()
        
        total_size = 0
        total_rows = 0
        checkpoint_details = []
        
        for step_name, operation_name in checkpoints:
            info = self.get_checkpoint_info(step_name, operation_name)
            if info:
                total_size += info.get('size_bytes', 0)
                total_rows += info.get('num_rows', 0)
                checkpoint_details.append({
                    'step_name': step_name,
                    'operation_name': operation_name,
                    'size_bytes': info.get('size_bytes', 0),
                    'num_rows': info.get('num_rows', 0),
                    'num_columns': info.get('num_columns', 0)
                })
        
        return {
            'total_checkpoints': len(checkpoints),
            'total_size_bytes': total_size,
            'total_rows': total_rows,
            'checkpoints': checkpoint_details
        }
    
    def load_checkpoint_sample(
        self, 
        step_name: str, 
        operation_name: str, 
        n_rows: int = 10
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Load a sample of rows from a checkpoint for quick inspection.
        
        Args:
            step_name: Name of the pipeline step
            operation_name: Name of the operation
            n_rows: Number of rows to sample
            
        Returns:
            Sample of the data or None if checkpoint doesn't exist
        """
        assert pq is not None  # PyArrow is checked in __init__
        
        checkpoint_path = self._get_checkpoint_path(step_name, operation_name)
        if not os.path.exists(checkpoint_path):
            return None
            
        try:
            # Read only the first n_rows
            table = pq.read_table(checkpoint_path)
            
            # Take sample
            if table.num_rows <= n_rows:
                sample_table = table
            else:
                sample_table = table.slice(0, n_rows)
            
            return self._convert_from_arrow_table(sample_table)
        except Exception:
            return None