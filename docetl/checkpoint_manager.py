"""
Checkpoint Manager for efficient storage of intermediate datasets using PyArrow.

This module provides utilities for saving and loading checkpoint data using a single
PyArrow dataset with incremental storage to eliminate redundancy.

Key benefits:
- Single Arrow dataset eliminates storage redundancy
- Automatic deduplication through Arrow's compression
- Incremental record storage with metadata
- Efficient columnar storage and compression
"""

import json
import os
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore
import pandas as pd  # type: ignore


class CheckpointManager:
    """
    Manages checkpoint storage using a single Arrow dataset with incremental records.
    
    Features:
    - Single dataset eliminates redundancy across checkpoints
    - Automatic compression and deduplication
    - Efficient metadata-based record retrieval
    - Incremental storage of only new/changed records
    """
    
    def __init__(self, intermediate_dir: str):
        """
        Initialize the checkpoint manager.
        
        Args:
            intermediate_dir: Directory for storing the checkpoint dataset
        """
        self.intermediate_dir = intermediate_dir
        self.dataset_path = os.path.join(intermediate_dir, "checkpoints.parquet")
        self.index_path = os.path.join(intermediate_dir, "checkpoint_index.parquet")
        
        # Ensure directory exists
        os.makedirs(intermediate_dir, exist_ok=True)
        
        # Initialize dataset and index if they don't exist
        self._ensure_dataset_exists()
        self._ensure_index_exists()
    
    def _ensure_dataset_exists(self):
        """Ensure the main dataset file exists with proper schema."""
        if not os.path.exists(self.dataset_path):
            # Create empty dataset with base schema
            schema = pa.schema([
                ('record_id', pa.string()),
                ('record_hash', pa.string()),
                ('data_json', pa.string())  # Serialized record data
            ])
            empty_table = pa.table([], schema=schema)
            pq.write_table(empty_table, self.dataset_path)
    
    def _ensure_index_exists(self):
        """Ensure the checkpoint index file exists with proper schema."""
        if not os.path.exists(self.index_path):
            # Create empty index
            schema = pa.schema([
                ('step_name', pa.string()),
                ('operation_name', pa.string()),
                ('record_ids', pa.list_(pa.string())),  # List of record IDs for this checkpoint
                ('num_records', pa.int64()),
                ('created_at', pa.timestamp('ns'))
            ])
            empty_table = pa.table([], schema=schema)
            pq.write_table(empty_table, self.index_path)
    
    def _compute_record_hash(self, record: Dict[str, Any]) -> str:
        """Compute a hash for a record to enable deduplication."""
        record_str = json.dumps(record, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(record_str.encode()).hexdigest()
    
    def _load_existing_records(self) -> Dict[str, str]:
        """Load existing record hashes to avoid duplicates."""
        try:
            table = pq.read_table(self.dataset_path)
            if table.num_rows == 0:
                return {}
            
            df = table.to_pandas()
            return dict(zip(df['record_hash'], df['record_id']))
        except Exception:
            return {}
    
    def save_checkpoint(self, step_name: str, operation_name: str, data: List[Dict[str, Any]]) -> str:
        """
        Save checkpoint data using incremental storage.
        
        Args:
            step_name: Name of the pipeline step
            operation_name: Name of the operation
            data: Data to checkpoint
            
        Returns:
            Path where the checkpoint was saved (always the dataset path)
        """
        if not data:
            # Handle empty data case
            self._update_index(step_name, operation_name, [], len(data))
            return self.dataset_path
        
        # Load existing records to avoid duplicates
        existing_hashes = self._load_existing_records()
        
        # Process new records
        new_records = []
        record_ids = []
        
        for i, record in enumerate(data):
            record_hash = self._compute_record_hash(record)
            
            if record_hash in existing_hashes:
                # Reuse existing record
                record_ids.append(existing_hashes[record_hash])
            else:
                # Create new record
                record_id = f"{step_name}_{operation_name}_{i}_{record_hash[:8]}"
                record_ids.append(record_id)
                
                new_records.append({
                    'record_id': record_id,
                    'record_hash': record_hash,
                    'data_json': json.dumps(record)
                })
                existing_hashes[record_hash] = record_id
        
        # Append new records to dataset if any
        if new_records:
            new_table = pa.table(new_records)
            
            # Read existing data and append
            try:
                existing_table = pq.read_table(self.dataset_path)
                combined_table = pa.concat_tables([existing_table, new_table])
            except Exception:
                combined_table = new_table
            
            # Write back with compression
            pq.write_table(
                combined_table,
                self.dataset_path,
                compression='snappy',
                use_dictionary=True,
                write_statistics=True
            )
        
        # Update index
        self._update_index(step_name, operation_name, record_ids, len(data))
        
        return self.dataset_path
    
    def _update_index(self, step_name: str, operation_name: str, record_ids: List[str], num_records: int):
        """Update the checkpoint index with new checkpoint info."""
        import pyarrow.compute as pc  # type: ignore
        from datetime import datetime
        
        # Read existing index
        try:
            index_table = pq.read_table(self.index_path)
            index_df = index_table.to_pandas()
        except Exception:
            index_df = pd.DataFrame(columns=['step_name', 'operation_name', 'record_ids', 'num_records', 'created_at'])
        
        # Remove existing entry for this step/operation if it exists
        mask = ~((index_df['step_name'] == step_name) & (index_df['operation_name'] == operation_name))
        index_df = index_df[mask]
        
        # Add new entry
        new_entry = pd.DataFrame([{
            'step_name': step_name,
            'operation_name': operation_name,
            'record_ids': record_ids,
            'num_records': num_records,
            'created_at': datetime.now()
        }])
        
        index_df = pd.concat([index_df, new_entry], ignore_index=True)
        
        # Write back
        index_table = pa.table(index_df)
        pq.write_table(index_table, self.index_path)
    
    def load_checkpoint(self, step_name: str, operation_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        Load checkpoint data.
        
        Args:
            step_name: Name of the pipeline step
            operation_name: Name of the operation
            
        Returns:
            Loaded data or None if checkpoint doesn't exist
        """
        # Get record IDs from index
        record_ids = self._get_record_ids(step_name, operation_name)
        if record_ids is None:
            return None
        
        if not record_ids:
            return []
        
        # Load records from dataset
        try:
            table = pq.read_table(self.dataset_path)
            df = table.to_pandas()
            
            # Filter to only the records we need
            mask = df['record_id'].isin(record_ids)
            filtered_df = df[mask]
            
            # Preserve order based on record_ids
            filtered_df = filtered_df.set_index('record_id').reindex(record_ids).reset_index()
            
            # Deserialize JSON data
            records = []
            for data_json in filtered_df['data_json']:
                if pd.notna(data_json):
                    records.append(json.loads(data_json))
            
            return records
            
        except Exception:
            return None
    
    def _get_record_ids(self, step_name: str, operation_name: str) -> Optional[List[str]]:
        """Get record IDs for a specific checkpoint from the index."""
        try:
            index_table = pq.read_table(self.index_path)
            index_df = index_table.to_pandas()
            
            mask = (index_df['step_name'] == step_name) & (index_df['operation_name'] == operation_name)
            matching_rows = index_df[mask]
            
            if len(matching_rows) == 0:
                return None
            
            return matching_rows.iloc[0]['record_ids']
            
        except Exception:
            return None
    
    def save_batch_checkpoint(self, operation_name: str, batch_index: int, data: List[Dict[str, Any]]) -> str:
        """
        Save batch checkpoint data using the same incremental approach.
        
        Args:
            operation_name: Name of the operation
            batch_index: Index of the batch
            data: Batch data to save
            
        Returns:
            Path where the batch checkpoint was saved
        """
        # Use the same save mechanism with batch-specific naming
        return self.save_checkpoint(f"{operation_name}_batches", f"batch_{batch_index}", data)
    
    def list_checkpoints(self) -> List[Tuple[str, str]]:
        """
        List all available checkpoints.
        
        Returns:
            List of (step_name, operation_name) tuples for available checkpoints
        """
        try:
            index_table = pq.read_table(self.index_path)
            index_df = index_table.to_pandas()
            
            # Filter out batch checkpoints for the main listing
            mask = ~index_df['step_name'].str.endswith('_batches')
            main_checkpoints = index_df[mask]
            
            checkpoints = []
            for _, row in main_checkpoints.iterrows():
                checkpoints.append((row['step_name'], row['operation_name']))
            
            return sorted(checkpoints)
            
        except Exception:
            return []
    
    def load_checkpoint_as_dataframe(self, step_name: str, operation_name: str) -> Optional[pd.DataFrame]:
        """
        Load checkpoint data as a pandas DataFrame for analysis.
        
        Args:
            step_name: Name of the pipeline step
            operation_name: Name of the operation
            
        Returns:
            DataFrame or None if checkpoint doesn't exist
        """
        data = self.load_checkpoint(step_name, operation_name)
        if data is None:
            return None
        
        if not data:
            return pd.DataFrame()
        
        return pd.DataFrame(data)
    
    def get_checkpoint_info(self, step_name: str, operation_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a checkpoint.
        
        Args:
            step_name: Name of the pipeline step
            operation_name: Name of the operation
            
        Returns:
            Dictionary with checkpoint information or None if not found
        """
        try:
            index_table = pq.read_table(self.index_path)
            index_df = index_table.to_pandas()
            
            mask = (index_df['step_name'] == step_name) & (index_df['operation_name'] == operation_name)
            matching_rows = index_df[mask]
            
            if len(matching_rows) == 0:
                return None
            
            row = matching_rows.iloc[0]
            
            # Get dataset file size
            dataset_size = os.path.getsize(self.dataset_path) if os.path.exists(self.dataset_path) else 0
            
            # Load a sample record to get column info
            sample_data = self.load_checkpoint_sample(step_name, operation_name, n_rows=1)
            columns = list(sample_data[0].keys()) if sample_data else []
            
            return {
                'path': self.dataset_path,
                'size_bytes': dataset_size,  # Note: This is the total dataset size, not just this checkpoint
                'num_rows': int(row['num_records']),
                'num_columns': len(columns),
                'column_names': columns,
                'format': 'parquet',
                'created_at': row['created_at'],
                'storage_type': 'incremental'
            }
            
        except Exception:
            return None
    
    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all checkpoints.
        
        Returns:
            Dictionary with summary statistics
        """
        try:
            index_table = pq.read_table(self.index_path)
            index_df = index_table.to_pandas()
            
            # Filter out batch checkpoints
            mask = ~index_df['step_name'].str.endswith('_batches')
            main_checkpoints = index_df[mask]
            
            total_rows = main_checkpoints['num_records'].sum()
            dataset_size = os.path.getsize(self.dataset_path) if os.path.exists(self.dataset_path) else 0
            index_size = os.path.getsize(self.index_path) if os.path.exists(self.index_path) else 0
            
            checkpoint_details = []
            for _, row in main_checkpoints.iterrows():
                checkpoint_details.append({
                    'step_name': row['step_name'],
                    'operation_name': row['operation_name'],
                    'num_rows': int(row['num_records']),
                    'created_at': row['created_at']
                })
            
            return {
                'total_checkpoints': len(main_checkpoints),
                'total_size_bytes': dataset_size + index_size,
                'dataset_size_bytes': dataset_size,
                'index_size_bytes': index_size,
                'total_rows': int(total_rows),
                'checkpoints': checkpoint_details,
                'storage_type': 'incremental'
            }
            
        except Exception:
            return {
                'total_checkpoints': 0,
                'total_size_bytes': 0,
                'dataset_size_bytes': 0,
                'index_size_bytes': 0,
                'total_rows': 0,
                'checkpoints': [],
                'storage_type': 'incremental'
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
        data = self.load_checkpoint(step_name, operation_name)
        if data is None:
            return None
        
        return data[:n_rows]
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get detailed storage statistics showing efficiency of incremental storage.
        
        Returns:
            Storage efficiency statistics
        """
        try:
            # Get dataset stats
            dataset_table = pq.read_table(self.dataset_path)
            total_unique_records = dataset_table.num_rows
            
            # Get checkpoint stats
            index_table = pq.read_table(self.index_path)
            index_df = index_table.to_pandas()
            
            # Calculate total logical records (sum of all checkpoint sizes)
            total_logical_records = index_df['num_records'].sum()
            
            # Calculate deduplication ratio
            if total_logical_records > 0:
                deduplication_ratio = total_unique_records / total_logical_records
                space_saved_ratio = 1 - deduplication_ratio
            else:
                deduplication_ratio = 1.0
                space_saved_ratio = 0.0
            
            dataset_size = os.path.getsize(self.dataset_path)
            index_size = os.path.getsize(self.index_path)
            
            return {
                'total_unique_records': total_unique_records,
                'total_logical_records': int(total_logical_records),
                'deduplication_ratio': deduplication_ratio,
                'space_saved_ratio': space_saved_ratio,
                'dataset_size_bytes': dataset_size,
                'index_size_bytes': index_size,
                'total_size_bytes': dataset_size + index_size,
                'avg_record_size_bytes': dataset_size / total_unique_records if total_unique_records > 0 else 0
            }
            
        except Exception:
            return {
                'total_unique_records': 0,
                'total_logical_records': 0,
                'deduplication_ratio': 1.0,
                'space_saved_ratio': 0.0,
                'dataset_size_bytes': 0,
                'index_size_bytes': 0,
                'total_size_bytes': 0,
                'avg_record_size_bytes': 0
            }