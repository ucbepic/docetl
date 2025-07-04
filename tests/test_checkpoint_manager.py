import json
import os
import tempfile
import shutil
from typing import Dict, List, Any
import pytest  # type: ignore

import pyarrow as pa  # type: ignore
import pandas as pd  # type: ignore
from docetl.checkpoint_manager import CheckpointManager


class TestCheckpointManager:
    """Test suite for CheckpointManager functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        return [
            {"id": 1, "name": "Alice", "score": 95.5, "metadata": {"team": "A", "active": True}},
            {"id": 2, "name": "Bob", "score": 87.2, "metadata": {"team": "B", "active": False}},
            {"id": 3, "name": "Charlie", "score": 92.1, "metadata": {"team": "A", "active": True}},
            {"id": 4, "name": "Diana", "score": 88.9, "metadata": {"team": "C", "active": True}},
        ]
    
    @pytest.fixture
    def complex_data(self):
        """More complex sample data for testing."""
        return [
            {
                "user_id": "u1",
                "profile": {
                    "name": "John Doe",
                    "age": 30,
                    "preferences": ["music", "sports"],
                    "settings": {"theme": "dark", "notifications": True}
                },
                "activity": [
                    {"action": "login", "timestamp": "2023-01-01T10:00:00"},
                    {"action": "view_page", "timestamp": "2023-01-01T10:05:00", "page": "/dashboard"}
                ]
            },
            {
                "user_id": "u2", 
                "profile": {
                    "name": "Jane Smith",
                    "age": 25,
                    "preferences": ["reading", "travel"],
                    "settings": {"theme": "light", "notifications": False}
                },
                "activity": [
                    {"action": "login", "timestamp": "2023-01-01T11:00:00"},
                    {"action": "search", "timestamp": "2023-01-01T11:05:00", "query": "python tutorial"}
                ]
            }
        ]
    
    def test_checkpoint_manager_init(self, temp_dir):
        """Test CheckpointManager initialization."""
        manager = CheckpointManager(temp_dir)
        assert manager.intermediate_dir == temp_dir
    
    def test_incremental_storage_deduplication(self, temp_dir, sample_data):
        """Test that incremental storage properly deduplicates records."""
        manager = CheckpointManager(temp_dir)
        
        # Save same data twice with different step names
        manager.save_checkpoint("step1", "op1", sample_data)
        manager.save_checkpoint("step2", "op1", sample_data)  # Same data, different step
        
        # Both should load the same data
        data1 = manager.load_checkpoint("step1", "op1")
        data2 = manager.load_checkpoint("step2", "op1")
        assert data1 == data2 == sample_data
        
        # Check storage stats - should show deduplication
        stats = manager.get_storage_stats()
        assert stats['total_logical_records'] == len(sample_data) * 2  # Counted twice
        assert stats['total_unique_records'] == len(sample_data)  # But stored once
        assert stats['space_saved_ratio'] > 0  # Some space was saved
    
    
    def test_save_and_load_checkpoint(self, temp_dir, sample_data):
        """Test basic save and load functionality."""
        manager = CheckpointManager(temp_dir)
        
        # Save checkpoint
        path = manager.save_checkpoint("step1", "operation1", sample_data)
        assert os.path.exists(path)
        assert path.endswith("step1/operation1.parquet")
        
        # Load checkpoint
        loaded_data = manager.load_checkpoint("step1", "operation1")
        assert loaded_data is not None
        assert len(loaded_data) == len(sample_data)
        
        # Check data integrity (basic fields)
        for original, loaded in zip(sample_data, loaded_data):
            assert original["id"] == loaded["id"]
            assert original["name"] == loaded["name"]
            assert abs(original["score"] - loaded["score"]) < 0.01  # Float precision
    
    
    def test_save_and_load_complex_data(self, temp_dir, complex_data):
        """Test save and load with complex nested data structures."""
        manager = CheckpointManager(temp_dir)
        
        # Save checkpoint
        manager.save_checkpoint("step1", "complex_op", complex_data)
        
        # Load checkpoint
        loaded_data = manager.load_checkpoint("step1", "complex_op")
        assert loaded_data is not None
        assert len(loaded_data) == len(complex_data)
        
        # Check complex data integrity
        for original, loaded in zip(complex_data, loaded_data):
            assert original["user_id"] == loaded["user_id"]
            assert original["profile"] == loaded["profile"]
            assert original["activity"] == loaded["activity"]
    
    
    def test_load_nonexistent_checkpoint(self, temp_dir):
        """Test loading a checkpoint that doesn't exist."""
        manager = CheckpointManager(temp_dir)
        
        result = manager.load_checkpoint("nonexistent_step", "nonexistent_op")
        assert result is None
    
    
    def test_save_empty_data(self, temp_dir):
        """Test saving and loading empty data."""
        manager = CheckpointManager(temp_dir)
        
        # Save empty data
        manager.save_checkpoint("step1", "empty_op", [])
        
        # Load empty data
        loaded_data = manager.load_checkpoint("step1", "empty_op")
        assert loaded_data == []
    
    
    def test_batch_checkpoints(self, temp_dir, sample_data):
        """Test batch checkpoint functionality."""
        manager = CheckpointManager(temp_dir)
        
        # Save batch checkpoints
        path1 = manager.save_batch_checkpoint("batch_op", 0, sample_data[:2])
        path2 = manager.save_batch_checkpoint("batch_op", 1, sample_data[2:])
        
        assert os.path.exists(path1)
        assert os.path.exists(path2)
        assert "batch_op_batches" in path1
        assert "batch_0.parquet" in path1
        assert "batch_1.parquet" in path2
    
    
    def test_list_checkpoints(self, temp_dir, sample_data):
        """Test listing available checkpoints."""
        manager = CheckpointManager(temp_dir)
        
        # Initially no checkpoints
        checkpoints = manager.list_checkpoints()
        assert checkpoints == []
        
        # Save some checkpoints
        manager.save_checkpoint("step1", "op1", sample_data)
        manager.save_checkpoint("step1", "op2", sample_data)
        manager.save_checkpoint("step2", "op1", sample_data)
        
        # List checkpoints
        checkpoints = manager.list_checkpoints()
        expected = [("step1", "op1"), ("step1", "op2"), ("step2", "op1")]
        assert sorted(checkpoints) == sorted(expected)
    
    
    def test_load_checkpoint_as_dataframe(self, temp_dir, sample_data):
        """Test loading checkpoint as pandas DataFrame."""
        manager = CheckpointManager(temp_dir)
        
        # Save checkpoint
        manager.save_checkpoint("step1", "op1", sample_data)
        
        # Load as DataFrame
        df = manager.load_checkpoint_as_dataframe("step1", "op1")
        assert df is not None
        assert len(df) == len(sample_data)
        assert "id" in df.columns
        assert "name" in df.columns
        assert "score" in df.columns
        
        # Check data values
        assert df["id"].tolist() == [1, 2, 3, 4]
        assert df["name"].tolist() == ["Alice", "Bob", "Charlie", "Diana"]
    
    
    def test_get_checkpoint_info(self, temp_dir, sample_data):
        """Test getting checkpoint information."""
        manager = CheckpointManager(temp_dir)
        
        # Save checkpoint
        manager.save_checkpoint("step1", "op1", sample_data)
        
        # Get checkpoint info
        info = manager.get_checkpoint_info("step1", "op1")
        assert info is not None
        assert info["format"] == "parquet"
        assert info["num_rows"] == len(sample_data)
        assert info["size_bytes"] > 0
        assert "column_names" in info
        assert "schema" in info
        
        # Test nonexistent checkpoint
        info = manager.get_checkpoint_info("nonexistent", "nonexistent")
        assert info is None
    
    
    def test_get_checkpoint_summary(self, temp_dir, sample_data):
        """Test getting checkpoint summary."""
        manager = CheckpointManager(temp_dir)
        
        # Initially no checkpoints
        summary = manager.get_checkpoint_summary()
        assert summary["total_checkpoints"] == 0
        assert summary["total_size_bytes"] == 0
        assert summary["total_rows"] == 0
        
        # Save some checkpoints
        manager.save_checkpoint("step1", "op1", sample_data)
        manager.save_checkpoint("step2", "op1", sample_data[:2])
        
        # Get summary
        summary = manager.get_checkpoint_summary()
        assert summary["total_checkpoints"] == 2
        assert summary["total_size_bytes"] > 0
        assert summary["total_rows"] == len(sample_data) + 2
        assert len(summary["checkpoints"]) == 2
    
    
    def test_load_checkpoint_sample(self, temp_dir, sample_data):
        """Test loading a sample of checkpoint data."""
        manager = CheckpointManager(temp_dir)
        
        # Save checkpoint
        manager.save_checkpoint("step1", "op1", sample_data)
        
        # Load sample (smaller than data size)
        sample = manager.load_checkpoint_sample("step1", "op1", n_rows=2)
        assert sample is not None
        assert len(sample) == 2
        assert sample[0]["id"] == 1
        assert sample[1]["id"] == 2
        
        # Load sample (larger than data size)
        sample = manager.load_checkpoint_sample("step1", "op1", n_rows=10)
        assert sample is not None
        assert len(sample) == len(sample_data)  # Should return all data
        
        # Test nonexistent checkpoint
        sample = manager.load_checkpoint_sample("nonexistent", "nonexistent")
        assert sample is None
    
    
    def test_data_type_preservation(self, temp_dir):
        """Test that different data types are preserved correctly."""
        manager = CheckpointManager(temp_dir)
        
        # Test data with various types
        test_data = [
            {
                "string_field": "hello",
                "int_field": 42,
                "float_field": 3.14159,
                "bool_field": True,
                "null_field": None,
                "list_field": [1, 2, 3],
                "dict_field": {"nested": "value"}
            },
            {
                "string_field": "world", 
                "int_field": 0,
                "float_field": -1.5,
                "bool_field": False,
                "null_field": None,
                "list_field": ["a", "b"],
                "dict_field": {"another": "nested", "number": 123}
            }
        ]
        
        # Save and load
        manager.save_checkpoint("types_test", "op1", test_data)
        loaded_data = manager.load_checkpoint("types_test", "op1")
        
        assert loaded_data is not None
        assert len(loaded_data) == 2
        
        # Check first record
        record1 = loaded_data[0]
        assert record1["string_field"] == "hello"
        assert record1["int_field"] == 42
        assert abs(record1["float_field"] - 3.14159) < 0.00001
        assert record1["bool_field"] == True
        assert record1["null_field"] is None
        assert record1["list_field"] == [1, 2, 3]
        assert record1["dict_field"] == {"nested": "value"}
    
    
    def test_checkpoint_path_structure(self, temp_dir, sample_data):
        """Test that checkpoint files are saved in correct directory structure."""
        manager = CheckpointManager(temp_dir)
        
        # Save checkpoint
        path = manager.save_checkpoint("my_step", "my_operation", sample_data)
        
        # Check path structure
        expected_path = os.path.join(temp_dir, "my_step", "my_operation.parquet")
        assert path == expected_path
        assert os.path.exists(expected_path)
        
        # Check that directory was created
        step_dir = os.path.join(temp_dir, "my_step")
        assert os.path.isdir(step_dir)
    
     
    def test_checkpoint_overwrite(self, temp_dir, sample_data):
        """Test that checkpoints can be overwritten."""
        manager = CheckpointManager(temp_dir)
        
        # Save initial checkpoint
        manager.save_checkpoint("step1", "op1", sample_data)
        original_data = manager.load_checkpoint("step1", "op1")
        
        # Overwrite with different data
        new_data = [{"id": 999, "name": "New Person", "score": 100.0, "metadata": {}}]
        manager.save_checkpoint("step1", "op1", new_data)
        
        # Load and verify overwrite worked
        loaded_data = manager.load_checkpoint("step1", "op1")
        assert loaded_data is not None
        assert len(loaded_data) == 1
        assert loaded_data[0]["id"] == 999
        assert loaded_data[0]["name"] == "New Person"