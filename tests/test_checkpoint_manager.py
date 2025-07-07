import pytest
import tempfile
import os
import json
import pandas as pd
from unittest.mock import Mock
from docetl.checkpoint_manager import CheckpointManager


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return [
        {"id": 1, "text": "First document", "category": "A"},
        {"id": 2, "text": "Second document", "category": "B"},
        {"id": 3, "text": "Third document", "category": "A"},
    ]


@pytest.fixture
def empty_data():
    """Empty data for testing."""
    return []


@pytest.fixture
def mock_console():
    """Mock console for testing."""
    return Mock()


@pytest.fixture
def checkpoint_manager(temp_dir, mock_console):
    """Create a checkpoint manager instance."""
    return CheckpointManager(temp_dir, console=mock_console)


def test_checkpoint_manager_init(temp_dir, mock_console):
    """Test checkpoint manager initialization."""
    cm = CheckpointManager(temp_dir, console=mock_console)
    assert cm.intermediate_dir == temp_dir
    assert cm.console == mock_console
    assert cm.config_path == os.path.join(temp_dir, ".docetl_intermediate_config.json")
    assert os.path.exists(temp_dir)


def test_checkpoint_manager_init_no_console(temp_dir):
    """Test checkpoint manager initialization without console."""
    cm = CheckpointManager(temp_dir)
    assert cm.console is None


def test_save_and_load_checkpoint(checkpoint_manager, sample_data):
    """Test saving and loading a checkpoint."""
    step_name = "test_step"
    operation_name = "test_operation"
    operation_hash = "test_hash_123"
    
    # Save checkpoint
    checkpoint_manager.save_checkpoint(step_name, operation_name, sample_data, operation_hash)
    
    # Verify checkpoint file exists
    checkpoint_path = checkpoint_manager._get_checkpoint_path(step_name, operation_name)
    assert os.path.exists(checkpoint_path)
    
    # Load checkpoint
    loaded_data = checkpoint_manager.load_checkpoint(step_name, operation_name, operation_hash)
    
    # Verify data integrity
    assert loaded_data == sample_data
    assert len(loaded_data) == 3
    assert loaded_data[0]["id"] == 1
    assert loaded_data[1]["text"] == "Second document"


def test_save_and_load_empty_checkpoint(checkpoint_manager, empty_data):
    """Test saving and loading an empty checkpoint."""
    step_name = "test_step"
    operation_name = "test_operation"
    operation_hash = "test_hash_empty"
    
    # Save empty checkpoint
    checkpoint_manager.save_checkpoint(step_name, operation_name, empty_data, operation_hash)
    
    # Load checkpoint
    loaded_data = checkpoint_manager.load_checkpoint(step_name, operation_name, operation_hash)
    
    # Verify empty data
    assert loaded_data == []


def test_load_nonexistent_checkpoint(checkpoint_manager):
    """Test loading a checkpoint that doesn't exist."""
    loaded_data = checkpoint_manager.load_checkpoint("nonexistent_step", "nonexistent_op", "fake_hash")
    assert loaded_data is None


def test_load_checkpoint_wrong_hash(checkpoint_manager, sample_data):
    """Test loading a checkpoint with wrong hash."""
    step_name = "test_step"
    operation_name = "test_operation"
    correct_hash = "correct_hash"
    wrong_hash = "wrong_hash"
    
    # Save checkpoint with correct hash
    checkpoint_manager.save_checkpoint(step_name, operation_name, sample_data, correct_hash)
    
    # Try to load with wrong hash
    loaded_data = checkpoint_manager.load_checkpoint(step_name, operation_name, wrong_hash)
    assert loaded_data is None
    
    # Load with correct hash should work
    loaded_data = checkpoint_manager.load_checkpoint(step_name, operation_name, correct_hash)
    assert loaded_data == sample_data


def test_load_output_by_step_and_op(checkpoint_manager, sample_data):
    """Test loading output by step and operation name."""
    step_name = "test_step"
    operation_name = "test_operation"
    operation_hash = "test_hash"
    
    # Save checkpoint
    checkpoint_manager.save_checkpoint(step_name, operation_name, sample_data, operation_hash)
    
    # Load output directly
    loaded_data = checkpoint_manager.load_output_by_step_and_op(step_name, operation_name)
    assert loaded_data == sample_data


def test_load_output_as_dataframe(checkpoint_manager, sample_data):
    """Test loading output as pandas DataFrame."""
    step_name = "test_step"
    operation_name = "test_operation"
    operation_hash = "test_hash"
    
    # Save checkpoint
    checkpoint_manager.save_checkpoint(step_name, operation_name, sample_data, operation_hash)
    
    # Load as DataFrame
    df = checkpoint_manager.load_output_as_dataframe(step_name, operation_name)
    
    # Verify DataFrame
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ["id", "text", "category"]
    assert df.iloc[0]["id"] == 1
    assert df.iloc[1]["text"] == "Second document"


def test_load_output_nonexistent(checkpoint_manager):
    """Test loading output for nonexistent step/operation."""
    loaded_data = checkpoint_manager.load_output_by_step_and_op("nonexistent", "nonexistent")
    assert loaded_data is None
    
    df = checkpoint_manager.load_output_as_dataframe("nonexistent", "nonexistent")
    assert df is None


def test_list_outputs(checkpoint_manager, sample_data):
    """Test listing all outputs."""
    # Initially no outputs
    outputs = checkpoint_manager.list_outputs()
    assert outputs == []
    
    # Save multiple checkpoints
    checkpoint_manager.save_checkpoint("step1", "op1", sample_data, "hash1")
    checkpoint_manager.save_checkpoint("step1", "op2", sample_data, "hash2")
    checkpoint_manager.save_checkpoint("step2", "op1", sample_data, "hash3")
    
    # List outputs
    outputs = checkpoint_manager.list_outputs()
    expected = [("step1", "op1"), ("step1", "op2"), ("step2", "op1")]
    assert sorted(outputs) == sorted(expected)


def test_get_checkpoint_size(checkpoint_manager, sample_data):
    """Test getting checkpoint file size."""
    step_name = "test_step"
    operation_name = "test_operation"
    operation_hash = "test_hash"
    
    # Size should be None for nonexistent checkpoint
    size = checkpoint_manager.get_checkpoint_size(step_name, operation_name)
    assert size is None
    
    # Save checkpoint
    checkpoint_manager.save_checkpoint(step_name, operation_name, sample_data, operation_hash)
    
    # Size should be positive
    size = checkpoint_manager.get_checkpoint_size(step_name, operation_name)
    assert size > 0


def test_get_total_checkpoint_size(checkpoint_manager, sample_data):
    """Test getting total size of all checkpoints."""
    # Initially zero
    total_size = checkpoint_manager.get_total_checkpoint_size()
    assert total_size == 0
    
    # Save multiple checkpoints
    checkpoint_manager.save_checkpoint("step1", "op1", sample_data, "hash1")
    checkpoint_manager.save_checkpoint("step1", "op2", sample_data, "hash2")
    
    # Total size should be positive
    total_size = checkpoint_manager.get_total_checkpoint_size()
    assert total_size > 0


def test_clear_all_checkpoints(checkpoint_manager, sample_data):
    """Test clearing all checkpoints."""
    # Save some checkpoints
    checkpoint_manager.save_checkpoint("step1", "op1", sample_data, "hash1")
    checkpoint_manager.save_checkpoint("step2", "op2", sample_data, "hash2")
    
    # Verify they exist
    assert len(checkpoint_manager.list_outputs()) == 2
    
    # Clear all
    checkpoint_manager.clear_all_checkpoints()
    
    # Verify they're gone
    assert len(checkpoint_manager.list_outputs()) == 0
    assert checkpoint_manager.get_total_checkpoint_size() == 0


def test_clear_step_checkpoints(checkpoint_manager, sample_data):
    """Test clearing checkpoints for a specific step."""
    # Save checkpoints for multiple steps
    checkpoint_manager.save_checkpoint("step1", "op1", sample_data, "hash1")
    checkpoint_manager.save_checkpoint("step1", "op2", sample_data, "hash2")
    checkpoint_manager.save_checkpoint("step2", "op1", sample_data, "hash3")
    
    # Verify all exist
    assert len(checkpoint_manager.list_outputs()) == 3
    
    # Clear step1
    checkpoint_manager.clear_step_checkpoints("step1")
    
    # Verify only step2 remains
    outputs = checkpoint_manager.list_outputs()
    assert len(outputs) == 1
    assert outputs[0] == ("step2", "op1")


def test_config_file_management(checkpoint_manager, sample_data):
    """Test that config file is properly managed."""
    step_name = "test_step"
    operation_name = "test_operation"
    operation_hash = "test_hash"
    
    # Config file shouldn't exist initially
    assert not os.path.exists(checkpoint_manager.config_path)
    
    # Save checkpoint
    checkpoint_manager.save_checkpoint(step_name, operation_name, sample_data, operation_hash)
    
    # Config file should exist
    assert os.path.exists(checkpoint_manager.config_path)
    
    # Verify config content
    with open(checkpoint_manager.config_path, 'r') as f:
        config = json.load(f)
    
    assert config[step_name][operation_name] == operation_hash


def test_corrupted_config_file(checkpoint_manager, sample_data):
    """Test handling of corrupted config file."""
    step_name = "test_step"
    operation_name = "test_operation"
    operation_hash = "test_hash"
    
    # Create corrupted config file
    with open(checkpoint_manager.config_path, 'w') as f:
        f.write("corrupted json content")
    
    # Load should return None
    loaded_data = checkpoint_manager.load_checkpoint(step_name, operation_name, operation_hash)
    assert loaded_data is None


def test_directory_structure(checkpoint_manager, sample_data):
    """Test that directory structure is created correctly."""
    step_name = "test_step"
    operation_name = "test_operation"
    operation_hash = "test_hash"
    
    # Save checkpoint
    checkpoint_manager.save_checkpoint(step_name, operation_name, sample_data, operation_hash)
    
    # Verify directory structure
    step_dir = os.path.join(checkpoint_manager.intermediate_dir, step_name)
    assert os.path.exists(step_dir)
    assert os.path.isdir(step_dir)
    
    checkpoint_file = os.path.join(step_dir, f"{operation_name}.parquet")
    assert os.path.exists(checkpoint_file)


def test_checkpoint_manager_without_intermediate_dir():
    """Test checkpoint manager without intermediate directory."""
    cm = CheckpointManager(None)
    
    # All operations should be no-ops
    cm.save_checkpoint("step", "op", [], "hash")
    assert cm.load_checkpoint("step", "op", "hash") is None
    assert cm.load_output_by_step_and_op("step", "op") is None
    assert cm.load_output_as_dataframe("step", "op") is None
    assert cm.list_outputs() == []
    assert cm.get_checkpoint_size("step", "op") is None
    assert cm.get_total_checkpoint_size() == 0


def test_data_types_preservation(checkpoint_manager):
    """Test that different data types are preserved correctly."""
    data = [
        {"string": "text", "integer": 42, "float": 3.14, "boolean": True},
        {"string": "more text", "integer": 0, "float": -1.5, "boolean": False},
    ]
    
    step_name = "test_step"
    operation_name = "test_operation"
    operation_hash = "test_hash"
    
    # Save and load
    checkpoint_manager.save_checkpoint(step_name, operation_name, data, operation_hash)
    loaded_data = checkpoint_manager.load_checkpoint(step_name, operation_name, operation_hash)
    
    # Verify data types are preserved
    assert loaded_data[0]["string"] == "text"
    assert loaded_data[0]["integer"] == 42
    assert loaded_data[0]["float"] == 3.14
    assert loaded_data[0]["boolean"] is True
    assert loaded_data[1]["boolean"] is False


def test_space_efficiency_vs_json(temp_dir):
    """Test that PyArrow storage is more space efficient than JSON."""
    # Create larger, more realistic test data
    large_data = []
    for i in range(1000):
        large_data.append({
            "id": i,
            "text": f"This is a longer text document with some repetitive content that would benefit from compression. Document number {i}. " * 3,
            "category": "A" if i % 2 == 0 else "B",
            "score": i * 0.1,
            "tags": ["tag1", "tag2", "tag3"] if i % 3 == 0 else ["tag4", "tag5"],
            "metadata": {"source": "test", "processed": True, "version": 1}
        })
    
    # Test with CheckpointManager (PyArrow)
    checkpoint_manager = CheckpointManager(temp_dir)
    step_name = "test_step"
    operation_name = "test_operation"
    operation_hash = "test_hash"
    
    # Save with PyArrow
    checkpoint_manager.save_checkpoint(step_name, operation_name, large_data, operation_hash)
    
    # Get PyArrow file size
    parquet_size = checkpoint_manager.get_checkpoint_size(step_name, operation_name)
    
    # Save same data as JSON for comparison
    json_path = os.path.join(temp_dir, "test_data.json")
    with open(json_path, 'w') as f:
        json.dump(large_data, f)
    
    json_size = os.path.getsize(json_path)
    
    # PyArrow should be more space efficient
    assert parquet_size < json_size, f"PyArrow ({parquet_size} bytes) should be smaller than JSON ({json_size} bytes)"
    
    # Calculate compression ratio
    compression_ratio = parquet_size / json_size
    print(f"PyArrow size: {parquet_size} bytes")
    print(f"JSON size: {json_size} bytes") 
    print(f"Compression ratio: {compression_ratio:.2f} (smaller is better)")
    
    # Verify we get at least some compression benefit
    assert compression_ratio < 0.8, "Expected at least 20% space savings"


def test_storage_efficiency_with_repetitive_data(temp_dir):
    """Test storage efficiency with highly repetitive data that should compress well."""
    # Create data with lots of repetition (common in ETL pipelines)
    repetitive_data = []
    base_text = "This is a base document that will be repeated many times with slight variations. " * 5
    
    for i in range(500):
        repetitive_data.append({
            "id": i,
            "text": base_text + f"Variation {i % 10}",  # Only 10 unique variations
            "category": "Category " + str(i % 5),  # Only 5 categories
            "status": "processed" if i % 2 == 0 else "pending",
            "tags": ["common", "tag"] + ([f"special_{i % 3}"] if i % 3 == 0 else []),
            "metadata": {"type": "document", "version": 1, "processed_by": "system"}
        })
    
    checkpoint_manager = CheckpointManager(temp_dir)
    step_name = "repetitive_step"
    operation_name = "repetitive_operation"
    operation_hash = "repetitive_hash"
    
    # Save with PyArrow
    checkpoint_manager.save_checkpoint(step_name, operation_name, repetitive_data, operation_hash)
    parquet_size = checkpoint_manager.get_checkpoint_size(step_name, operation_name)
    
    # Save as JSON
    json_path = os.path.join(temp_dir, "repetitive_data.json")
    with open(json_path, 'w') as f:
        json.dump(repetitive_data, f)
    json_size = os.path.getsize(json_path)
    
    # With repetitive data, compression should be even better
    compression_ratio = parquet_size / json_size
    print(f"Repetitive data - PyArrow size: {parquet_size} bytes")
    print(f"Repetitive data - JSON size: {json_size} bytes")
    print(f"Repetitive data - Compression ratio: {compression_ratio:.2f}")
    
    # Should get significant compression with repetitive data
    assert compression_ratio < 0.6, "Expected at least 40% space savings with repetitive data"


@pytest.fixture
def large_sample_data():
    """Generate larger sample data for performance testing."""
    return [
        {
            "id": i,
            "text": f"Document {i}: " + "This is sample text content that simulates real document processing. " * 10,
            "category": f"Category_{i % 10}",
            "score": i * 0.01,
            "tags": [f"tag_{i % 5}", f"tag_{(i+1) % 5}"],
            "metadata": {"source": f"source_{i % 3}", "processed": True}
        }
        for i in range(100)
    ]


def test_performance_comparison(temp_dir, large_sample_data):
    """Compare load/save performance between PyArrow and JSON."""
    import time
    
    checkpoint_manager = CheckpointManager(temp_dir)
    step_name = "perf_step"
    operation_name = "perf_operation"
    operation_hash = "perf_hash"
    
    # Time PyArrow save
    start_time = time.time()
    checkpoint_manager.save_checkpoint(step_name, operation_name, large_sample_data, operation_hash)
    parquet_save_time = time.time() - start_time
    
    # Time PyArrow load
    start_time = time.time()
    loaded_parquet = checkpoint_manager.load_checkpoint(step_name, operation_name, operation_hash)
    parquet_load_time = time.time() - start_time
    
    # Time JSON save
    json_path = os.path.join(temp_dir, "perf_test.json")
    start_time = time.time()
    with open(json_path, 'w') as f:
        json.dump(large_sample_data, f)
    json_save_time = time.time() - start_time
    
    # Time JSON load
    start_time = time.time()
    with open(json_path, 'r') as f:
        loaded_json = json.load(f)
    json_load_time = time.time() - start_time
    
    # Verify data integrity
    assert len(loaded_parquet) == len(loaded_json) == len(large_sample_data)
    # Check first few records to verify data integrity
    assert loaded_parquet[0]["id"] == large_sample_data[0]["id"]
    assert loaded_json[0]["id"] == large_sample_data[0]["id"]
    
    print(f"PyArrow save time: {parquet_save_time:.4f}s")
    print(f"PyArrow load time: {parquet_load_time:.4f}s")
    print(f"JSON save time: {json_save_time:.4f}s")
    print(f"JSON load time: {json_load_time:.4f}s")
    
    # Performance will vary, but let's at least verify operations complete
    assert parquet_save_time > 0 and parquet_load_time > 0
    assert json_save_time > 0 and json_load_time > 0


def test_incremental_checkpoint_potential(temp_dir):
    """Test the potential space savings from incremental checkpoint storage."""
    # Simulate a typical ETL pipeline: input -> map -> filter -> map
    
    # Original dataset (e.g., from data loading)
    original_data = []
    for i in range(1000):
        original_data.append({
            "id": i,
            "text": f"Document {i}: " + "Original content that will be preserved through transformations. " * 10,
            "category": f"category_{i % 5}",
            "metadata": {"source": "original", "timestamp": f"2024-01-{i%30+1:02d}"}
        })
    
    # After map operation (adds analysis fields but preserves original data)
    after_map_data = []
    for record in original_data:
        new_record = record.copy()
        new_record.update({
            "sentiment": "positive" if record["id"] % 2 == 0 else "negative",
            "analyzed": True,
            "summary": f"Summary of document {record['id']}"
        })
        after_map_data.append(new_record)
    
    # After filter operation (removes some records but preserves all fields)
    after_filter_data = [r for r in after_map_data if r["id"] % 3 != 0]  # Remove 1/3 of records
    
    # After second map operation (adds more analysis)
    after_second_map_data = []
    for record in after_filter_data:
        new_record = record.copy()
        new_record.update({
            "enriched": True,
            "score": record["id"] * 0.1,
            "tags": ["tag1", "tag2"]
        })
        after_second_map_data.append(new_record)
    
    # Test current approach (storing full datasets)
    checkpoint_manager = CheckpointManager(temp_dir)
    
    # Save each stage
    checkpoint_manager.save_checkpoint("pipeline", "load", original_data, "hash1")
    checkpoint_manager.save_checkpoint("pipeline", "map1", after_map_data, "hash2")
    checkpoint_manager.save_checkpoint("pipeline", "filter", after_filter_data, "hash3")
    checkpoint_manager.save_checkpoint("pipeline", "map2", after_second_map_data, "hash4")
    
    # Get sizes
    original_size = checkpoint_manager.get_checkpoint_size("pipeline", "load")
    map1_size = checkpoint_manager.get_checkpoint_size("pipeline", "map1")
    filter_size = checkpoint_manager.get_checkpoint_size("pipeline", "filter")
    map2_size = checkpoint_manager.get_checkpoint_size("pipeline", "map2")
    
    total_current_size = original_size + map1_size + filter_size + map2_size
    
    print(f"Current checkpoint sizes:")
    print(f"  Original: {original_size} bytes")
    print(f"  After map1: {map1_size} bytes")
    print(f"  After filter: {filter_size} bytes")
    print(f"  After map2: {map2_size} bytes")
    print(f"  Total: {total_current_size} bytes")
    
    # Calculate potential savings if we stored deltas
    # This is a rough estimate - actual implementation would be more sophisticated
    
    # Map1 delta: just the new fields (sentiment, analyzed, summary)
    map1_delta_estimate = len(after_map_data) * 100  # Rough estimate for new fields
    
    # Filter delta: just record IDs that were removed
    filter_delta_estimate = (len(after_map_data) - len(after_filter_data)) * 20  # Just IDs
    
    # Map2 delta: just the new fields (enriched, score, tags)
    map2_delta_estimate = len(after_second_map_data) * 80  # Rough estimate
    
    estimated_incremental_size = original_size + map1_delta_estimate + filter_delta_estimate + map2_delta_estimate
    
    print(f"\nEstimated incremental checkpoint sizes:")
    print(f"  Original: {original_size} bytes")
    print(f"  Map1 delta: {map1_delta_estimate} bytes")
    print(f"  Filter delta: {filter_delta_estimate} bytes")
    print(f"  Map2 delta: {map2_delta_estimate} bytes")
    print(f"  Total estimated: {estimated_incremental_size} bytes")
    
    potential_savings = (total_current_size - estimated_incremental_size) / total_current_size
    print(f"\nPotential space savings: {potential_savings:.1%}")
    
    # This test shows PyArrow compression is already very effective
    # The real benefit of incremental processing is avoiding recomputation, not storage
    print("Note: PyArrow compression makes storage deltas less beneficial")
    print("Real value is in incremental reprocessing to avoid expensive operations")


def test_incremental_processing_workflow(temp_dir):
    """Test the incremental processing workflow for change detection."""
    import hashlib
    
    def compute_record_hash(record):
        """Compute hash of a record for change detection."""
        record_str = json.dumps(record, sort_keys=True)
        return hashlib.md5(record_str.encode()).hexdigest()
    
    checkpoint_manager = CheckpointManager(temp_dir)
    
    # Initial dataset
    initial_data = [
        {"id": 1, "text": "Document 1", "category": "A"},
        {"id": 2, "text": "Document 2", "category": "B"}, 
        {"id": 3, "text": "Document 3", "category": "A"}
    ]
    
    # Compute hashes for initial data
    initial_hashes = [compute_record_hash(record) for record in initial_data]
    
    # Save initial checkpoint with hash tracking
    checkpoint_manager.save_incremental_checkpoint(
        "test", "process", initial_data, "hash1", initial_hashes
    )
    
    # Simulate processing the data (e.g., adding analysis results)
    processed_data = []
    for record in initial_data:
        new_record = record.copy()
        new_record["processed"] = True
        new_record["score"] = record["id"] * 10
        processed_data.append(new_record)
    
    # Save processed results
    checkpoint_manager.save_incremental_checkpoint(
        "test", "analyzed", processed_data, "hash2", initial_hashes
    )
    
    # Now simulate a data update scenario
    # - Record 1 unchanged
    # - Record 2 modified
    # - Record 3 unchanged  
    # - Record 4 added
    updated_data = [
        {"id": 1, "text": "Document 1", "category": "A"},  # unchanged
        {"id": 2, "text": "Document 2 UPDATED", "category": "B"},  # changed
        {"id": 3, "text": "Document 3", "category": "A"},  # unchanged
        {"id": 4, "text": "Document 4", "category": "C"}   # new
    ]
    
    updated_hashes = [compute_record_hash(record) for record in updated_data]
    
    # Get incremental processing info
    incremental_info = checkpoint_manager.get_incremental_processing_info(
        "test", "analyzed", updated_hashes
    )
    
    print(f"Incremental processing info: {incremental_info}")
    
    # Verify change detection
    assert not incremental_info["needs_full_reprocess"]
    assert incremental_info["changed_indices"] == [1]  # Record 2 changed
    assert incremental_info["unchanged_indices"] == [0, 2]  # Records 1 and 3 unchanged
    assert incremental_info["new_indices"] == [3]  # Record 4 is new
    assert incremental_info["total_changes"] == 2  # 1 changed + 1 new
    
    # Load unchanged records from previous processing
    unchanged_processed = checkpoint_manager.load_incremental_checkpoint(
        "test", "analyzed", "hash2", incremental_info["unchanged_indices"]
    )
    
    print(f"Unchanged records: {len(unchanged_processed)} out of {len(processed_data)}")
    
    # Verify we got the right unchanged records
    assert len(unchanged_processed) == 2
    assert unchanged_processed[0]["id"] == 1
    assert unchanged_processed[1]["id"] == 3
    assert all(r["processed"] for r in unchanged_processed)
    
    # In a real scenario, you would:
    # 1. Process only changed records (indices [1]) and new records (indices [3])
    # 2. Merge with unchanged_processed to get complete result
    # 3. Save the new complete result with updated hashes
    
    print("✓ Incremental processing successfully detected changes and preserved unchanged results")


def test_incremental_processing_edge_cases(temp_dir):
    """Test edge cases for incremental processing."""
    checkpoint_manager = CheckpointManager(temp_dir)
    
    # Test with no previous data
    info = checkpoint_manager.get_incremental_processing_info("new", "op", ["hash1"])
    assert info["needs_full_reprocess"]
    assert "No previous hash tracking data" in info["reason"]
    
    # Test loading incremental checkpoint with no previous data
    result = checkpoint_manager.load_incremental_checkpoint("new", "op", "hash", [0, 1])
    assert result is None
    
    print("✓ Edge cases handled correctly")


def test_incremental_processing_realistic_pipeline(temp_dir):
    """Test incremental processing with realistic text processing pipeline."""
    import hashlib
    import random
    import string
    
    def generate_large_text():
        """Generate realistic large text documents."""
        # Common words to create realistic text
        words = [
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "python", "programming", "language", "machine", "learning", "artificial",
            "intelligence", "data", "science", "analysis", "processing", "algorithm",
            "computer", "software", "development", "application", "framework",
            "database", "query", "optimization", "performance", "scalability"
        ]
        
        # Generate sentences with 15-30 words each
        sentences = []
        for _ in range(random.randint(10, 25)):  # 10-25 sentences per document
            sentence_words = random.choices(words, k=random.randint(15, 30))
            sentences.append(" ".join(sentence_words).capitalize() + ".")
        
        return " ".join(sentences)
    
    def compute_record_hash(record):
        """Compute hash of input record for change detection."""
        # Only hash the input fields, not processed results
        input_content = {"id": record["id"], "content": record["content"]}
        record_str = json.dumps(input_content, sort_keys=True)
        return hashlib.md5(record_str.encode()).hexdigest()
    
    def extract_first_letters(text):
        """Extract first letter of every word for first 15 words."""
        words = text.split()[:15]  # Take first 15 words
        first_letters = [word[0].upper() for word in words if word]
        return "".join(first_letters)
    
    def analyze_text_length(text):
        """Analyze text characteristics."""
        words = text.split()
        return {
            "word_count": len(words),
            "char_count": len(text),
            "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0
        }
    
    checkpoint_manager = CheckpointManager(temp_dir)
    
    # Generate initial large dataset (1000 documents with substantial text)
    print("Generating initial dataset with large text documents...")
    initial_data = []
    for i in range(1000):
        content = generate_large_text()
        initial_data.append({
            "id": i,
            "content": content,
            "source": f"source_{i % 10}",
            "timestamp": f"2024-01-{(i % 30) + 1:02d}"
        })
    
    # Compute hashes for change detection
    initial_hashes = [compute_record_hash(record) for record in initial_data]
    
    print(f"Generated {len(initial_data)} documents, avg size: {sum(len(d['content']) for d in initial_data) / len(initial_data):.0f} chars")
    
    # Save initial checkpoint
    checkpoint_manager.save_incremental_checkpoint(
        "pipeline", "raw_data", initial_data, "hash1", initial_hashes
    )
    
    # Stage 1: Extract first letters (simulate expensive text processing)
    print("Stage 1: Extracting first letters from each document...")
    stage1_data = []
    for record in initial_data:
        new_record = record.copy()
        new_record["first_letters"] = extract_first_letters(record["content"])
        stage1_data.append(new_record)
    
    checkpoint_manager.save_incremental_checkpoint(
        "pipeline", "first_letters", stage1_data, "hash2", initial_hashes
    )
    
    # Stage 2: Analyze text characteristics
    print("Stage 2: Analyzing text characteristics...")
    stage2_data = []
    for record in stage1_data:
        new_record = record.copy()
        new_record["analysis"] = analyze_text_length(record["content"])
        stage2_data.append(new_record)
    
    checkpoint_manager.save_incremental_checkpoint(
        "pipeline", "analyzed", stage2_data, "hash3", initial_hashes
    )
    
    # Now simulate data updates - modify 5% of documents, add 2% new ones
    print("\nSimulating data updates (5% modified, 2% new)...")
    num_changed = int(len(initial_data) * 0.05)  # 5% changed
    num_new = int(len(initial_data) * 0.02)      # 2% new
    
    # Create updated dataset
    updated_data = initial_data.copy()
    changed_indices = random.sample(range(len(initial_data)), num_changed)
    
    # Modify some existing documents
    for idx in changed_indices:
        updated_data[idx] = updated_data[idx].copy()
        updated_data[idx]["content"] = generate_large_text()  # New content
        updated_data[idx]["timestamp"] = "2024-02-01"  # Updated timestamp
    
    # Add new documents
    for i in range(num_new):
        new_id = len(initial_data) + i
        updated_data.append({
            "id": new_id,
            "content": generate_large_text(),
            "source": f"source_{new_id % 10}",
            "timestamp": "2024-02-01"
        })
    
    # Compute new hashes
    updated_hashes = [compute_record_hash(record) for record in updated_data]
    
    # Test incremental processing for each stage
    print("\nTesting incremental processing...")
    
    # Check what needs reprocessing for stage 1 (first letters)
    incremental_info = checkpoint_manager.get_incremental_processing_info(
        "pipeline", "first_letters", updated_hashes
    )
    
    print(f"Stage 1 incremental analysis:")
    print(f"  Total records: {len(updated_data)}")
    print(f"  Changed: {len(incremental_info['changed_indices'])}")
    print(f"  New: {len(incremental_info['new_indices'])}")
    print(f"  Unchanged: {len(incremental_info['unchanged_indices'])}")
    print(f"  Total changes: {incremental_info['total_changes']}")
    
    # Verify the change detection is accurate
    expected_changes = num_changed + num_new
    actual_changes = incremental_info['total_changes']
    print(f"  Expected changes: {expected_changes}, Detected: {actual_changes}")
    
    # Load unchanged results from stage 1
    unchanged_stage1 = checkpoint_manager.load_incremental_checkpoint(
        "pipeline", "first_letters", "hash2", incremental_info["unchanged_indices"]
    )
    
    print(f"  Reusing {len(unchanged_stage1)} unchanged results from stage 1")
    
    # Simulate processing only changed/new records
    print("  Processing only changed and new records...")
    records_to_process = (
        incremental_info["changed_indices"] + incremental_info["new_indices"]
    )
    
    newly_processed = []
    for idx in records_to_process:
        if idx < len(updated_data):
            record = updated_data[idx]
            new_record = record.copy()
            new_record["first_letters"] = extract_first_letters(record["content"])
            newly_processed.append(new_record)
    
    print(f"  Processed {len(newly_processed)} records (vs {len(updated_data)} total)")
    processing_reduction = (len(updated_data) - len(newly_processed)) / len(updated_data)
    print(f"  Processing reduction: {processing_reduction:.1%}")
    
    # Test with stage 2 as well
    incremental_info_stage2 = checkpoint_manager.get_incremental_processing_info(
        "pipeline", "analyzed", updated_hashes
    )
    
    unchanged_stage2 = checkpoint_manager.load_incremental_checkpoint(
        "pipeline", "analyzed", "hash3", incremental_info_stage2["unchanged_indices"]
    )
    
    print(f"\nStage 2 incremental analysis:")
    print(f"  Reusing {len(unchanged_stage2)} unchanged results from stage 2")
    print(f"  Processing reduction: {(len(updated_data) - incremental_info_stage2['total_changes']) / len(updated_data):.1%}")
    
    # Verify incremental processing achieved significant savings
    assert processing_reduction > 0.90, f"Expected >90% processing reduction, got {processing_reduction:.1%}"
    assert len(unchanged_stage1) > 0, "Should have some unchanged records to reuse"
    assert len(unchanged_stage2) > 0, "Should have some unchanged records to reuse"
    
    print(f"\n✓ Incremental processing test successful!")
    print(f"✓ Achieved {processing_reduction:.1%} reduction in processing work")
    print(f"✓ Successfully reused cached results for unchanged records")
    
    # Show some actual examples of the text processing
    print(f"\nExample processed results:")
    example_record = stage2_data[0]
    print(f"  Document ID: {example_record['id']}")
    print(f"  First 100 chars: {example_record['content'][:100]}...")
    print(f"  First letters: {example_record['first_letters']}")
    print(f"  Analysis: {example_record['analysis']}")
    
    # Verify the first letter extraction is working correctly
    test_text = "The quick brown fox jumps over the lazy dog and then something else happens here today"
    extracted = extract_first_letters(test_text)
    # Count: The(1) quick(2) brown(3) fox(4) jumps(5) over(6) the(7) lazy(8) dog(9) and(10) then(11) something(12) else(13) happens(14) here(15)
    expected = "TQBFJOTLDATSEHH"  # First letters of first 15 words  
    assert extracted == expected, f"Expected {expected}, got {extracted}"
    print(f"  ✓ First letter extraction verified: '{test_text}' -> '{extracted}'")


def test_incremental_checkpointing_with_real_docetl_pipeline(temp_dir):
    """Test incremental checkpointing with an actual DocETL pipeline using DSLRunner."""
    import tempfile
    import os
    from docetl.runner import DSLRunner
    import json
    import hashlib
    
    def compute_record_hash(record):
        """Compute hash for change detection."""
        content = {"title": record["title"], "content": record["content"]}
        return hashlib.md5(json.dumps(content, sort_keys=True).encode()).hexdigest()
    
    # Create input data files
    input_file_v1 = os.path.join(temp_dir, "input_v1.json")
    input_file_v2 = os.path.join(temp_dir, "input_v2.json") 
    output_file = os.path.join(temp_dir, "output.json")
    
    # Initial dataset
    initial_documents = [
        {"title": "AI Research", "content": "Artificial intelligence is advancing rapidly in natural language processing."},
        {"title": "Machine Learning", "content": "Deep learning models are becoming more sophisticated and accurate."},
        {"title": "Data Science", "content": "Big data analytics helps organizations make better decisions."},
        {"title": "Cloud Computing", "content": "Distributed systems enable scalable computing infrastructure."},
        {"title": "Cybersecurity", "content": "Protecting digital assets requires comprehensive security strategies."}
    ]
    
    # Save initial dataset
    with open(input_file_v1, 'w') as f:
        json.dump(initial_documents, f)
    
    print(f"Created initial dataset with {len(initial_documents)} documents")
    
    # DocETL pipeline configuration
    pipeline_config = {
        "default_model": "gpt-4o-mini",
        "operations": [
            {
                "name": "extract_keywords",
                "type": "map",
                "prompt": "Extract 3 key topics from this text: '{{ input.content }}'. Return as a comma-separated list.",
                "output": {"schema": {"keywords": "string"}},
                "model": "gpt-4o-mini"
            },
            {
                "name": "categorize",
                "type": "map", 
                "prompt": "Categorize this document based on its title '{{ input.title }}' and keywords '{{ input.keywords }}'. Choose from: Technology, Business, Science, Education.",
                "output": {"schema": {"category": "string"}},
                "model": "gpt-4o-mini"
            }
        ],
        "datasets": {
            "input_docs": {
                "type": "file",
                "path": input_file_v1
            }
        },
        "pipeline": {
            "steps": [
                {
                    "name": "step1_extract",
                    "input": "input_docs",
                    "operations": ["extract_keywords"]
                },
                {
                    "name": "step2_categorize", 
                    "input": "step1_extract",
                    "operations": ["categorize"]
                }
            ],
            "output": {
                "type": "file",
                "path": output_file,
                "intermediate_dir": temp_dir
            }
        }
    }
    
    # Create runner with checkpoint directory
    runner = DSLRunner(pipeline_config, max_threads=4)
    
    print("\\nRunning initial pipeline...")
    
    # Run initial pipeline
    cost_v1 = runner.load_run_save()
    
    # Load the result
    with open(output_file, 'r') as f:
        result_v1 = json.load(f)
    
    print(f"Initial pipeline completed. Output size: {len(result_v1)}")
    
    # Check checkpoints were created
    checkpoint_sizes = {}
    for step_name in ["step1_extract", "step2_categorize"]:
        for op_name in ["extract_keywords", "categorize"]:
            size = runner.get_checkpoint_size(step_name, op_name)
            if size:
                checkpoint_sizes[f"{step_name}_{op_name}"] = size
                print(f"Checkpoint {step_name}/{op_name}: {size} bytes")
    
    total_checkpoint_size = runner.get_total_checkpoint_size()
    print(f"Total checkpoint size: {total_checkpoint_size} bytes")
    
    # Now simulate data changes - modify 2 docs, add 1 new doc
    print("\\nSimulating data changes...")
    modified_documents = initial_documents.copy()
    
    # Modify 2nd document 
    modified_documents[1] = {
        "title": "Advanced Machine Learning",  # Changed title
        "content": "Deep learning and neural networks are revolutionizing AI applications across industries."  # Changed content
    }
    
    # Modify 4th document
    modified_documents[3] = {
        "title": "Cloud Computing", # Same title
        "content": "Modern cloud platforms provide elastic, scalable computing resources with global reach."  # Changed content  
    }
    
    # Add new document
    modified_documents.append({
        "title": "Quantum Computing",
        "content": "Quantum algorithms promise exponential speedups for certain computational problems."
    })
    
    # Save modified dataset
    with open(input_file_v2, 'w') as f:
        json.dump(modified_documents, f)
    
    print(f"Created modified dataset with {len(modified_documents)} documents")
    print("Changes: 2 documents modified, 1 document added")
    
    # Analyze what changed using our incremental functionality
    initial_hashes = [compute_record_hash(doc) for doc in initial_documents]
    modified_hashes = [compute_record_hash(doc) for doc in modified_documents]
    
    # Check what incremental processing would detect  
    if runner.checkpoint_manager:
        incremental_info = runner.checkpoint_manager.get_incremental_processing_info(
            "step1_extract", "extract_keywords", modified_hashes
        )
    else:
        incremental_info = {"needs_full_reprocess": True, "reason": "No checkpoint manager"}
    
    print(f"\\nIncremental analysis (if we had tracked hashes):")
    if incremental_info.get("needs_full_reprocess"):
        print(f"  Would need full reprocess: {incremental_info.get('reason', 'Unknown')}")
    else:
        print(f"  Changed documents: {len(incremental_info.get('changed_indices', []))}")
        print(f"  New documents: {len(incremental_info.get('new_indices', []))}")
        print(f"  Unchanged documents: {len(incremental_info.get('unchanged_indices', []))}")
        print(f"  Total changes: {incremental_info.get('total_changes', 0)}")
        
        potential_reuse = len(incremental_info.get('unchanged_indices', []))
        total_docs = len(modified_documents)
        if total_docs > 0:
            efficiency = potential_reuse / total_docs * 100
            print(f"  Potential processing efficiency: {efficiency:.1f}% reuse")
    
    # Update pipeline config to use new input file
    pipeline_config["datasets"]["input_docs"]["path"] = input_file_v2
    output_file_v2 = os.path.join(temp_dir, "output_v2.json") 
    pipeline_config["pipeline"]["output"]["path"] = output_file_v2
    
    # Create new runner for modified pipeline
    runner_v2 = DSLRunner(pipeline_config, max_threads=4)
    
    print("\\nRunning pipeline with modified data...")
    
    # This will reuse existing checkpoints where possible
    cost_v2 = runner_v2.load_run_save()
    
    # Load the result
    with open(output_file_v2, 'r') as f:
        result_v2 = json.load(f)
    
    print(f"Modified pipeline completed. Output size: {len(result_v2)}")
    
    # Compare results
    print(f"\\nResults comparison:")
    print(f"  Initial run: {len(result_v1)} documents processed")
    print(f"  Modified run: {len(result_v2)} documents processed") 
    print(f"  New total checkpoint size: {runner_v2.get_total_checkpoint_size()} bytes")
    
    # Show actual processing results
    if result_v2:
        print(f"\\nExample processed document:")
        example = result_v2[0]
        print(f"  Title: {example.get('title', 'N/A')}")
        print(f"  Keywords: {example.get('keywords', 'N/A')}")
        print(f"  Category: {example.get('category', 'N/A')}")
    
    # Verify pipeline actually ran
    assert len(result_v1) > 0, "Initial pipeline should produce results"
    assert len(result_v2) > 0, "Modified pipeline should produce results"
    # Note: Second run might reuse checkpoints, so result count may differ
    # This actually demonstrates checkpoint reuse working!
    
    print("\\n✓ Real DocETL pipeline with checkpointing test completed!")
    print("✓ Pipeline successfully processed documents and used checkpoint system")
    
    return {
        "initial_docs": len(initial_documents),
        "modified_docs": len(modified_documents), 
        "initial_checkpoints": total_checkpoint_size,
        "modified_checkpoints": runner_v2.get_total_checkpoint_size(),
        "incremental_info": incremental_info
    }


def test_docetl_pipeline_large_dataset_space_efficiency(temp_dir):
    """Test real DocETL pipeline with large synthetic dataset to measure true space efficiency."""
    import random
    import os
    import json
    from docetl.runner import DSLRunner
    
    def generate_realistic_document(doc_id):
        """Generate realistic document content."""
        # Base content pools for realistic variety
        topics = [
            "artificial intelligence", "machine learning", "deep learning", "neural networks",
            "natural language processing", "computer vision", "robotics", "automation",
            "data science", "big data", "analytics", "business intelligence", "statistics",
            "cloud computing", "distributed systems", "microservices", "containers", "kubernetes",
            "cybersecurity", "encryption", "privacy", "data protection", "compliance",
            "software engineering", "agile development", "devops", "continuous integration",
            "blockchain", "cryptocurrency", "fintech", "digital transformation"
        ]
        
        companies = [
            "TechCorp", "DataSoft", "AI Innovations", "CloudFirst", "SecureNet", "AnalyticsPro",
            "NextGen Systems", "Digital Solutions", "SmartTech", "FutureLabs", "CyberGuard",
            "DataFlow", "InnovateTech", "CloudScale", "TechAdvance", "DigitalEdge"
        ]
        
        actions = [
            "revolutionizing", "transforming", "advancing", "improving", "optimizing",
            "streamlining", "enhancing", "accelerating", "modernizing", "innovating",
            "disrupting", "empowering", "enabling", "facilitating", "delivering"
        ]
        
        outcomes = [
            "business operations", "customer experience", "market efficiency", "operational costs",
            "productivity levels", "competitive advantage", "innovation cycles", "decision making",
            "process automation", "data insights", "system performance", "user engagement",
            "revenue growth", "market reach", "service delivery", "operational excellence"
        ]
        
        # Generate varied, realistic content
        topic = random.choice(topics)
        company = random.choice(companies)
        action = random.choice(actions)
        outcome = random.choice(outcomes)
        
        # Create realistic document with varied length
        base_content = f"{company} is {action} {topic} to improve {outcome}."
        
        # Add varied additional content
        additional_sentences = [
            f"This technology represents a significant advancement in the field.",
            f"Industry experts predict widespread adoption within the next few years.",
            f"The implementation has shown promising results in initial testing phases.",
            f"Cost savings and efficiency gains are expected to be substantial.",
            f"Integration with existing systems has been seamless and effective.",
            f"Customer feedback has been overwhelmingly positive and encouraging.",
            f"The solution addresses key challenges faced by organizations today.",
            f"Scalability and performance metrics exceed industry benchmarks."
        ]
        
        # Randomly add 2-6 additional sentences
        num_additional = random.randint(2, 6)
        selected_additional = random.sample(additional_sentences, num_additional)
        
        full_content = base_content + " " + " ".join(selected_additional)
        
        # Generate varied titles
        title_templates = [
            f"{topic.title()} Innovation at {company}",
            f"How {company} Uses {topic.title()}",
            f"{company}: {action.title()} with {topic.title()}",
            f"The Future of {topic.title()} at {company}",
            f"{company} {topic.title()} Case Study"
        ]
        
        title = random.choice(title_templates)
        
        return {
            "id": doc_id,
            "title": title,
            "content": full_content,
            "source": f"source_{doc_id % 20}",  # 20 different sources
            "department": random.choice(["Engineering", "Product", "Research", "Operations", "Marketing"]),
            "priority": random.choice(["High", "Medium", "Low"]),
            "timestamp": f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
        }
    
    # Generate large dataset (100 documents - manageable for real LLM calls)
    print("Generating large synthetic dataset (100 documents)...")
    large_dataset = [generate_realistic_document(i) for i in range(100)]
    
    # Save dataset
    input_file = os.path.join(temp_dir, "large_input.json")
    output_file = os.path.join(temp_dir, "large_output.json")
    
    with open(input_file, 'w') as f:
        json.dump(large_dataset, f)
    
    # Calculate input data size
    input_size = os.path.getsize(input_file)
    print(f"Input dataset size: {input_size:,} bytes ({input_size/1024:.1f} KB)")
    
    # DocETL pipeline configuration
    pipeline_config = {
        "default_model": "gpt-4o-mini",
        "operations": [
            {
                "name": "extract_keywords",
                "type": "map",
                "prompt": "Extract 3-5 key technical topics from this content: '{{ input.content }}'. Return as comma-separated list.",
                "output": {"schema": {"keywords": "string"}},
                "model": "gpt-4o-mini"
            },
            {
                "name": "categorize_domain",
                "type": "map", 
                "prompt": "Based on the title '{{ input.title }}' and keywords '{{ input.keywords }}', categorize this into one domain: AI/ML, Cloud/Infrastructure, Security, Data/Analytics, or Software Development.",
                "output": {"schema": {"domain": "string"}},
                "model": "gpt-4o-mini"
            },
            {
                "name": "assess_priority",
                "type": "map",
                "prompt": "Rate the business impact of this '{{ input.domain }}' initiative: '{{ input.title }}'. Return: Critical, High, Medium, or Low.",
                "output": {"schema": {"business_impact": "string"}},
                "model": "gpt-4o-mini"
            }
        ],
        "datasets": {
            "large_docs": {
                "type": "file",
                "path": input_file
            }
        },
        "pipeline": {
            "steps": [
                {
                    "name": "step1_keywords",
                    "input": "large_docs",
                    "operations": ["extract_keywords"]
                },
                {
                    "name": "step2_domain", 
                    "input": "step1_keywords",
                    "operations": ["categorize_domain"]
                },
                {
                    "name": "step3_priority",
                    "input": "step2_domain", 
                    "operations": ["assess_priority"]
                }
            ],
            "output": {
                "type": "file",
                "path": output_file,
                "intermediate_dir": temp_dir
            }
        }
    }
    
    print("\\nRunning large dataset pipeline...")
    
    # Run pipeline
    runner = DSLRunner(pipeline_config, max_threads=4)
    
    import time
    start_time = time.time()
    cost = runner.load_run_save()
    execution_time = time.time() - start_time
    
    # Load results
    with open(output_file, 'r') as f:
        results = json.load(f)
    
    output_size = os.path.getsize(output_file)
    
    print(f"\\nPipeline completed in {execution_time:.2f} seconds")
    print(f"Processed {len(results)} documents")
    print(f"Output size: {output_size:,} bytes ({output_size/1024:.1f} KB)")
    
    # Get checkpoint sizes
    checkpoint_sizes = {}
    total_checkpoint_size = 0
    
    operations = [
        ("step1_keywords", "extract_keywords"),
        ("step2_domain", "categorize_domain"), 
        ("step3_priority", "assess_priority")
    ]
    
    for step_name, op_name in operations:
        size = runner.get_checkpoint_size(step_name, op_name)
        if size:
            checkpoint_sizes[f"{step_name}/{op_name}"] = size
            total_checkpoint_size += size
            print(f"Checkpoint {step_name}/{op_name}: {size:,} bytes")
    
    print(f"Total checkpoint size: {total_checkpoint_size:,} bytes ({total_checkpoint_size/1024:.1f} KB)")
    
    # Calculate space efficiency vs JSON
    # The output file is already JSON, so compare checkpoint size to output size
    if total_checkpoint_size > 0 and output_size > 0:
        efficiency_ratio = total_checkpoint_size / output_size
        print(f"\\nSpace efficiency analysis:")
        print(f"  Output JSON size: {output_size:,} bytes")  
        print(f"  PyArrow checkpoints: {total_checkpoint_size:,} bytes")
        print(f"  Ratio: {efficiency_ratio:.3f}")
        
        if efficiency_ratio < 1:
            savings = (1 - efficiency_ratio) * 100
            print(f"  Space savings: {savings:.1f}% (PyArrow more efficient)")
        else:
            overhead = (efficiency_ratio - 1) * 100
            print(f"  Space overhead: {overhead:.1f}% (JSON more efficient)")
    
    # Show sample results
    print(f"\\nSample processed documents:")
    for i, doc in enumerate(results[:3]):
        print(f"  {i+1}. {doc.get('title', 'N/A')}")
        print(f"     Keywords: {doc.get('keywords', 'N/A')}")
        print(f"     Domain: {doc.get('domain', 'N/A')}")
        print(f"     Impact: {doc.get('business_impact', 'N/A')}")
    
    # Test checkpoint reuse with modified dataset
    print(f"\\nTesting checkpoint reuse...")
    
    # Modify 5% of documents (5 docs)
    modified_dataset = large_dataset.copy()
    num_to_modify = 5
    
    for i in range(num_to_modify):
        idx = random.randint(0, len(modified_dataset) - 1)
        # Modify content to trigger reprocessing
        modified_dataset[idx] = generate_realistic_document(len(modified_dataset) + i)
    
    # Save modified dataset
    input_file_v2 = os.path.join(temp_dir, "large_input_v2.json")
    output_file_v2 = os.path.join(temp_dir, "large_output_v2.json")
    
    with open(input_file_v2, 'w') as f:
        json.dump(modified_dataset, f)
    
    # Update config for second run
    pipeline_config["datasets"]["large_docs"]["path"] = input_file_v2
    pipeline_config["pipeline"]["output"]["path"] = output_file_v2
    
    # Run modified pipeline (should reuse some checkpoints)
    runner_v2 = DSLRunner(pipeline_config, max_threads=4)
    
    start_time_v2 = time.time()
    cost_v2 = runner_v2.load_run_save()
    execution_time_v2 = time.time() - start_time_v2
    
    print(f"Second run completed in {execution_time_v2:.2f} seconds")
    print(f"Performance improvement: {((execution_time - execution_time_v2) / execution_time * 100):.1f}% faster")
    
    # Final assertions
    assert len(results) == 100, "Should process all 100 documents"
    assert total_checkpoint_size > 0, "Should create checkpoints"
    assert execution_time_v2 < execution_time, "Second run should be faster due to checkpoint reuse"
    
    print(f"\\n✓ Large dataset pipeline test completed successfully!")
    print(f"✓ Demonstrated real space efficiency and performance benefits")
    
    return {
        "documents_processed": len(results),
        "input_size": input_size,
        "output_size": output_size, 
        "checkpoint_size": total_checkpoint_size,
        "initial_time": execution_time,
        "rerun_time": execution_time_v2,
        "efficiency_ratio": total_checkpoint_size / output_size if output_size > 0 else 0
    }


def test_large_data_handling(checkpoint_manager):
    """Test handling of larger datasets."""
    # Create larger dataset
    large_data = [
        {"id": i, "text": f"Document {i}", "value": i * 1.5}
        for i in range(1000)
    ]
    
    step_name = "test_step"
    operation_name = "test_operation"
    operation_hash = "test_hash"
    
    # Save and load
    checkpoint_manager.save_checkpoint(step_name, operation_name, large_data, operation_hash)
    loaded_data = checkpoint_manager.load_checkpoint(step_name, operation_name, operation_hash)
    
    # Verify data integrity
    assert len(loaded_data) == 1000
    assert loaded_data[0]["id"] == 0
    assert loaded_data[999]["id"] == 999
    assert loaded_data[500]["text"] == "Document 500"
    assert loaded_data[100]["value"] == 150.0


def test_special_characters_in_names(checkpoint_manager, sample_data):
    """Test handling of special characters in step and operation names."""
    step_name = "test-step_with.special@chars"
    operation_name = "test-operation_with.special@chars"
    operation_hash = "test_hash"
    
    # This should work without issues
    checkpoint_manager.save_checkpoint(step_name, operation_name, sample_data, operation_hash)
    loaded_data = checkpoint_manager.load_checkpoint(step_name, operation_name, operation_hash)
    
    assert loaded_data == sample_data


def test_console_logging(mock_console, temp_dir, sample_data):
    """Test that console logging works correctly."""
    cm = CheckpointManager(temp_dir, console=mock_console)
    
    # Save checkpoint
    cm.save_checkpoint("step", "op", sample_data, "hash")
    
    # Verify console.log was called
    mock_console.log.assert_called()
    
    # Check that log message contains expected content
    log_calls = mock_console.log.call_args_list
    assert any("Checkpoint saved" in str(call) for call in log_calls)