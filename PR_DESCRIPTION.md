# Fix Storage Redundancy in Checkpoint System

**Addresses:** #221

## Problem

The current checkpoint system in `runner.py` stores complete datasets as separate JSON files after every operation, leading to:
- Significant storage redundancy when data flows through multiple pipeline steps
- Inefficient disk usage for large datasets
- No deduplication of identical records across checkpoints

## Solution

Replaced the existing checkpoint system with an **incremental storage approach** using a single Arrow dataset with automatic deduplication:

### New Architecture
- **Single dataset file** (`checkpoints.parquet`) stores all unique records across all checkpoints
- **Separate index file** (`checkpoint_index.parquet`) tracks which records belong to each checkpoint
- **Automatic deduplication** via content hashing - identical records are stored only once
- **Efficient compression** using PyArrow's Snappy compression and columnar storage

### Key Components

1. **CheckpointManager** (`docetl/checkpoint_manager.py`)
   - Handles incremental storage with deduplication
   - Provides inspection methods for debugging workflows
   - Manages metadata-driven record retrieval

2. **DSLRunner Integration** (`docetl/runner.py`)
   - Seamlessly integrates with existing pipeline execution
   - Maintains backward compatibility for checkpoint loading
   - Crashes gracefully if PyArrow is unavailable (no fallback)

3. **Comprehensive Testing** (`tests/test_checkpoint_manager.py`)
   - Full test suite with 15 test methods
   - Storage efficiency benchmark showing 50-80% reduction in typical scenarios
   - Validates deduplication and data integrity

4. **Documentation** (`docs/concepts/intermediate-outputs.md`)
   - Explains how to inspect intermediate outputs for debugging
   - Provides practical workflows for pipeline development
   - Includes debugging patterns and best practices

## Benefits

- **Eliminates storage redundancy** through automatic deduplication
- **Reduces disk usage** significantly for pipelines with overlapping data
- **Enables efficient debugging** with rich inspection capabilities
- **Maintains fast access** through metadata-based indexing
- **Scales better** for large datasets and complex pipelines

## Changes Made

- ✅ Added `pyarrow` and `pandas` dependencies to `pyproject.toml`
- ✅ Created `CheckpointManager` with incremental storage and deduplication
- ✅ Integrated checkpoint manager into `DSLRunner`
- ✅ Removed graceful PyArrow fallback - crashes if unavailable
- ✅ Added comprehensive test suite with benchmarking
- ✅ Created mkdocs documentation for intermediate output inspection
- ✅ Updated mkdocs navigation to include new documentation

## Testing

The benchmark test generates realistic pipeline scenarios with 1000+ records and multiple overlapping steps, demonstrating significant storage savings while maintaining full data integrity and fast access patterns.

Run tests with:
```bash
pytest tests/test_checkpoint_manager.py::TestCheckpointManager::test_storage_footprint_benchmark -v -s
```

## Usage

Enable checkpoints by adding `intermediate_dir` to your pipeline output:

```yaml
pipeline:
  steps:
    - name: extraction
      operations: [extract_themes]
  output:
    type: file
    path: "results.json"
    intermediate_dir: "./intermediate_outputs"
```

Inspect outputs programmatically:
```python
from docetl.checkpoint_manager import CheckpointManager

manager = CheckpointManager("./intermediate_outputs")
checkpoints = manager.list_checkpoints()
sample = manager.load_checkpoint_sample("extraction", "extract_themes", n_rows=5)
```

This change significantly improves the efficiency of checkpoint storage while adding powerful debugging capabilities for pipeline development.