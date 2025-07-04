# Intermediate Outputs & Checkpoint Storage

When running complex data processing pipelines, especially those involving LLM operations that can be expensive and time-consuming, it's crucial to have visibility into what's happening at each step. DocETL provides powerful checkpoint storage and inspection capabilities that allow you to save, analyze, and debug intermediate outputs throughout your pipeline execution.

## Why Save Intermediate Outputs?

### 🐛 **Debugging & Development**
- **Identify bottlenecks**: See exactly where your pipeline might be failing or producing unexpected results
- **Validate transformations**: Inspect data after each operation to ensure transformations are working correctly
- **Iterative development**: Modify downstream operations without re-running expensive upstream LLM calls

### 💰 **Cost Management**
- **Avoid re-computation**: Resume pipelines from checkpoints instead of starting from scratch
- **LLM cost control**: Don't lose expensive API calls due to downstream failures
- **Batch processing**: Process large datasets incrementally with checkpoint recovery

### 📊 **Data Quality Assurance**
- **Schema validation**: Verify that each step produces data in the expected format
- **Content inspection**: Sample and review actual outputs to ensure quality
- **Pipeline monitoring**: Track data flow and transformations across complex workflows

## How Checkpoint Storage Works

DocETL uses an efficient incremental storage system powered by PyArrow that:

- **Eliminates redundancy**: Automatically deduplicates records across checkpoints
- **Compresses efficiently**: Uses columnar storage with Snappy compression
- **Enables fast access**: Provides metadata-based indexing for quick retrieval
- **Scales well**: Handles large datasets without exponential storage growth

!!! info "Storage Efficiency"
    The checkpoint system typically achieves 50-80% storage reduction compared to naive separate file storage, thanks to automatic deduplication and compression.

## Enabling Checkpoints

To enable checkpoint storage, simply specify an `intermediate_dir` when creating your pipeline:

=== "YAML Configuration"

    ```yaml
    # config.yaml
    default:
      intermediate_dir: "./intermediate_outputs"
    
    dataset:
      type: file
      path: "data.jsonl"
    
    operations:
      - name: extract_themes
        type: map
        prompt: "Extract key themes from this text: {{ content }}"
        output:
          schema:
            themes: "list[str]"
      
      - name: categorize_themes  
        type: map
        prompt: "Categorize these themes: {{ themes }}"
        output:
          schema:
            category: "str"
            subcategory: "str"
    
    pipeline:
      steps:
        - name: extraction
          operations:
            - extract_themes
        - name: categorization
          operations:
            - categorize_themes
      output:
        type: file
        path: "results.jsonl"
    ```

=== "Python API"

    ```python
    from docetl import DSLRunner
    
    config = {
        "default": {
            "intermediate_dir": "./intermediate_outputs"
        },
        "dataset": {
            "type": "file", 
            "path": "data.jsonl"
        },
        "operations": [
            {
                "name": "extract_themes",
                "type": "map", 
                "prompt": "Extract key themes: {{ content }}",
                "output": {"schema": {"themes": "list[str]"}}
            }
        ],
        "pipeline": {
            "steps": [
                {
                    "name": "extraction",
                    "operations": ["extract_themes"]
                }
            ],
            "output": {"type": "file", "path": "results.jsonl"}
        }
    }
    
    runner = DSLRunner.from_yaml("config.yaml")
    results = runner.run()
    ```

## Inspecting Intermediate Outputs

### Command Line Inspection

Once you have checkpoints saved, you can inspect them programmatically:

```python
from docetl.checkpoint_manager import CheckpointManager

# Initialize checkpoint manager
manager = CheckpointManager("./intermediate_outputs")

# List all available checkpoints
checkpoints = manager.list_checkpoints()
print("Available checkpoints:")
for step_name, operation_name in checkpoints:
    print(f"  📁 {step_name}/{operation_name}")

# Get detailed information about a specific checkpoint
info = manager.get_checkpoint_info("extraction", "extract_themes")
print(f"\nCheckpoint Info:")
print(f"  📊 Rows: {info['num_rows']:,}")
print(f"  📋 Columns: {info['num_columns']}")
print(f"  💾 Size: {info['size_bytes'] / 1024 / 1024:.2f} MB")
print(f"  🏛️ Format: {info['format']}")

# Load and inspect sample data
sample = manager.load_checkpoint_sample("extraction", "extract_themes", n_rows=3)
print(f"\nSample Data:")
for i, record in enumerate(sample, 1):
    print(f"  Record {i}: {record}")
```

### Jupyter Notebook Analysis

For deeper analysis, load checkpoints as pandas DataFrames:

```python
import pandas as pd
from docetl.checkpoint_manager import CheckpointManager

manager = CheckpointManager("./intermediate_outputs")

# Load checkpoint as DataFrame for analysis
df = manager.load_checkpoint_as_dataframe("extraction", "extract_themes")

# Analyze the data
print("DataFrame Info:")
print(df.info())
print(f"\nShape: {df.shape}")
print(f"\nColumn types:\n{df.dtypes}")

# Sample analysis
print(f"\nFirst 5 rows:")
print(df.head())

# Analyze extracted themes
if 'themes' in df.columns:
    # Count theme frequencies
    all_themes = []
    for themes_list in df['themes']:
        if isinstance(themes_list, list):
            all_themes.extend(themes_list)
    
    theme_counts = pd.Series(all_themes).value_counts()
    print(f"\nTop 10 most common themes:")
    print(theme_counts.head(10))

# Check for potential issues
print(f"\nData Quality Checks:")
print(f"  Null values: {df.isnull().sum().sum()}")
print(f"  Empty theme lists: {sum(1 for x in df['themes'] if not x)}")
print(f"  Unique records: {df.drop_duplicates().shape[0]} / {df.shape[0]}")
```

### Storage Efficiency Analysis

Monitor your checkpoint storage efficiency:

```python
# Get overall storage statistics
summary = manager.get_checkpoint_summary()
print("📊 Checkpoint Summary:")
print(f"  Total checkpoints: {summary['total_checkpoints']}")
print(f"  Total records: {summary['total_rows']:,}")
print(f"  Storage size: {summary['total_size_bytes'] / 1024 / 1024:.2f} MB")

# Get detailed storage efficiency stats
stats = manager.get_storage_stats()
print(f"\n💾 Storage Efficiency:")
print(f"  Logical records (total): {stats['total_logical_records']:,}")
print(f"  Unique records (stored): {stats['total_unique_records']:,}")  
print(f"  Deduplication ratio: {stats['deduplication_ratio']:.3f}")
print(f"  Space saved: {stats['space_saved_ratio']:.1%}")
```

## Common Debugging Patterns

### 1. **Schema Validation**
```python
# Check if outputs match expected schema
df = manager.load_checkpoint_as_dataframe("extraction", "extract_themes")
expected_columns = ["content", "themes"]

missing_cols = set(expected_columns) - set(df.columns)
if missing_cols:
    print(f"❌ Missing columns: {missing_cols}")
else:
    print("✅ All expected columns present")

# Validate data types
for col in ["themes"]:
    if col in df.columns:
        sample_val = df[col].iloc[0]
        if isinstance(sample_val, list):
            print(f"✅ {col} is correctly formatted as list")
        else:
            print(f"❌ {col} is {type(sample_val)}, expected list")
```

### 2. **Content Quality Checks**
```python
# Check for empty or malformed outputs
df = manager.load_checkpoint_as_dataframe("extraction", "extract_themes")

# Find records with empty themes
empty_themes = df[df['themes'].apply(lambda x: not x or len(x) == 0)]
print(f"Records with empty themes: {len(empty_themes)}")

# Check theme quality
for i, themes in enumerate(df['themes'].head(10)):
    print(f"Record {i}: {len(themes)} themes extracted")
    if themes:
        avg_length = sum(len(theme) for theme in themes) / len(themes)
        print(f"  Average theme length: {avg_length:.1f} characters")
```

### 3. **Performance Analysis**
```python
# Compare record counts across pipeline steps
checkpoints = manager.list_checkpoints()
for step_name, operation_name in checkpoints:
    info = manager.get_checkpoint_info(step_name, operation_name)
    print(f"{step_name}/{operation_name}: {info['num_rows']:,} records")

# Identify bottlenecks by comparing record counts
```

## Best Practices

### 🔍 **Regular Inspection**
- **Sample early, sample often**: Use `load_checkpoint_sample()` for quick quality checks
- **Validate schemas**: Ensure each step produces the expected data structure
- **Monitor progress**: Check record counts to identify filtering or expansion operations

### 💾 **Storage Management**  
- **Use descriptive intermediate directories**: Organize by project or pipeline version
- **Clean up old checkpoints**: Remove intermediate outputs from successful runs periodically
- **Monitor storage usage**: Use `get_storage_stats()` to track efficiency

### 🐛 **Debugging Workflow**
1. **Start small**: Test with a subset of data first
2. **Checkpoint frequently**: Enable checkpoints for all major operations
3. **Inspect incrementally**: Review outputs after each step before proceeding
4. **Compare expectations**: Validate that outputs match your mental model

### 🚀 **Performance Optimization**
- **Leverage deduplication**: The checkpoint system automatically optimizes storage
- **Resume from checkpoints**: Use existing outputs to avoid re-running expensive operations
- **Batch processing**: Process large datasets in chunks with checkpoint recovery

## Example: Complete Debugging Session

Here's a complete example of debugging a pipeline that's producing unexpected results:

```python
from docetl.checkpoint_manager import CheckpointManager
import pandas as pd

def debug_pipeline(intermediate_dir: str):
    """Complete debugging session for a pipeline."""
    manager = CheckpointManager(intermediate_dir)
    
    print("🔍 PIPELINE DEBUGGING SESSION")
    print("=" * 50)
    
    # 1. Overview of all checkpoints
    checkpoints = manager.list_checkpoints()
    print(f"\n📁 Found {len(checkpoints)} checkpoints:")
    
    for step_name, operation_name in checkpoints:
        info = manager.get_checkpoint_info(step_name, operation_name)
        print(f"  {step_name}/{operation_name}: {info['num_rows']:,} records")
    
    # 2. Deep dive into each step
    for step_name, operation_name in checkpoints:
        print(f"\n🔬 Analyzing {step_name}/{operation_name}")
        print("-" * 40)
        
        # Load sample data
        sample = manager.load_checkpoint_sample(step_name, operation_name, n_rows=3)
        print(f"Sample records:")
        for i, record in enumerate(sample[:2], 1):
            print(f"  Record {i}: {record}")
        
        # Load as DataFrame for analysis
        df = manager.load_checkpoint_as_dataframe(step_name, operation_name)
        
        # Basic statistics
        print(f"\nStatistics:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # Quality checks
        null_counts = df.isnull().sum()
        if null_counts.any():
            print(f"  ⚠️  Null values found: {null_counts[null_counts > 0].to_dict()}")
        else:
            print(f"  ✅ No null values")
    
    # 3. Storage efficiency report
    stats = manager.get_storage_stats()
    print(f"\n💾 STORAGE EFFICIENCY")
    print("-" * 40)
    print(f"Total logical records: {stats['total_logical_records']:,}")
    print(f"Unique records stored: {stats['total_unique_records']:,}")
    print(f"Space saved: {stats['space_saved_ratio']:.1%}")
    print(f"Storage size: {stats['total_size_bytes'] / 1024 / 1024:.2f} MB")

# Usage
debug_pipeline("./intermediate_outputs")
```

This debugging approach helps you quickly identify issues, validate data quality, and understand your pipeline's behavior at each step. The checkpoint system makes it easy to iterate and improve your pipeline without losing expensive computational work.