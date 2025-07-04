# Intermediate Outputs & Inspection

When running complex data processing pipelines, especially those involving LLM operations, it's helpful to inspect what's happening at each step. DocETL allows you to save and inspect intermediate outputs throughout your pipeline execution for debugging and validation purposes.

## Why Inspect Intermediate Outputs?

When developing pipelines, you often need to debug issues or validate that operations are working as expected. Intermediate outputs let you see exactly what data looks like after each step, making it much easier to identify problems and iterate on your pipeline without re-running expensive LLM operations.

## Enabling Checkpoints

To save intermediate outputs, specify an `intermediate_dir` when creating your pipeline:

```yaml
# config.yaml
dataset:
  type: file
  path: "data.json"

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
    path: "results.json"
    intermediate_dir: "./intermediate_outputs"
```

## Inspecting Intermediate Outputs

### Basic Inspection

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

# Get information about a specific checkpoint
info = manager.get_checkpoint_info("extraction", "extract_themes")
print(f"\nCheckpoint Info:")
print(f"  📊 Rows: {info['num_rows']:,}")
print(f"  📋 Columns: {info['num_columns']}")
print(f"  🏛️ Format: {info['format']}")

# Load and inspect sample data
sample = manager.load_checkpoint_sample("extraction", "extract_themes", n_rows=3)
print(f"\nSample Data:")
for i, record in enumerate(sample, 1):
    print(f"  Record {i}: {record}")
```

### DataFrame Analysis

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
if 'themes' in df.columns:
    sample_val = df['themes'].iloc[0]
    if isinstance(sample_val, list):
        print(f"✅ themes is correctly formatted as list")
    else:
        print(f"❌ themes is {type(sample_val)}, expected list")
```

### 2. **Content Quality Checks**
```python
# Check for empty or malformed outputs
df = manager.load_checkpoint_as_dataframe("extraction", "extract_themes")

# Find records with empty themes
empty_themes = df[df['themes'].apply(lambda x: not x or len(x) == 0)]
print(f"Records with empty themes: {len(empty_themes)}")

# Check theme quality
for i, themes in enumerate(df['themes'].head(5)):
    print(f"Record {i}: {len(themes)} themes extracted")
    if themes:
        avg_length = sum(len(theme) for theme in themes) / len(themes)
        print(f"  Average theme length: {avg_length:.1f} characters")
```

### 3. **Pipeline Flow Analysis**
```python
# Compare record counts across pipeline steps
checkpoints = manager.list_checkpoints()
for step_name, operation_name in checkpoints:
    info = manager.get_checkpoint_info(step_name, operation_name)
    print(f"{step_name}/{operation_name}: {info['num_rows']:,} records")

# This helps identify filtering or expansion operations
```

### 4. **Side-by-Side Comparison**
```python
# Compare inputs and outputs of a transformation
input_df = manager.load_checkpoint_as_dataframe("step1", "operation1")
output_df = manager.load_checkpoint_as_dataframe("step2", "operation2")

print(f"Input records: {len(input_df)}")
print(f"Output records: {len(output_df)}")

# Sample comparison
for i in range(min(3, len(input_df), len(output_df))):
    print(f"\nRecord {i}:")
    print(f"  Input: {input_df.iloc[i].to_dict()}")
    print(f"  Output: {output_df.iloc[i].to_dict()}")
```

## Complete Debugging Example

Here's a complete workflow for debugging a pipeline:

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
        sample = manager.load_checkpoint_sample(step_name, operation_name, n_rows=2)
        print(f"Sample records:")
        for i, record in enumerate(sample, 1):
            print(f"  Record {i}: {record}")
        
        # Load as DataFrame for analysis
        df = manager.load_checkpoint_as_dataframe(step_name, operation_name)
        
        # Basic statistics
        print(f"\nStatistics:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # Quality checks
        null_counts = df.isnull().sum()
        if null_counts.any():
            print(f"  ⚠️  Null values found: {null_counts[null_counts > 0].to_dict()}")
        else:
            print(f"  ✅ No null values")

# Usage
debug_pipeline("./intermediate_outputs")
```

## Best Practices

### 🔍 **Inspection Workflow**
- **Start small**: Test with a subset of data first
- **Sample early**: Use `load_checkpoint_sample()` for quick checks
- **Validate schemas**: Ensure each step produces the expected data structure
- **Check incrementally**: Review outputs after each step before proceeding

### 🐛 **Debugging Strategy**
1. **List all checkpoints** to see the pipeline flow
2. **Sample data** from each step to spot issues quickly
3. **Load full DataFrames** when you need deeper analysis
4. **Compare inputs/outputs** to understand transformations
5. **Validate expectations** against actual results

### 💾 **File Management**
- Use descriptive intermediate directories (e.g., `./debug_run_2024_01_15`)
- Clean up intermediate outputs from successful runs periodically
- Keep intermediate outputs for failed runs to debug issues

This inspection workflow helps you quickly identify issues, validate data quality, and understand your pipeline's behavior at each step, making it much easier to debug and iterate on complex data processing pipelines.