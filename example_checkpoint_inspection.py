#!/usr/bin/env python3
"""
Example: Inspecting DocETL Checkpoint Outputs

This script demonstrates how to use the CheckpointManager to inspect
intermediate outputs from DocETL pipeline runs. This is useful for:

- Debugging pipeline steps
- Analyzing intermediate data transformations  
- Creating reports or notebooks based on pipeline outputs
- Understanding data flow through the pipeline

Usage:
    python example_checkpoint_inspection.py /path/to/intermediate/dir
"""

import sys
import os
from typing import Optional

try:
    from docetl.checkpoint_manager import CheckpointManager
    import pandas as pd  # type: ignore
    CHECKPOINT_AVAILABLE = True
except ImportError as e:
    print(f"Error: Required packages not available: {e}")
    print("Install with: pip install pyarrow pandas")
    CHECKPOINT_AVAILABLE = False
    sys.exit(1)


def inspect_pipeline_outputs(intermediate_dir: str) -> None:
    """
    Inspect all outputs from a DocETL pipeline run.
    
    Args:
        intermediate_dir: Path to the intermediate directory from a pipeline run
    """
    if not os.path.exists(intermediate_dir):
        print(f"Error: Directory {intermediate_dir} does not exist")
        return
    
    try:
        manager = CheckpointManager(intermediate_dir)
    except RuntimeError as e:
        print(f"Error initializing checkpoint manager: {e}")
        return
    
    print("🔍 DocETL Pipeline Output Inspector")
    print("=" * 50)
    
    # Get overall summary
    summary = manager.get_checkpoint_summary()
    print(f"\n📊 Summary:")
    print(f"   Total checkpoints: {summary['total_checkpoints']}")
    print(f"   Total rows: {summary['total_rows']:,}")
    print(f"   Total storage: {format_bytes(summary['total_size_bytes'])}")
    
    # List all available checkpoints
    checkpoints = manager.list_checkpoints()
    if not checkpoints:
        print("\n❌ No checkpoints found in the directory")
        return
    
    print(f"\n📁 Available Outputs:")
    for step_name, operation_name in checkpoints:
        info = manager.get_checkpoint_info(step_name, operation_name)
        if info:
            rows = info.get('num_rows', 0)
            size = info.get('size_bytes', 0)
            print(f"   📄 {step_name}/{operation_name}: {rows:,} rows, {format_bytes(size)}")
    
    # Interactive inspection
    print(f"\n🔎 Interactive Inspection")
    print("Available commands:")
    print("  'list' - List all checkpoints")
    print("  'info <step>/<op>' - Get detailed info about a checkpoint")
    print("  'sample <step>/<op>' - Show sample data")
    print("  'df <step>/<op>' - Load as DataFrame (for analysis)")
    print("  'json <step>/<op>' - Export as JSON")
    print("  'quit' - Exit")
    
    while True:
        try:
            cmd = input("\n> ").strip()
            
            if cmd == 'quit':
                break
            elif cmd == 'list':
                print_checkpoint_list(checkpoints, manager)
            elif cmd.startswith('info '):
                handle_info_command(cmd, manager)
            elif cmd.startswith('sample '):
                handle_sample_command(cmd, manager)
            elif cmd.startswith('df '):
                handle_dataframe_command(cmd, manager)
            elif cmd.startswith('json '):
                handle_json_command(cmd, manager)
            else:
                print("❌ Unknown command. Type 'quit' to exit.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def print_checkpoint_list(checkpoints, manager):
    """Print detailed list of all checkpoints."""
    print("\n📋 Detailed Checkpoint List:")
    print("-" * 60)
    print(f"{'Step/Operation':<30} {'Rows':<10} {'Size':<12} {'Columns'}")
    print("-" * 60)
    
    for step_name, operation_name in checkpoints:
        info = manager.get_checkpoint_info(step_name, operation_name)
        if info:
            name = f"{step_name}/{operation_name}"
            rows = info.get('num_rows', 0)
            size = format_bytes(info.get('size_bytes', 0))
            cols = info.get('num_columns', 0)
            print(f"{name:<30} {rows:<10,} {size:<12} {cols}")


def handle_info_command(cmd: str, manager: CheckpointManager):
    """Handle 'info <step>/<op>' command."""
    try:
        path = cmd.split(' ', 1)[1]
        step_name, operation_name = path.split('/')
        
        info = manager.get_checkpoint_info(step_name, operation_name)
        if not info:
            print(f"❌ Checkpoint {path} not found")
            return
        
        print(f"\n📄 Checkpoint Info: {path}")
        print("-" * 40)
        print(f"Rows: {info.get('num_rows', 0):,}")
        print(f"Columns: {info.get('num_columns', 0)}")
        print(f"Size: {format_bytes(info.get('size_bytes', 0))}")
        print(f"Format: {info.get('format', 'unknown')}")
        
        columns = info.get('column_names', [])
        if columns:
            print(f"Column names: {', '.join(columns[:10])}")
            if len(columns) > 10:
                print(f"  ... and {len(columns) - 10} more")
        
    except (IndexError, ValueError):
        print("❌ Usage: info <step>/<operation>")
    except Exception as e:
        print(f"❌ Error: {e}")


def handle_sample_command(cmd: str, manager: CheckpointManager):
    """Handle 'sample <step>/<op>' command."""
    try:
        path = cmd.split(' ', 1)[1]
        step_name, operation_name = path.split('/')
        
        sample = manager.load_checkpoint_sample(step_name, operation_name, n_rows=5)
        if not sample:
            print(f"❌ Checkpoint {path} not found")
            return
        
        print(f"\n📋 Sample Data: {path} (first 5 rows)")
        print("-" * 60)
        
        for i, record in enumerate(sample):
            print(f"Row {i+1}:")
            for key, value in record.items():
                # Truncate long values
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:97] + "..."
                print(f"  {key}: {value_str}")
            print()
        
    except (IndexError, ValueError):
        print("❌ Usage: sample <step>/<operation>")
    except Exception as e:
        print(f"❌ Error: {e}")


def handle_dataframe_command(cmd: str, manager: CheckpointManager):
    """Handle 'df <step>/<op>' command."""
    try:
        path = cmd.split(' ', 1)[1]
        step_name, operation_name = path.split('/')
        
        df = manager.load_checkpoint_as_dataframe(step_name, operation_name)
        if df is None:
            print(f"❌ Checkpoint {path} not found")
            return
        
        print(f"\n📊 DataFrame: {path}")
        print("-" * 40)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst 5 rows:")
        print(df.head())
        
        print(f"\n💡 The DataFrame is available as variable 'df' if you want to analyze it further")
        print("   You can copy this data to a Jupyter notebook for deeper analysis")
        
        # Make df available for further analysis (in interactive mode)
        globals()['df'] = df
        
    except (IndexError, ValueError):
        print("❌ Usage: df <step>/<operation>")
    except Exception as e:
        print(f"❌ Error: {e}")


def handle_json_command(cmd: str, manager: CheckpointManager):
    """Handle 'json <step>/<op>' command."""
    try:
        parts = cmd.split(' ', 1)[1].split()
        path = parts[0]
        filename = parts[1] if len(parts) > 1 else f"{path.replace('/', '_')}.json"
        
        step_name, operation_name = path.split('/')
        
        data = manager.load_checkpoint(step_name, operation_name)
        if not data:
            print(f"❌ Checkpoint {path} not found")
            return
        
        import json
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Exported {len(data)} records to {filename}")
        
    except (IndexError, ValueError):
        print("❌ Usage: json <step>/<operation> [filename]")
    except Exception as e:
        print(f"❌ Error: {e}")


def format_bytes(bytes_size: int) -> str:
    """Format bytes into human readable string."""
    size_float = float(bytes_size)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_float < 1024.0:
            return f"{size_float:.1f}{unit}"
        size_float /= 1024.0
    return f"{size_float:.1f}TB"


def main():
    """Main CLI interface."""
    if len(sys.argv) != 2:
        print("Usage: python example_checkpoint_inspection.py <intermediate_directory>")
        print("\nExample:")
        print("  python example_checkpoint_inspection.py ./intermediate")
        sys.exit(1)
    
    intermediate_dir = sys.argv[1]
    inspect_pipeline_outputs(intermediate_dir)


if __name__ == "__main__":
    main()


# Example for Jupyter Notebook usage:
"""
# In a Jupyter notebook, you can use the checkpoint manager like this:

from docetl.checkpoint_manager import CheckpointManager
import pandas as pd

# Initialize checkpoint manager
manager = CheckpointManager("path/to/intermediate")

# List all available outputs
checkpoints = manager.list_checkpoints()
print("Available outputs:", checkpoints)

# Load specific output as DataFrame for analysis
df = manager.load_checkpoint_as_dataframe("extract_step", "extract_entities")
print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")

# Analyze the data
df.describe()
df.head()

# Or load as JSON for further processing
data = manager.load_checkpoint("extract_step", "extract_entities")
print(f"First record: {data[0]}")

# Get summary of all outputs
summary = manager.get_checkpoint_summary()
print(f"Total checkpoints: {summary['total_checkpoints']}")
print(f"Total data: {summary['total_rows']} rows, {summary['total_size_bytes']} bytes")
"""