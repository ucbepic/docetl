#!/usr/bin/env python3
"""
Checkpoint utilities for DocETL.

This script provides utilities for managing checkpoint storage, including:
- Migrating existing JSON checkpoints to Arrow format
- Comparing storage sizes between formats
- Analyzing checkpoint storage efficiency
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
from rich.console import Console  # type: ignore
from rich.table import Table  # type: ignore
from rich.panel import Panel  # type: ignore

from docetl.checkpoint_manager import CheckpointManager


def find_json_checkpoints(intermediate_dir: str) -> List[Tuple[str, str]]:
    """Find all JSON checkpoint files in the intermediate directory."""
    checkpoints = []
    intermediate_path = Path(intermediate_dir)
    
    if not intermediate_path.exists():
        return checkpoints
    
    # Find all .json files that match the checkpoint pattern
    for json_file in intermediate_path.rglob("*.json"):
        # Skip the config file
        if json_file.name == ".docetl_intermediate_config.json":
            continue
            
        # Skip batch files for now (they have a different pattern)
        if "_batches" in str(json_file.parent):
            continue
            
        # Extract step and operation names from path
        relative_path = json_file.relative_to(intermediate_path)
        if len(relative_path.parts) >= 2:
            step_name = relative_path.parts[0]
            operation_name = json_file.stem  # filename without extension
            checkpoints.append((step_name, operation_name))
    
    return checkpoints


def get_file_size(file_path: str) -> int:
    """Get file size in bytes."""
    try:
        return os.path.getsize(file_path)
    except OSError:
        return 0


def analyze_checkpoints(intermediate_dir: str) -> Dict[str, Any]:
    """Analyze all checkpoints in the directory and return size information."""
    console = Console()
    checkpoint_manager = CheckpointManager(intermediate_dir)
    
    json_checkpoints = find_json_checkpoints(intermediate_dir)
    analysis = {
        'total_json_size': 0,
        'total_arrow_size': 0,
        'checkpoints': [],
        'savings': 0,
        'savings_percent': 0
    }
    
    console.print(f"[blue]Analyzing checkpoints in {intermediate_dir}[/blue]")
    
    for step_name, operation_name in json_checkpoints:
        checkpoint_info = checkpoint_manager.get_checkpoint_info(step_name, operation_name)
        
        json_size = 0
        arrow_size = 0
        
        if checkpoint_info:
            if 'json' in checkpoint_info:
                json_size = checkpoint_info['json']['size_bytes']
            if 'arrow' in checkpoint_info:
                arrow_size = checkpoint_info['arrow']['size_bytes']
        
        analysis['total_json_size'] += json_size
        analysis['total_arrow_size'] += arrow_size
        
        savings = json_size - arrow_size if arrow_size > 0 else 0
        savings_percent = (savings / json_size * 100) if json_size > 0 else 0
        
        analysis['checkpoints'].append({
            'step_name': step_name,
            'operation_name': operation_name,
            'json_size': json_size,
            'arrow_size': arrow_size,
            'savings': savings,
            'savings_percent': savings_percent
        })
    
    if analysis['total_json_size'] > 0:
        analysis['savings'] = analysis['total_json_size'] - analysis['total_arrow_size']
        analysis['savings_percent'] = (analysis['savings'] / analysis['total_json_size'] * 100)
    
    return analysis


def format_bytes(bytes_size: int) -> str:
    """Format bytes into human readable string."""
    size_float = float(bytes_size)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_float < 1024.0:
            return f"{size_float:.1f} {unit}"
        size_float /= 1024.0
    return f"{size_float:.1f} TB"


def migrate_checkpoints(intermediate_dir: str, dry_run: bool = False) -> None:
    """Migrate JSON checkpoints to Arrow format."""
    console = Console()
    checkpoint_manager = CheckpointManager(intermediate_dir)
    
    json_checkpoints = find_json_checkpoints(intermediate_dir)
    
    if not json_checkpoints:
        console.print("[yellow]No JSON checkpoints found to migrate.[/yellow]")
        return
    
    console.print(f"[blue]Found {len(json_checkpoints)} JSON checkpoints to migrate[/blue]")
    
    if dry_run:
        console.print("[yellow]DRY RUN: No files will be modified[/yellow]")
    
    success_count = 0
    total_size_before = 0
    total_size_after = 0
    
    for step_name, operation_name in json_checkpoints:
        console.print(f"Processing {step_name}/{operation_name}...")
        
        # Get size before migration
        checkpoint_info = checkpoint_manager.get_checkpoint_info(step_name, operation_name)
        if checkpoint_info and 'json' in checkpoint_info:
            size_before = checkpoint_info['json']['size_bytes']
            total_size_before += size_before
        else:
            size_before = 0
        
        if not dry_run:
            success = checkpoint_manager.migrate_json_to_arrow(step_name, operation_name)
            if success:
                success_count += 1
                
                # Get size after migration
                updated_info = checkpoint_manager.get_checkpoint_info(step_name, operation_name)
                if updated_info and 'arrow' in updated_info:
                    size_after = updated_info['arrow']['size_bytes']
                    total_size_after += size_after
                    
                    savings = size_before - size_after
                    savings_percent = (savings / size_before * 100) if size_before > 0 else 0
                    
                    console.print(
                        f"  [green]✓[/green] Migrated: {format_bytes(size_before)} → "
                        f"{format_bytes(size_after)} "
                        f"({savings_percent:.1f}% savings)"
                    )
                else:
                    console.print(f"  [green]✓[/green] Migrated (size unknown)")
            else:
                console.print(f"  [red]✗[/red] Failed to migrate")
        else:
            console.print(f"  [blue]Would migrate {format_bytes(size_before)}[/blue]")
            success_count += 1
    
    if not dry_run:
        total_savings = total_size_before - total_size_after
        total_savings_percent = (total_savings / total_size_before * 100) if total_size_before > 0 else 0
        
        console.print(f"\n[green]Migration complete![/green]")
        console.print(f"Successfully migrated: {success_count}/{len(json_checkpoints)} checkpoints")
        console.print(f"Total size reduction: {format_bytes(total_savings)} ({total_savings_percent:.1f}%)")
    else:
        console.print(f"\n[blue]Dry run complete. Would migrate {success_count} checkpoints.[/blue]")


def display_analysis(analysis: Dict[str, Any]) -> None:
    """Display checkpoint analysis in a nice table format."""
    console = Console()
    
    # Create summary panel
    summary_text = f"""
Total JSON size: {format_bytes(analysis['total_json_size'])}
Total Arrow size: {format_bytes(analysis['total_arrow_size'])}
Total savings: {format_bytes(analysis['savings'])} ({analysis['savings_percent']:.1f}%)
"""
    
    console.print(Panel(summary_text.strip(), title="Storage Summary", expand=False))
    
    # Create detailed table
    if analysis['checkpoints']:
        table = Table(title="Checkpoint Details")
        table.add_column("Step", style="cyan")
        table.add_column("Operation", style="magenta")
        table.add_column("JSON Size", justify="right")
        table.add_column("Arrow Size", justify="right")
        table.add_column("Savings", justify="right", style="green")
        table.add_column("Savings %", justify="right", style="green")
        
        for checkpoint in analysis['checkpoints']:
            json_size_str = format_bytes(checkpoint['json_size']) if checkpoint['json_size'] else "N/A"
            arrow_size_str = format_bytes(checkpoint['arrow_size']) if checkpoint['arrow_size'] else "N/A"
            savings_str = format_bytes(checkpoint['savings']) if checkpoint['savings'] else "N/A"
            savings_pct_str = f"{checkpoint['savings_percent']:.1f}%" if checkpoint['savings_percent'] else "N/A"
            
            table.add_row(
                checkpoint['step_name'],
                checkpoint['operation_name'],
                json_size_str,
                arrow_size_str,
                savings_str,
                savings_pct_str
            )
        
        console.print(table)


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="DocETL Checkpoint Management Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze checkpoint storage efficiency
  python -m docetl.checkpoint_utils analyze /path/to/intermediate

  # Migrate JSON checkpoints to Arrow format
  python -m docetl.checkpoint_utils migrate /path/to/intermediate

  # Preview migration without making changes
  python -m docetl.checkpoint_utils migrate /path/to/intermediate --dry-run
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze checkpoint storage')
    analyze_parser.add_argument('intermediate_dir', help='Path to intermediate directory')
    
    # Migrate command
    migrate_parser = subparsers.add_parser('migrate', help='Migrate JSON to Arrow format')
    migrate_parser.add_argument('intermediate_dir', help='Path to intermediate directory')
    migrate_parser.add_argument('--dry-run', action='store_true', 
                               help='Preview migration without making changes')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if not os.path.exists(args.intermediate_dir):
        print(f"Error: Directory {args.intermediate_dir} does not exist")
        return
    
    if args.command == 'analyze':
        analysis = analyze_checkpoints(args.intermediate_dir)
        display_analysis(analysis)
    
    elif args.command == 'migrate':
        migrate_checkpoints(args.intermediate_dir, args.dry_run)


if __name__ == "__main__":
    main()