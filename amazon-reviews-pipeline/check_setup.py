#!/usr/bin/env python3
"""
Setup verification script for Amazon Reviews Pipeline
Checks that all requirements are met before running the pipeline
"""

import os
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def check_api_key():
    """Check if OpenAI API key is configured"""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        return False, "Not set"
    elif api_key == "your-openai-api-key-here" or api_key == "your-api-key-here":
        return False, "Using placeholder value"
    elif api_key.startswith("sk-"):
        # Mask the key for security
        masked = f"{api_key[:7]}...{api_key[-4:]}"
        return True, f"Set ({masked})"
    else:
        return False, "Invalid format"

def check_dependencies():
    """Check if required Python packages are installed"""
    required = {
        "openai": "OpenAI API client",
        "rich": "Terminal formatting",
        "dotenv": "Environment file support",
        "pandas": "Data processing",
        "numpy": "Numerical operations"
    }
    
    missing = []
    for package, description in required.items():
        try:
            __import__(package)
        except ImportError:
            missing.append((package, description))
    
    return missing

def check_files():
    """Check if required files exist"""
    files = {
        "data/amazon_reviews.json": "Review data",
        "collect_reviews.py": "Data generation script",
        "run_analysis.py": "Main pipeline script",
        "visualize_results.py": "Results visualization"
    }
    
    missing = []
    for file, description in files.items():
        if not os.path.exists(file):
            missing.append((file, description))
    
    return missing

def main():
    console.print(Panel.fit(
        "[bold cyan]Amazon Reviews Pipeline Setup Check[/bold cyan]",
        border_style="cyan"
    ))
    
    # Create status table
    table = Table(title="Environment Status", show_header=True)
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details")
    
    # Check API key
    api_ok, api_details = check_api_key()
    table.add_row(
        "OpenAI API Key",
        "✅ OK" if api_ok else "❌ Failed",
        api_details
    )
    
    # Check dependencies
    missing_deps = check_dependencies()
    deps_ok = len(missing_deps) == 0
    table.add_row(
        "Python Dependencies",
        "✅ OK" if deps_ok else "❌ Missing",
        f"{len(missing_deps)} missing" if missing_deps else "All installed"
    )
    
    # Check files
    missing_files = check_files()
    files_ok = len(missing_files) == 0
    table.add_row(
        "Required Files",
        "✅ OK" if files_ok else "❌ Missing",
        f"{len(missing_files)} missing" if missing_files else "All present"
    )
    
    # Check output directory
    output_exists = os.path.exists("output")
    table.add_row(
        "Output Directory",
        "✅ OK" if output_exists else "⚠️  Missing",
        "Will be created" if not output_exists else "Ready"
    )
    
    console.print("\n")
    console.print(table)
    console.print("\n")
    
    # Show detailed issues if any
    if not api_ok:
        console.print("[red]❌ API Key Issue:[/red]")
        console.print("  Set your OpenAI API key using:")
        console.print("  [cyan]export OPENAI_API_KEY='your-actual-key-here'[/cyan]")
        console.print("  Or create a .env file from .env.example\n")
    
    if missing_deps:
        console.print("[red]❌ Missing Dependencies:[/red]")
        for pkg, desc in missing_deps:
            console.print(f"  - {pkg} ({desc})")
        console.print("  Install with: [cyan]pip3 install -r requirements.txt[/cyan]\n")
    
    if missing_files:
        console.print("[red]❌ Missing Files:[/red]")
        for file, desc in missing_files:
            console.print(f"  - {file} ({desc})")
        if "data/amazon_reviews.json" in [f[0] for f in missing_files]:
            console.print("  Generate data with: [cyan]python3 collect_reviews.py[/cyan]\n")
    
    # Overall status
    all_ok = api_ok and deps_ok and files_ok
    
    if all_ok:
        console.print(Panel(
            "[bold green]✅ All checks passed![/bold green]\n"
            "You're ready to run the pipeline with:\n"
            "[cyan]python3 run_analysis.py[/cyan]",
            border_style="green"
        ))
    else:
        console.print(Panel(
            "[bold red]❌ Setup incomplete[/bold red]\n"
            "Please fix the issues above before running the pipeline.",
            border_style="red"
        ))
        sys.exit(1)

if __name__ == "__main__":
    main()