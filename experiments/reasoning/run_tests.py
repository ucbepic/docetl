#!/usr/bin/env python3
"""
Directive Testing Runner

This script provides a convenient way to run directive tests from the experiments folder.
It links to the main test runner in the tests directory.
"""

import argparse
import sys
import subprocess
from pathlib import Path

from docetl.reasoning_optimizer.directives import DEFAULT_MODEL

def main():
    """Run directive tests by calling the test runner directly."""
    parser = argparse.ArgumentParser(description="Run directive tests")
    parser.add_argument("--directive", "-d", type=str, help="Specific directive to test")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL, help=f"LLM model to use (default: {DEFAULT_MODEL})")
    
    args = parser.parse_args()
    
    # Path to the actual test runner
    test_runner_path = Path(__file__).parent.parent.parent / "tests" / "reasoning_optimizer" / "test_runner.py"
    
    # Build command to run the test runner
    cmd = [sys.executable, str(test_runner_path)]
    if args.directive:
        cmd.extend(["--directive", args.directive])
    if args.model != DEFAULT_MODEL:
        cmd.extend(["--model", args.model])
    
    # Run the test runner
    result = subprocess.run(cmd, cwd=Path.cwd())
    return result.returncode

if __name__ == "__main__":
    exit(main())