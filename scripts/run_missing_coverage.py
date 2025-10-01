#!/usr/bin/env python3
"""
Find and report on missing coverage in the codebase.

This script runs coverage analysis and reports on lines that aren't covered by tests,
suggesting which areas need additional test coverage.

Example usage:
    python scripts/run_missing_coverage.py                      # Default: analyze data analyzer modules
    python scripts/run_missing_coverage.py --all                # Analyze all modules
    python scripts/run_missing_coverage.py --modules module1,module2  # Analyze specific modules
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

# Find the project root (directory containing the script)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dictionary mapping module names to file paths
MODULE_PATHS = {
    "kmr.utils.data_analyzer": "kmr/utils/data_analyzer.py",
    "kmr.utils.data_analyzer_cli": "kmr/utils/data_analyzer_cli.py"
    # Add more modules here as needed
}

# Directory to store coverage data
htmlcov_dir = os.path.join(project_root, "htmlcov")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze test coverage and find missing coverage.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Analyze all modules in the project")
    group.add_argument("--modules", type=str, help="Comma-separated list of modules to analyze")
    
    return parser.parse_args()

def get_modules_to_analyze(args):
    """Determine which modules to analyze based on command line arguments."""
    if args.all:
        # In a real implementation, you would scan the project to find all modules
        print("Analyzing all modules in the project")
        return list(MODULE_PATHS.keys())
    elif args.modules:
        # Parse comma-separated modules
        modules = [m.strip() for m in args.modules.split(",")]
        valid_modules = [m for m in modules if m in MODULE_PATHS]
        if len(valid_modules) != len(modules):
            invalid = set(modules) - set(valid_modules)
            print(f"Warning: Unknown modules: {', '.join(invalid)}")
        return valid_modules
    else:
        # Default to analyzing data analyzer modules
        return ["kmr.utils.data_analyzer", "kmr.utils.data_analyzer_cli"]

def run_coverage(modules, test_dir="tests/"):
    """Run coverage analysis and export to JSON."""
    print("Running tests with coverage...")
    
    # Create a temporary .coveragerc file
    coveragerc_path = os.path.join(project_root, '.coveragerc')
    with open(coveragerc_path, 'w') as f:
        f.write(f"""
[run]
source = {','.join(modules)}
omit = */__pycache__/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
    if __name__ == .__main__.:
    pass
    raise ImportError
""")
    
    try:
        # Run coverage with pytest
        subprocess.run([
            'poetry', 'run', 'coverage', 'run',
            '--rcfile=' + coveragerc_path,
            '-m', 'pytest',
            test_dir
        ], check=True)
        
        # Generate JSON report
        json_path = os.path.join(htmlcov_dir, 'coverage.json')
        subprocess.run([
            'poetry', 'run', 'coverage', 'json',
            '--rcfile=' + coveragerc_path,
            '-o', json_path
        ], check=True)
        
        return json_path
    finally:
        # Clean up the temporary config file
        if os.path.exists(coveragerc_path):
            os.remove(coveragerc_path)

def analyze_coverage(json_output):
    """Analyze coverage report and find missing lines."""
    with open(json_output, 'r') as f:
        data = json.load(f)
    
    results = {}
    
    # Get modules that have missing lines
    for module_name, module_data in data["files"].items():
        if not module_name.startswith(tuple(MODULE_PATHS.keys())):
            continue
            
        missing_lines = module_data.get("missing_lines", [])
        if missing_lines:
            results[module_name] = missing_lines
    
    return results, data

def display_missing_coverage(file_path, missing_lines, data):
    """Display context around missing lines to help understand what's not covered."""
    if not missing_lines:
        return
    
    print(f"Module {file_path} has {len(missing_lines)} missing lines:")
    print(f"  Line numbers: {missing_lines}")
    print()
    
    # Get the source code
    file_data = data["files"].get(file_path, {})
    source = file_data.get("source_digest", "")
    
    # Group consecutive missing lines
    groups = []
    current_group = []
    for line in sorted(missing_lines):
        if not current_group or line == current_group[-1] + 1:
            current_group.append(line)
        else:
            groups.append(current_group)
            current_group = [line]
    if current_group:
        groups.append(current_group)
    
    # Display context around each group
    for group in groups:
        # Get a small window of context (2 lines before and after)
        start = max(1, group[0] - 2)
        end = min(group[-1] + 2, file_data.get("executed_lines", [])[-1] if file_data.get("executed_lines") else group[-1] + 2)
        
        print(f"Context around line {group[0]}:")
        
        # Read the actual file to get the source code
        try:
            actual_path = MODULE_PATHS.get(file_path, file_path)
            with open(os.path.join(project_root, actual_path), 'r') as f:
                lines = f.readlines()
                
            for i in range(start, end + 1):
                if i-1 < len(lines):
                    prefix = ">>> " if i in group else "    "
                    print(f"{prefix}{i}: {lines[i-1].rstrip()}")
        except Exception as e:
            print(f"Error displaying source: {e}")
        
        print()

def suggest_tests(missing_coverage):
    """Suggest tests to write for missing coverage."""
    print("Test Suggestions:")
    print()
    
    for module, missing_lines in missing_coverage.items():
        if not missing_lines:
            continue
        
        blocks = []
        current_block = []
        
        # Group consecutive lines
        for line in sorted(missing_lines):
            if not current_block or line == current_block[-1] + 1:
                current_block.append(line)
            else:
                blocks.append(current_block)
                current_block = [line]
        if current_block:
            blocks.append(current_block)
        
        print(f"Module: {module}")
        for block in blocks:
            if len(block) == 1:
                print(f"  - Line {block[0]} is not covered. Consider testing this case.")
            else:
                print(f"  - Lines {block[0]}-{block[-1]} are not covered. Consider testing this block.")
        print()

def main():
    """Main function to run coverage analysis and report missing areas."""
    print("=" * 80)
    print("Missing Coverage Analyzer")
    print("=" * 80)
    
    args = parse_args()
    modules = get_modules_to_analyze(args)
    
    if not modules:
        print("No modules specified for analysis. Exiting.")
        return
    
    # The directory to run tests from - default to all tests
    test_dir = "tests/"
    
    # For specific modules, we can target specific test directories
    if set(modules) == set(["kmr.utils.data_analyzer", "kmr.utils.data_analyzer_cli"]):
        test_dir = "tests/utils/"
    
    json_output = run_coverage(modules, test_dir)
    missing_coverage, coverage_data = analyze_coverage(json_output)
    
    # Create directory for coverage reports if it doesn't exist
    os.makedirs(htmlcov_dir, exist_ok=True)
    
    # Display missing lines with context
    for module, missing_lines in missing_coverage.items():
        display_missing_coverage(module, missing_lines, coverage_data)
    
    # Suggest tests to write
    suggest_tests(missing_coverage)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main() 