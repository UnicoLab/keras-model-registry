#!/usr/bin/env python
"""
Detailed coverage report generator.

This script runs a more detailed coverage analysis for specified modules 
including branch coverage and provides a comprehensive HTML report.

Example usage:
    python tests/utils/coverage_report.py                       # Default: analyze data analyzer
    python tests/utils/coverage_report.py --all                 # Analyze all modules
    python tests/utils/coverage_report.py --modules module1,module2  # Specific modules
    python tests/utils/coverage_report.py --output custom_dir   # Custom output directory
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

# Ensure parent directory is in path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Dictionary mapping module prefixes to test directories
MODULE_TEST_MAP = {
    "kmr.utils.data_analyzer": "tests/utils/",
    # Add more mappings as needed
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate detailed coverage reports.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Analyze all modules in the project")
    group.add_argument("--modules", type=str, help="Comma-separated list of modules to analyze")
    parser.add_argument("--output", type=str, help="Output directory for coverage reports")
    
    return parser.parse_args()

def get_modules_to_analyze(args):
    """Determine which modules to analyze based on command line arguments."""
    if args.all:
        # In a real implementation, you might scan the project to find all modules
        print("Analyzing all modules in the project")
        return list(MODULE_TEST_MAP.keys()) 
    elif args.modules:
        # Parse comma-separated modules
        return [m.strip() for m in args.modules.split(",")]
    else:
        # Default to analyzing data analyzer modules
        return ["kmr.utils.data_analyzer", "kmr.utils.data_analyzer_cli"]

def get_test_files(modules):
    """Determine which test files to use based on specified modules."""
    test_dirs = set()
    
    for module in modules:
        # Try to find a matching test directory
        found = False
        for prefix, test_dir in MODULE_TEST_MAP.items():
            if module.startswith(prefix):
                test_dirs.add(test_dir)
                found = True
                break
        
        # If no match, default to all tests
        if not found:
            test_dirs.add("tests/")
    
    # If we have specific directories, use them, otherwise run all tests
    if test_dirs:
        return list(test_dirs)
    else:
        return ["tests/"]

def check_dependencies():
    """Check if all dependencies are installed."""
    try:
        import coverage
        import pytest
    except ImportError as e:
        print(f"Error: Missing dependencies - {e}")
        print("Install with: poetry add coverage pytest pytest-cov -G dev")
        sys.exit(1)

def run_coverage(modules, test_paths, output_dir):
    """Run coverage analysis for the specified modules."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Remove existing coverage data
        subprocess.run(["coverage", "erase"], check=True)
        
        # Run tests with coverage
        cmd = ["coverage", "run", "--branch", "--source=" + ",".join(modules)]
        cmd.extend(["-m", "pytest"] + test_paths)
        subprocess.run(cmd, check=True)
        
        # Generate reports
        subprocess.run(["coverage", "report", "-m"], check=True)
        subprocess.run(["coverage", "html", "-d", output_dir], check=True)
        subprocess.run(["coverage", "json", "-o", os.path.join(output_dir, "coverage.json")], check=True)
        
        # Print summary
        print(f"\nCoverage report generated in: {output_dir}")
        print("Open index.html in a browser to view the detailed report.")
        
        # Calculate total coverage
        output = subprocess.check_output(["coverage", "report"], text=True)
        lines = output.strip().split('\n')
        if len(lines) > 1:
            total_line = lines[-1]
            if total_line.startswith("TOTAL"):
                coverage_percent = total_line.split()[-1]
                print(f"Total coverage: {coverage_percent}")
                
                # Check if coverage is below 90%
                try:
                    if float(coverage_percent.rstrip('%')) < 90:
                        print("Warning: Coverage is below 90%")
                except ValueError:
                    pass
                
    except subprocess.CalledProcessError as e:
        print(f"Error running coverage: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Check for dependencies
    check_dependencies()
    
    # Parse command line arguments
    args = parse_args()
    
    # Determine modules to analyze
    modules = get_modules_to_analyze(args)
    
    # Determine test files to run
    test_paths = get_test_files(modules)
    
    # Determine output directory
    if args.output:
        output_dir = os.path.join(project_root, args.output)
    elif len(modules) == 2 and "kmr.utils.data_analyzer" in modules and "kmr.utils.data_analyzer_cli" in modules:
        # Default for data analyzer
        output_dir = os.path.join(project_root, "htmlcov", "data_analyzer")
    else:
        # General output directory
        output_dir = os.path.join(project_root, "htmlcov")
    
    # Print banner
    print("=" * 80)
    print(f"Running detailed coverage analysis for: {', '.join(modules)}")
    print("=" * 80)
    
    # Run coverage
    run_coverage(modules, test_paths, output_dir)
    
    # Exit with success
    sys.exit(0) 