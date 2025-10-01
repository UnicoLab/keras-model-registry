#!/bin/bash
# Script to run all tests and generate coverage reports for the entire project

set -e  # Exit on error

# Make sure we're in the project root
cd "$(dirname "$0")"

echo "=== Installing dependencies ==="
poetry install

echo "=== Running All Tests ==="
make all_tests

echo "=== Generating Detailed Coverage Report ==="
make detailed_coverage

echo "=== Running Data Analyzer Tests and Coverage ==="
echo "This is a specialized coverage report for the data analyzer module"
make data_analyzer_all_tests
make data_analyzer_coverage_detailed 
make data_analyzer_missing_coverage

echo "=== All tests completed successfully! ==="
echo "Project-wide coverage report is available in htmlcov/index.html"
echo "Data analyzer specific coverage report is available in htmlcov/data_analyzer/index.html"
