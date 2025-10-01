#!/bin/bash
# Script to set up and run data analyzer tests

set -e  # Exit on error

# Make sure we're in the project root
cd "$(dirname "$0")"

echo "=== Installing dependencies ==="
poetry install

echo "=== Running Unit and Integration Tests ==="
make data_analyzer_all_tests

echo "=== Generating Detailed Coverage Report ==="
make data_analyzer_coverage_detailed

echo "=== Finding Missing Coverage ==="
make data_analyzer_missing_coverage

echo "=== All tests completed successfully! ==="
echo "Detailed coverage report is available in htmlcov/data_analyzer/index.html" 