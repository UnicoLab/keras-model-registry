#!/usr/bin/env python
"""Test runner for kerasfactory.utils package tests.

This script runs all the unit tests for the kerasfactory.utils package
and generates a coverage report.
"""

import unittest
import sys
import os
from pathlib import Path
import coverage

# Add the project root to the path
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Start code coverage
cov = coverage.Coverage(
    source=["kerasfactory.utils"],
    omit=["*/__pycache__/*", "*/tests/*"],
)
cov.start()

# Find and run all tests in the utils directory
test_loader = unittest.TestLoader()
test_suite = test_loader.discover(os.path.dirname(__file__), pattern="test_*.py")

# Run the tests
test_runner = unittest.TextTestRunner(verbosity=2)
result = test_runner.run(test_suite)

# Stop coverage and generate report
cov.stop()
cov.save()
cov.report()

# Generate HTML report
cov.html_report(directory="htmlcov")

# Exit with appropriate code
sys.exit(not result.wasSuccessful())
