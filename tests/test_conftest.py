"""
Tests for the conftest.py configuration.
"""

import os
import sys
import unittest
from pathlib import Path
import importlib.util

class TestConftest(unittest.TestCase):
    """Test case for conftest.py configuration."""
    
    def test_path_setup(self):
        """Test that conftest.py adds the project root to sys.path."""
        # Save the original sys.path
        original_path = sys.path.copy()
        
        try:
            # Load the conftest module
            conftest_path = Path(__file__).parent / "conftest.py"
            self.assertTrue(conftest_path.exists(), "conftest.py not found")
            
            # Get the expected project root
            expected_root = str(Path(__file__).parent.parent)
            
            # Remove the project root from sys.path if it exists
            if expected_root in sys.path:
                sys.path.remove(expected_root)
            
            # Load the conftest module
            spec = importlib.util.spec_from_file_location("conftest", conftest_path)
            conftest = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(conftest)
            
            # Check that project root was added to sys.path
            self.assertIn(expected_root, sys.path, "Project root not added to sys.path")
            
        finally:
            # Restore the original sys.path
            sys.path = original_path

if __name__ == '__main__':
    unittest.main() 