"""
Unit tests for the kmr.utils.data_analyzer_cli module.
"""

import os
import sys
import json
import tempfile
import unittest
from unittest import mock
import io
import pandas as pd
import numpy as np

# Ensure parent directory is in path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from kmr.utils.data_analyzer_cli import parse_args, setup_logging, format_result, main


class TestDataAnalyzerCLI(unittest.TestCase):
    """Test case for the DataAnalyzer CLI module."""

    def setUp(self):
        """Set up test fixtures, if any."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create sample test data
        sample_data = pd.DataFrame({
            'num1': [1.0, 2.0, 3.0],
            'cat1': ['A', 'B', 'C']
        })
        
        # Create a sample CSV file
        self.csv_path = os.path.join(self.temp_dir.name, 'sample.csv')
        sample_data.to_csv(self.csv_path, index=False)

    def tearDown(self):
        """Tear down test fixtures, if any."""
        self.temp_dir.cleanup()

    def test_parse_args(self):
        """Test argument parsing."""
        # Test with minimum required arguments
        with mock.patch('sys.argv', ['data_analyzer_cli.py', self.csv_path]):
            args = parse_args()
            self.assertEqual(args.source, self.csv_path)
            self.assertEqual(args.pattern, '*.csv')
            self.assertIsNone(args.output)
            self.assertFalse(args.verbose)
            self.assertFalse(args.recommendations_only)
        
        # Test with all arguments
        with mock.patch('sys.argv', [
            'data_analyzer_cli.py',
            self.csv_path,
            '--pattern', '*.data',
            '--output', 'output.json',
            '--verbose',
            '--recommendations-only'
        ]):
            args = parse_args()
            self.assertEqual(args.source, self.csv_path)
            self.assertEqual(args.pattern, '*.data')
            self.assertEqual(args.output, 'output.json')
            self.assertTrue(args.verbose)
            self.assertTrue(args.recommendations_only)

    @mock.patch('kmr.utils.data_analyzer_cli.logger')
    def test_setup_logging(self, mock_logger):
        """Test logging setup."""
        # Test with verbose=False
        setup_logging(False)
        mock_logger.remove.assert_called_once()
        mock_logger.add.assert_called_once()
        # Check that the level parameter is INFO (without relying on specific positional arguments)
        args, kwargs = mock_logger.add.call_args
        self.assertEqual(kwargs.get('level', None) or args[1], "INFO")
        
        # Reset mock and test with verbose=True
        mock_logger.reset_mock()
        setup_logging(True)
        mock_logger.remove.assert_called_once()
        mock_logger.add.assert_called_once()
        # Check that the level parameter is DEBUG (without relying on specific positional arguments)
        args, kwargs = mock_logger.add.call_args
        self.assertEqual(kwargs.get('level', None) or args[1], "DEBUG")

    def test_format_result(self):
        """Test result formatting."""
        # Create a sample result
        result = {
            'analysis': {'key': 'value'},
            'recommendations': {'rec_key': 'rec_value'}
        }
        
        # Test with recommendations_only=False
        formatted = format_result(result, False)
        self.assertEqual(formatted, result)
        
        # Test with recommendations_only=True
        formatted = format_result(result, True)
        self.assertEqual(formatted, {'recommendations': {'rec_key': 'rec_value'}})
        
        # Test with empty recommendations
        result = {'analysis': {'key': 'value'}}
        formatted = format_result(result, True)
        self.assertEqual(formatted, {'recommendations': {}})

    @mock.patch('kmr.utils.data_analyzer_cli.DataAnalyzer')
    @mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_main_stdout(self, mock_stdout, mock_analyzer_class):
        """Test main function with output to stdout."""
        # Mock the analyzer instance
        mock_analyzer = mock.MagicMock()
        mock_analyzer_class.return_value = mock_analyzer
        
        # Mock the analyze_and_recommend method
        mock_analyzer.analyze_and_recommend.return_value = {
            'analysis': {'file': 'sample.csv', 'stats': {'row_count': 3}},
            'recommendations': {'continuous_features': [('Layer', 'Desc', 'Use')]}
        }
        
        # Run main with minimal arguments
        with mock.patch('sys.argv', ['data_analyzer_cli.py', self.csv_path]):
            main()
            
        # Check that analyze_and_recommend was called
        mock_analyzer.analyze_and_recommend.assert_called_once_with(self.csv_path, '*.csv')
        
        # Check that output was printed to stdout
        output = mock_stdout.getvalue()
        self.assertIn('analysis', output)
        self.assertIn('recommendations', output)

    @mock.patch('kmr.utils.data_analyzer_cli.DataAnalyzer')
    def test_main_file_output(self, mock_analyzer_class):
        """Test main function with output to file."""
        # Mock the analyzer instance
        mock_analyzer = mock.MagicMock()
        mock_analyzer_class.return_value = mock_analyzer
        
        # Mock the analyze_and_recommend method
        mock_analyzer.analyze_and_recommend.return_value = {
            'analysis': {'file': 'sample.csv', 'stats': {'row_count': 3}},
            'recommendations': {'continuous_features': [('Layer', 'Desc', 'Use')]}
        }
        
        # Create a temporary output file
        output_path = os.path.join(self.temp_dir.name, 'output.json')
        
        # Run main with output file
        with mock.patch('sys.argv', [
            'data_analyzer_cli.py',
            self.csv_path,
            '--output', output_path
        ]):
            main()
            
        # Check that output file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Check file contents
        with open(output_path, 'r') as f:
            output_data = json.load(f)
            self.assertIn('analysis', output_data)
            self.assertIn('recommendations', output_data)

    @mock.patch('kmr.utils.data_analyzer_cli.DataAnalyzer')
    def test_main_recommendations_only(self, mock_analyzer_class):
        """Test main function with recommendations-only flag."""
        # Mock the analyzer instance
        mock_analyzer = mock.MagicMock()
        mock_analyzer_class.return_value = mock_analyzer
        
        # Mock the analyze_and_recommend method
        mock_analyzer.analyze_and_recommend.return_value = {
            'analysis': {'file': 'sample.csv', 'stats': {'row_count': 3}},
            'recommendations': {'continuous_features': [('Layer', 'Desc', 'Use')]}
        }
        
        # Create a temporary output file
        output_path = os.path.join(self.temp_dir.name, 'output.json')
        
        # Run main with recommendations-only flag
        with mock.patch('sys.argv', [
            'data_analyzer_cli.py',
            self.csv_path,
            '--output', output_path,
            '--recommendations-only'
        ]):
            main()
            
        # Check file contents
        with open(output_path, 'r') as f:
            output_data = json.load(f)
            self.assertNotIn('analysis', output_data)
            self.assertIn('recommendations', output_data)

    @mock.patch('kmr.utils.data_analyzer_cli.logger')
    @mock.patch('sys.exit')
    def test_main_nonexistent_file(self, mock_exit, mock_logger):
        """Test main function with a nonexistent file."""
        # Run main with a nonexistent file
        with mock.patch('sys.argv', ['data_analyzer_cli.py', 'nonexistent.csv']):
            main()
            
        # Check that error was logged and sys.exit was called
        mock_logger.error.assert_called_once()
        mock_exit.assert_called_once_with(1)

    @mock.patch('kmr.utils.data_analyzer_cli.DataAnalyzer')
    @mock.patch('kmr.utils.data_analyzer_cli.logger')
    @mock.patch('sys.exit')
    def test_main_exception(self, mock_exit, mock_logger, mock_analyzer_class):
        """Test main function when an exception occurs."""
        # Mock the analyzer instance to raise an exception
        mock_analyzer = mock.MagicMock()
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.analyze_and_recommend.side_effect = Exception("Test error")
        
        # Run main
        with mock.patch('sys.argv', ['data_analyzer_cli.py', self.csv_path]):
            main()
            
        # Check that error was logged and sys.exit was called
        mock_logger.error.assert_called_once()
        mock_exit.assert_called_once_with(1)

    def test_parse_args_help(self):
        """Test parsing of the help flag."""
        # Test with --help argument
        with mock.patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            with self.assertRaises(SystemExit) as cm:
                with mock.patch('sys.argv', ['data_analyzer_cli.py', '--help']):
                    args = parse_args()
                self.assertEqual(cm.exception.code, 0)
                
                # Check that help text was printed
                output = mock_stdout.getvalue()
                self.assertIn('Analyze CSV data and recommend KMR layers', output)


if __name__ == '__main__':
    unittest.main() 