"""
Unit tests for the kmr.utils.data_analyzer module.
"""

import os
import sys
import json
import tempfile
import unittest
from unittest import mock
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

# Ensure parent directory is in path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from kmr.utils.data_analyzer import DataAnalyzer, analyze_data


class TestDataAnalyzer(unittest.TestCase):
    """Test case for the DataAnalyzer class."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.analyzer = DataAnalyzer()
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create sample test data
        self.sample_data = pd.DataFrame({
            # Numerical features
            'num1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'num2': [10.5, 20.5, 30.5, 40.5, 50.5],
            # Categorical features
            'cat1': ['A', 'B', 'A', 'C', 'B'],
            'cat2': ['X', 'Y', 'Z', 'X', 'Y'],
            # Date feature
            'date1': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05'],
            # Text feature
            'text1': ['This is a long text field for testing', 'Another long text', 
                     'More sample text that is quite long', 'Text features need to be long',
                     'This should be detected as a text field'],
            # Feature with missing values
            'missing': [1.0, np.nan, 3.0, np.nan, 5.0]
        })
        
        # Create a sample CSV file
        self.csv_path = os.path.join(self.temp_dir.name, 'sample.csv')
        self.sample_data.to_csv(self.csv_path, index=False)
        
        # Create a second sample CSV file with high cardinality
        high_card_data = pd.DataFrame({
            'id': [f'ID_{i}' for i in range(150)],
            'value': np.random.rand(150)
        })
        self.high_card_csv = os.path.join(self.temp_dir.name, 'high_card.csv')
        high_card_data.to_csv(self.high_card_csv, index=False)
        
        # Create correlated data
        corr_data = pd.DataFrame({
            'x': np.random.rand(100),
        })
        corr_data['y'] = corr_data['x'] * 0.9 + np.random.rand(100) * 0.1
        self.corr_csv = os.path.join(self.temp_dir.name, 'correlated.csv')
        corr_data.to_csv(self.corr_csv, index=False)

    def tearDown(self):
        """Tear down test fixtures, if any."""
        self.temp_dir.cleanup()

    def test_init(self):
        """Test initialization of DataAnalyzer."""
        # Check that the registrations dictionary is initialized
        self.assertIsInstance(self.analyzer.registrations, dict)
        self.assertGreater(len(self.analyzer.registrations), 0)
        
        # Check that default recommendations are registered
        self.assertIn('continuous_features', self.analyzer.registrations)
        self.assertIn('categorical_features', self.analyzer.registrations)
        self.assertIn('date_features', self.analyzer.registrations)

    def test_register_recommendation(self):
        """Test registration of layer recommendations."""
        # Register a new layer recommendation
        self.analyzer.register_recommendation(
            characteristic='test_characteristic',
            layer_name='TestLayer',
            description='Test description',
            use_case='Test use case'
        )
        
        # Check that the recommendation was registered
        self.assertIn('test_characteristic', self.analyzer.registrations)
        self.assertEqual(
            self.analyzer.registrations['test_characteristic'][-1],
            ('TestLayer', 'Test description', 'Test use case')
        )
        
        # Register a recommendation for an existing characteristic
        initial_count = len(self.analyzer.registrations['continuous_features'])
        self.analyzer.register_recommendation(
            characteristic='continuous_features',
            layer_name='CustomNumericalLayer',
            description='Custom layer for numerical features',
            use_case='Special numerical processing'
        )
        
        # Check that the recommendation was added
        self.assertEqual(
            len(self.analyzer.registrations['continuous_features']),
            initial_count + 1
        )
        self.assertEqual(
            self.analyzer.registrations['continuous_features'][-1][0],
            'CustomNumericalLayer'
        )

    def test_analyze_csv(self):
        """Test analysis of a CSV file."""
        # Analyze the sample CSV file
        stats = self.analyzer.analyze_csv(self.csv_path)
        
        # Check basic statistics
        self.assertEqual(stats['row_count'], 5)
        self.assertEqual(stats['column_count'], 7)
        
        # Check column types
        self.assertIn('column_types', stats)
        self.assertEqual(len(stats['column_types']), 7)
        
        # Check characteristics
        self.assertIn('characteristics', stats)
        characteristics = stats['characteristics']
        
        # Check continuous features
        self.assertIn('continuous_features', characteristics)
        self.assertIn('num1', characteristics['continuous_features'])
        self.assertIn('num2', characteristics['continuous_features'])
        
        # Check categorical features
        self.assertIn('categorical_features', characteristics)
        self.assertIn('cat1', characteristics['categorical_features'])
        self.assertIn('cat2', characteristics['categorical_features'])
        
        # Check date features
        self.assertIn('date_features', characteristics)
        self.assertIn('date1', characteristics['date_features'])
        
        # Check text features
        self.assertIn('text_features', characteristics)
        self.assertIn('text1', characteristics['text_features'])
        
        # Check missing values
        self.assertIn('high_missing_value_features', characteristics)
        self.assertIn('missing', characteristics['high_missing_value_features'])
        
        # Check numeric stats
        self.assertIn('numeric_stats', stats)
        self.assertIn('num1', stats['numeric_stats'])
        num1_stats = stats['numeric_stats']['num1']
        self.assertEqual(num1_stats['min'], 1.0)
        self.assertEqual(num1_stats['max'], 5.0)
        self.assertEqual(num1_stats['mean'], 3.0)

    def test_analyze_high_cardinality(self):
        """Test analysis of high cardinality features."""
        # Analyze the high cardinality CSV file
        stats = self.analyzer.analyze_csv(self.high_card_csv)
        
        # Check high cardinality detection
        self.assertIn('high_cardinality_categorical', stats['characteristics'])
        self.assertIn('id', stats['characteristics']['high_cardinality_categorical'])

    def test_analyze_correlated_features(self):
        """Test analysis of correlated features."""
        # Analyze the correlated features CSV
        stats = self.analyzer.analyze_csv(self.corr_csv)
        
        # Check feature interaction detection
        self.assertIn('feature_interaction', stats['characteristics'])
        interactions = stats['characteristics']['feature_interaction']
        self.assertEqual(len(interactions), 1)  # One pair of correlated features
        self.assertEqual(interactions[0][0], 'x')
        self.assertEqual(interactions[0][1], 'y')
        self.assertGreater(interactions[0][2], 0.7)  # Correlation coefficient > 0.7

    def test_analyze_directory(self):
        """Test analysis of a directory of CSV files."""
        # Analyze the directory
        results = self.analyzer.analyze_directory(self.temp_dir.name)
        
        # Check that all CSV files were analyzed
        self.assertEqual(len(results), 3)
        self.assertIn('sample.csv', results)
        self.assertIn('high_card.csv', results)
        self.assertIn('correlated.csv', results)
        
        # Check that each result contains the right structure
        for filename, result in results.items():
            self.assertIn('row_count', result)
            self.assertIn('characteristics', result)

    def test_recommend_layers(self):
        """Test layer recommendations based on statistics."""
        # Analyze the sample CSV
        stats = self.analyzer.analyze_csv(self.csv_path)
        
        # Get recommendations
        recommendations = self.analyzer.recommend_layers(stats)
        
        # Check that recommendations are provided for the detected characteristics
        for characteristic in stats['characteristics']:
            if characteristic in self.analyzer.registrations:
                self.assertIn(characteristic, recommendations)
                
        # Check structure of recommendations
        for characteristic, layers in recommendations.items():
            self.assertIsInstance(layers, list)
            if layers:  # If there are any layers recommended
                layer_info = layers[0]
                self.assertEqual(len(layer_info), 3)  # (name, description, use_case)

    def test_analyze_and_recommend(self):
        """Test combined analysis and recommendation."""
        # Analyze and recommend for a file
        result = self.analyzer.analyze_and_recommend(self.csv_path)
        
        # Check result structure
        self.assertIn('analysis', result)
        self.assertIn('recommendations', result)
        
        # Check file analysis
        self.assertEqual(result['analysis']['file'], 'sample.csv')
        self.assertIn('stats', result['analysis'])
        
        # Check recommendations
        self.assertIsInstance(result['recommendations'], dict)
        self.assertGreater(len(result['recommendations']), 0)
        
        # Analyze and recommend for a directory
        dir_result = self.analyzer.analyze_and_recommend(self.temp_dir.name)
        
        # Check directory analysis
        self.assertIsInstance(dir_result['analysis'], dict)
        self.assertGreater(len(dir_result['analysis']), 0)

    def test_analyze_data_function(self):
        """Test the analyze_data convenience function."""
        # Use the analyze_data function
        result = analyze_data(self.csv_path)
        
        # Check result
        self.assertIn('analysis', result)
        self.assertIn('recommendations', result)

    @mock.patch('kmr.utils.data_analyzer.pd.read_csv')
    def test_error_handling(self, mock_read_csv):
        """Test error handling when analyzing invalid files."""
        # Mock pd.read_csv to raise an exception
        mock_read_csv.side_effect = Exception("Test error")
        
        # Analyze an invalid CSV
        result = self.analyzer.analyze_csv('invalid.csv')
        
        # Check that an empty dict is returned
        self.assertEqual(result, {})

    def test_custom_pattern(self):
        """Test analysis with custom file pattern."""
        # Create a non-CSV file that should be excluded
        with open(os.path.join(self.temp_dir.name, 'data.txt'), 'w') as f:
            f.write('This is not a CSV file')
        
        # Analyze directory with custom pattern
        results = self.analyzer.analyze_directory(self.temp_dir.name, pattern='*.csv')
        
        # Check that only CSV files were analyzed
        self.assertEqual(len(results), 3)
        self.assertNotIn('data.txt', results)

    def test_empty_directory(self):
        """Test analysis of an empty directory."""
        # Create an empty directory
        empty_dir = tempfile.TemporaryDirectory()
        
        # Analyze the empty directory
        results = self.analyzer.analyze_directory(empty_dir.name)
        
        # Check that no results were returned
        self.assertEqual(results, {})
        
        # Clean up
        empty_dir.cleanup()

    def test_invalid_source(self):
        """Test analysis with an invalid source."""
        # Use a non-existent path
        non_existent_path = os.path.join(self.temp_dir.name, "non_existent")
        
        # Mock the logger to capture error messages
        with mock.patch("kmr.utils.data_analyzer.logger") as mock_logger:
            # Call analyze_and_recommend with invalid source
            result = self.analyzer.analyze_and_recommend(non_existent_path)
            
            # Verify that the logger.error was called
            mock_logger.error.assert_called_once()
            
            # Check that the result has the expected structure
            self.assertIsNone(result["analysis"])
            self.assertIsNone(result["recommendations"])


if __name__ == '__main__':
    unittest.main() 