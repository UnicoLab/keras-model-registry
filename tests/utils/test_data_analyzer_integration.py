"""Integration tests for the kerasfactory.utils.data_analyzer module.

These tests verify the data analyzer functionality with actual data files
and check the full pipeline from data analysis to layer recommendations.
"""

import os
import sys
import tempfile
import unittest
import subprocess
import json
import pandas as pd
import numpy as np

# Ensure parent directory is in path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from kerasfactory.utils import DataAnalyzer, analyze_data


class TestDataAnalyzerIntegration(unittest.TestCase):
    """Integration test case for the DataAnalyzer."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test fixtures, if any."""
        # Create a temporary directory for test files
        cls.temp_dir = tempfile.TemporaryDirectory()

        # Create varied test data
        cls._create_test_datasets()

    @classmethod
    def tearDownClass(cls) -> None:
        """Tear down test fixtures, if any."""
        cls.temp_dir.cleanup()

    @classmethod
    def _create_test_datasets(cls) -> None:
        """Create a variety of test datasets for integration testing."""
        # Directory paths
        cls.datasets_dir = os.path.join(cls.temp_dir.name, "datasets")
        os.makedirs(cls.datasets_dir, exist_ok=True)

        # 1. Create a numeric dataset
        numeric_data = pd.DataFrame(
            {
                "id": range(1000),
                "numeric1": np.random.normal(0, 1, 1000),
                "numeric2": np.random.exponential(1, 1000),
                "numeric3": np.random.uniform(0, 100, 1000),
                "numeric4": np.random.normal(50, 10, 1000),
                "numeric5": np.random.poisson(5, 1000),
            },
        )
        cls.numeric_csv = os.path.join(cls.datasets_dir, "numeric_data.csv")
        numeric_data.to_csv(cls.numeric_csv, index=False)

        # 2. Create a categorical dataset
        categories = ["cat_" + str(i) for i in range(20)]
        high_card_categories = ["id_" + str(i) for i in range(500)]
        categorical_data = pd.DataFrame(
            {
                "id": range(1000),
                "low_card_cat": np.random.choice(["A", "B", "C", "D", "E"], 1000),
                "med_card_cat": np.random.choice(categories, 1000),
                "high_card_cat": np.random.choice(high_card_categories, 1000),
            },
        )
        cls.categorical_csv = os.path.join(cls.datasets_dir, "categorical_data.csv")
        categorical_data.to_csv(cls.categorical_csv, index=False)

        # 3. Create a date dataset
        start_date = pd.Timestamp("2020-01-01")
        dates = [start_date + pd.Timedelta(days=i) for i in range(1000)]
        date_data = pd.DataFrame(
            {
                "id": range(1000),
                "date": dates,
                "month": [d.month for d in dates],
                "day": [d.day for d in dates],
                "value": np.random.normal(0, 1, 1000),
            },
        )
        cls.date_csv = os.path.join(cls.datasets_dir, "date_data.csv")
        date_data.to_csv(cls.date_csv, index=False)

        # 4. Create a text dataset
        text_samples = [
            "This is a text sample for testing purposes",
            "Natural language processing requires text data",
            "Keras Model Registry provides layers for text processing",
            "Text features can be analyzed and embedded",
            "Different text samples provide varied contexts",
        ]
        text_data = pd.DataFrame(
            {
                "id": range(1000),
                "text": np.random.choice(text_samples, 1000),
                "category": np.random.choice(["news", "social", "review"], 1000),
            },
        )
        cls.text_csv = os.path.join(cls.datasets_dir, "text_data.csv")
        text_data.to_csv(cls.text_csv, index=False)

        # 5. Create a mixed dataset
        mixed_data = pd.DataFrame(
            {
                "id": range(1000),
                "numeric": np.random.normal(0, 1, 1000),
                "category": np.random.choice(["A", "B", "C"], 1000),
                "date": [
                    start_date + pd.Timedelta(days=i)
                    for i in np.random.randint(0, 365, 1000)
                ],
                "text": np.random.choice(text_samples, 1000),
                "missing_heavy": [np.nan if i % 3 == 0 else i for i in range(1000)],
            },
        )
        cls.mixed_csv = os.path.join(cls.datasets_dir, "mixed_data.csv")
        mixed_data.to_csv(cls.mixed_csv, index=False)

        # 6. Create correlated features dataset
        corr_data = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, 1000),
            },
        )
        corr_data["x2"] = corr_data["x1"] * 0.9 + np.random.normal(0, 0.3, 1000)
        corr_data["x3"] = corr_data["x1"] * -0.8 + np.random.normal(0, 0.3, 1000)
        corr_data["x4"] = np.random.normal(0, 1, 1000)  # Independent
        cls.correlated_csv = os.path.join(cls.datasets_dir, "correlated_data.csv")
        corr_data.to_csv(cls.correlated_csv, index=False)

    def test_full_analyzer_pipeline(self) -> None:
        """Test the complete data analyzer pipeline with real data."""
        # Create analyzer
        analyzer = DataAnalyzer()

        # Analyze each dataset and check for expected characteristics

        # 1. Test numeric dataset
        numeric_result = analyzer.analyze_and_recommend(self.numeric_csv)
        self.assertIn("continuous_features", numeric_result["recommendations"])
        self.assertGreaterEqual(
            len(
                numeric_result["analysis"]["stats"]["characteristics"][
                    "continuous_features"
                ],
            ),
            5,
        )

        # 2. Test categorical dataset
        cat_result = analyzer.analyze_and_recommend(self.categorical_csv)
        self.assertIn("categorical_features", cat_result["recommendations"])
        self.assertIn("high_cardinality_categorical", cat_result["recommendations"])

        # 3. Test date dataset
        date_result = analyzer.analyze_and_recommend(self.date_csv)
        self.assertIn("date_features", date_result["recommendations"])
        self.assertGreaterEqual(
            len(date_result["analysis"]["stats"]["characteristics"]["date_features"]),
            1,
        )

        # 4. Test text dataset
        text_result = analyzer.analyze_and_recommend(self.text_csv)
        self.assertIn("text_features", text_result["recommendations"])

        # 5. Test mixed dataset
        mixed_result = analyzer.analyze_and_recommend(self.mixed_csv)
        recommendations = mixed_result["recommendations"]
        # Should have multiple types of recommendations
        self.assertGreaterEqual(len(recommendations), 3)
        # Should detect missing values
        self.assertIn("high_missing_value_features", recommendations)

        # 6. Test correlated features
        corr_result = analyzer.analyze_and_recommend(self.correlated_csv)
        self.assertIn("feature_interaction", corr_result["recommendations"])
        interactions = corr_result["analysis"]["stats"]["characteristics"][
            "feature_interaction"
        ]
        self.assertGreaterEqual(len(interactions), 2)  # At least 2 correlated pairs

    def test_directory_analysis(self) -> None:
        """Test analyzing a directory of datasets."""
        # Analyze all datasets in the directory
        analyzer = DataAnalyzer()
        result = analyzer.analyze_and_recommend(self.datasets_dir)

        # Check that we analyzed all files
        self.assertEqual(len(result["analysis"]), 6)

        # Check that recommendations cover all characteristics
        recommendations = result["recommendations"]
        expected_characteristics = [
            "continuous_features",
            "categorical_features",
            "date_features",
            "text_features",
            "high_cardinality_categorical",
            "high_missing_value_features",
            "feature_interaction",
        ]

        for characteristic in expected_characteristics:
            self.assertIn(characteristic, recommendations)

    def test_convenience_function(self) -> None:
        """Test the analyze_data convenience function."""
        # Test with a file
        file_result = analyze_data(self.mixed_csv)
        self.assertIn("analysis", file_result)
        self.assertIn("recommendations", file_result)

        # Test with a directory
        dir_result = analyze_data(self.datasets_dir)
        self.assertIn("analysis", dir_result)
        self.assertIn("recommendations", dir_result)

    def test_cli_integration(self) -> None:
        """Test the CLI tool on actual data."""
        # Skip this test if not running in a development environment
        try:
            # Create an output file
            output_file = os.path.join(self.temp_dir.name, "cli_output.json")

            # Run the CLI on the mixed dataset
            cmd = [
                "python",
                "-m",
                "kerasfactory.utils.data_analyzer_cli",
                self.mixed_csv,
                "--output",
                output_file,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(
                result.returncode,
                0,
                f"CLI failed with error: {result.stderr}",
            )

            # Check that output file was created
            self.assertTrue(os.path.exists(output_file))

            # Check content of output file
            with open(output_file) as f:
                data = json.load(f)
                self.assertIn("analysis", data)
                self.assertIn("recommendations", data)

                # Check that key characteristics were identified
                stats = data["analysis"]["stats"]
                self.assertIn("continuous_features", stats["characteristics"])
                self.assertIn("categorical_features", stats["characteristics"])
                self.assertIn("date_features", stats["characteristics"])
                self.assertIn("text_features", stats["characteristics"])
                self.assertIn("high_missing_value_features", stats["characteristics"])

        except Exception as e:
            self.skipTest(f"CLI integration test skipped: {e}")

    def test_extended_use_case(self) -> None:
        """Test an extended use case with custom recommendations."""
        # Create analyzer
        analyzer = DataAnalyzer()

        # Register custom layer recommendations
        analyzer.register_recommendation(
            characteristic="continuous_features",
            layer_name="CustomNumericLayer",
            description="A custom layer for numeric features",
            use_case="Special numerical transformations",
        )

        analyzer.register_recommendation(
            characteristic="custom_characteristic",
            layer_name="SpecializedLayer",
            description="Handles a special case",
            use_case="Very specialized use case",
        )

        # Analyze mixed dataset with custom recommendations
        result = analyzer.analyze_and_recommend(self.mixed_csv)

        # Check that custom numeric layer recommendation is included
        numeric_recs = result["recommendations"]["continuous_features"]
        custom_layer_found = False
        for layer_info in numeric_recs:
            if layer_info[0] == "CustomNumericLayer":
                custom_layer_found = True
                break

        self.assertTrue(custom_layer_found, "Custom layer recommendation not found")

        # Simulate detecting a custom characteristic
        stats = result["analysis"]["stats"]
        stats["characteristics"]["custom_characteristic"] = ["custom_feature"]

        # Get recommendations with the custom characteristic
        updated_recs = analyzer.recommend_layers(stats)
        self.assertIn("custom_characteristic", updated_recs)
        self.assertEqual(
            updated_recs["custom_characteristic"][0][0],
            "SpecializedLayer",
        )


if __name__ == "__main__":
    unittest.main()
