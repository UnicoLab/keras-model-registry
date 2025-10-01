"""Unit tests for the DateParsingLayer.

Note: TensorFlow is used in tests for validation purposes only.
The actual layer implementation uses only Keras 3 operations.
"""

import unittest
import numpy as np
import tensorflow as tf  # Used for testing only
from keras import Model, layers
from kmr.layers.DateParsingLayer import DateParsingLayer

class TestDateParsingLayer(unittest.TestCase):
    """Test cases for the DateParsingLayer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Using TensorFlow for test data generation only
        self.test_dates = tf.constant([
            "2023-01-15",  # Sunday
            "2022/12/31",  # Saturday
            "2021-06-01",  # Tuesday
            "2020/02/29",  # Saturday (leap year)
        ])
        tf.random.set_seed(42)  # For reproducibility

    def test_initialization(self) -> None:
        """Test layer initialization with various parameters."""
        # Test default initialization
        layer = DateParsingLayer()
        self.assertEqual(layer.date_format, "YYYY-MM-DD")
        
        # Test with custom date format
        layer = DateParsingLayer(date_format="YYYY/MM/DD")
        self.assertEqual(layer.date_format, "YYYY/MM/DD")

    def test_invalid_initialization(self) -> None:
        """Test layer initialization with invalid parameters."""
        # Test with unsupported date format
        with self.assertRaises(ValueError):
            DateParsingLayer(date_format="MM-DD-YYYY")

    def test_date_parsing(self) -> None:
        """Test that dates are correctly parsed into components."""
        layer = DateParsingLayer()
        output = layer(self.test_dates)
        
        # Expected components [year, month, day, day_of_week]
        expected = np.array([
            [2023, 1, 15, 0],   # Sunday (0)
            [2022, 12, 31, 6],  # Saturday (6)
            [2021, 6, 1, 2],    # Tuesday (2)
            [2020, 2, 29, 6],   # Saturday (6)
        ], dtype=np.int32)
        
        # Check that output matches expected
        np.testing.assert_array_equal(output.numpy(), expected)

    def test_date_format_handling(self) -> None:
        """Test that different date formats are handled correctly."""
        # Test with mixed formats
        mixed_formats = tf.constant([
            "2023-01-15",  # Hyphen format
            "2022/12/31",  # Slash format
        ])
        
        layer = DateParsingLayer()
        output = layer(mixed_formats)
        
        # Expected components [year, month, day, day_of_week]
        expected = np.array([
            [2023, 1, 15, 0],   # Sunday (0)
            [2022, 12, 31, 6],  # Saturday (6)
        ], dtype=np.int32)
        
        # Check that output matches expected
        np.testing.assert_array_equal(output.numpy(), expected)

    def test_day_of_week_calculation(self) -> None:
        """Test that day of week is calculated correctly."""
        # Test with dates with known day of week
        known_days = tf.constant([
            "2023-01-01",  # Sunday (0)
            "2023-01-02",  # Monday (1)
            "2023-01-03",  # Tuesday (2)
            "2023-01-04",  # Wednesday (3)
            "2023-01-05",  # Thursday (4)
            "2023-01-06",  # Friday (5)
            "2023-01-07",  # Saturday (6)
        ])
        
        layer = DateParsingLayer()
        output = layer(known_days)
        
        # Extract day of week (4th column)
        days_of_week = output.numpy()[:, 3]
        
        # Expected days of week (0=Sunday, 6=Saturday)
        expected_days = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int32)
        
        # Check that days of week match expected
        np.testing.assert_array_equal(days_of_week, expected_days)

    def test_leap_year_handling(self) -> None:
        """Test that leap years are handled correctly."""
        # Test with February 29 in leap and non-leap years
        leap_years = tf.constant([
            "2020-02-29",  # Valid leap year
            "2024-02-29",  # Valid leap year
        ])
        
        layer = DateParsingLayer()
        output = layer(leap_years)
        
        # Check that February 29 is parsed correctly in leap years
        self.assertEqual(output.numpy()[0, 1], 2)  # Month is February (2)
        self.assertEqual(output.numpy()[0, 2], 29)  # Day is 29
        self.assertEqual(output.numpy()[1, 1], 2)  # Month is February (2)
        self.assertEqual(output.numpy()[1, 2], 29)  # Day is 29

    def test_output_shape(self) -> None:
        """Test that output shape is correct."""
        layer = DateParsingLayer()
        
        # Test with various input shapes
        test_shapes = [
            (3,),      # 3 dates
            (2, 3),    # 2 batches of 3 dates
        ]
        
        for shape in test_shapes:
            # Create tensor with the given shape
            x = tf.fill(shape, "2023-01-15")
            output = layer(x)
            
            # Check that output shape is input_shape + (4,)
            expected_shape = shape + (4,)
            self.assertEqual(output.shape, expected_shape)

    def test_compute_output_shape(self) -> None:
        """Test the compute_output_shape method."""
        layer = DateParsingLayer()
        
        # Test with various input shapes
        test_shapes = [
            (3,),      # 3 dates
            (2, 3),    # 2 batches of 3 dates
            (None, 5)  # Unknown batch size, 5 dates
        ]
        
        for shape in test_shapes:
            output_shape = layer.compute_output_shape(shape)
            expected_shape = shape + (4,)
            self.assertEqual(output_shape, expected_shape)

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        original_layer = DateParsingLayer(date_format="YYYY/MM/DD")
        config = original_layer.get_config()
        
        # Create new layer from config
        restored_layer = DateParsingLayer.from_config(config)
        
        # Check if configurations match
        self.assertEqual(restored_layer.date_format, original_layer.date_format)
        
        # Check that outputs match
        original_output = original_layer(self.test_dates)
        restored_output = restored_layer(self.test_dates)
        
        np.testing.assert_array_equal(
            original_output.numpy(),
            restored_output.numpy()
        )

    def test_integration(self) -> None:
        """Test integration with a simple model."""
        # Create a simple model with the date parsing layer
        inputs = layers.Input(shape=(), dtype="string")
        x = DateParsingLayer()(inputs)
        
        # Flatten the output and add a dense layer
        x = layers.Flatten()(x)
        outputs = layers.Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile the model
        model.compile(optimizer="adam", loss="mse")
        
        # Generate some dummy data
        x_data = tf.constant(["2023-01-15", "2022-12-31"])
        y_data = tf.constant([[1.0], [2.0]])
        
        # Fit the model for one step to ensure everything works
        model.fit(x_data, y_data, epochs=1, verbose=0)
        
        # Make a prediction to ensure the pipeline works
        pred = model.predict(tf.constant(["2021-06-01"]))
        self.assertEqual(pred.shape, (1, 1))

if __name__ == "__main__":
    unittest.main() 