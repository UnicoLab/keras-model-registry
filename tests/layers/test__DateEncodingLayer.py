"""Unit tests for the DateEncodingLayer.

Note: TensorFlow is used in tests for validation purposes only.
The actual layer implementation uses only Keras 3 operations.
"""

import unittest
import numpy as np
import tensorflow as tf  # Used for testing only
from keras import Model, layers
from kmr.layers.DateEncodingLayer import DateEncodingLayer


class TestDateEncodingLayer(unittest.TestCase):
    """Test cases for the DateEncodingLayer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Using TensorFlow for test data generation only
        self.test_dates = tf.constant(
            [
                [2023, 1, 15, 0],  # Year, month, day, day_of_week
                [2022, 12, 31, 6],
                [2021, 6, 1, 2],
                [2020, 2, 29, 6],
            ],
            dtype=tf.int32,
        )
        tf.random.set_seed(42)  # For reproducibility

    def test_initialization(self) -> None:
        """Test layer initialization with various parameters."""
        # Test default initialization
        layer = DateEncodingLayer()
        self.assertEqual(layer.min_year, 1900)
        self.assertEqual(layer.max_year, 2100)

        # Test with custom parameters
        layer = DateEncodingLayer(min_year=2000, max_year=2050)
        self.assertEqual(layer.min_year, 2000)
        self.assertEqual(layer.max_year, 2050)

    def test_invalid_initialization(self) -> None:
        """Test layer initialization with invalid parameters."""
        # Test with min_year >= max_year
        with self.assertRaises(ValueError):
            DateEncodingLayer(min_year=2000, max_year=2000)

        with self.assertRaises(ValueError):
            DateEncodingLayer(min_year=2050, max_year=2000)

    def test_encoding(self) -> None:
        """Test that date components are correctly encoded."""
        layer = DateEncodingLayer(min_year=2000, max_year=2050)
        output = layer(self.test_dates)

        # Check output shape
        self.assertEqual(output.shape, (4, 8))

        # Check that values are in the expected range [-1, 1]
        self.assertTrue(np.all(output.numpy() >= -1.0))
        self.assertTrue(np.all(output.numpy() <= 1.0))

        # Check specific encoding for a known date
        # For year 2023 with min_year=2000, max_year=2050
        # normalized_year = (2023 - 2000) / (2050 - 2000) = 0.46
        # year_sin = sin(2π * 0.46) ≈ 0.309
        # year_cos = cos(2π * 0.46) ≈ -0.951
        year_sin = np.sin(2 * np.pi * 0.46)
        year_cos = np.cos(2 * np.pi * 0.46)

        # Allow for small floating point differences
        self.assertAlmostEqual(output.numpy()[0, 0], year_sin, places=5)
        self.assertAlmostEqual(output.numpy()[0, 1], year_cos, places=5)

    def test_cyclical_encoding(self) -> None:
        """Test that cyclical encoding works correctly."""
        # Create test data with month values
        months = tf.constant(
            [
                [2020, 1, 1, 0],  # January
                [2020, 4, 1, 0],  # April
                [2020, 7, 1, 0],  # July
                [2020, 10, 1, 0],  # October
            ],
            dtype=tf.int32,
        )

        layer = DateEncodingLayer()
        output = layer(months)

        # Extract month sine and cosine values
        month_sin = output.numpy()[:, 2]
        month_cos = output.numpy()[:, 3]

        # Expected values for months 1, 4, 7, 10
        expected_sin = np.sin(2 * np.pi * np.array([1, 4, 7, 10]) / 12)
        expected_cos = np.cos(2 * np.pi * np.array([1, 4, 7, 10]) / 12)

        # Check that values match expected
        np.testing.assert_allclose(month_sin, expected_sin, rtol=1e-5)
        np.testing.assert_allclose(month_cos, expected_cos, rtol=1e-5)

    def test_day_of_week_encoding(self) -> None:
        """Test that day of week encoding works correctly."""
        # Create test data with day of week values
        days_of_week = tf.constant(
            [
                [2020, 1, 1, 0],  # Sunday
                [2020, 1, 1, 1],  # Monday
                [2020, 1, 1, 3],  # Wednesday
                [2020, 1, 1, 6],  # Saturday
            ],
            dtype=tf.int32,
        )

        layer = DateEncodingLayer()
        output = layer(days_of_week)

        # Extract day of week sine and cosine values
        dow_sin = output.numpy()[:, 6]
        dow_cos = output.numpy()[:, 7]

        # Expected values for days 0, 1, 3, 6
        expected_sin = np.sin(2 * np.pi * np.array([0, 1, 3, 6]) / 7)
        expected_cos = np.cos(2 * np.pi * np.array([0, 1, 3, 6]) / 7)

        # Check that values match expected
        np.testing.assert_allclose(dow_sin, expected_sin, rtol=1e-5)
        np.testing.assert_allclose(dow_cos, expected_cos, rtol=1e-5)

    def test_output_shape(self) -> None:
        """Test that output shape is correct."""
        layer = DateEncodingLayer()

        # Test with various input shapes
        test_shapes = [
            (4,),  # 4 components
            (3, 4),  # 3 dates with 4 components each
            (2, 3, 4),  # 2 batches of 3 dates with 4 components each
        ]

        for shape in test_shapes:
            # Create tensor with the given shape
            x = tf.ones(shape, dtype=tf.int32)
            output = layer(x)

            # Check that output shape is input_shape[:-1] + (8,)
            expected_shape = shape[:-1] + (8,)
            self.assertEqual(output.shape, expected_shape)

    def test_compute_output_shape(self) -> None:
        """Test the compute_output_shape method."""
        layer = DateEncodingLayer()

        # Test with various input shapes
        test_shapes = [
            (4,),
            (3, 4),
            (None, 4),
        ]  # 4 components  # 3 dates with 4 components each  # Unknown batch size

        for shape in test_shapes:
            output_shape = layer.compute_output_shape(shape)
            expected_shape = shape[:-1] + (8,)
            self.assertEqual(output_shape, expected_shape)

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        original_layer = DateEncodingLayer(min_year=2000, max_year=2050)
        config = original_layer.get_config()

        # Create new layer from config
        restored_layer = DateEncodingLayer.from_config(config)

        # Check if configurations match
        self.assertEqual(restored_layer.min_year, original_layer.min_year)
        self.assertEqual(restored_layer.max_year, original_layer.max_year)

        # Check that outputs match
        original_output = original_layer(self.test_dates)
        restored_output = restored_layer(self.test_dates)

        np.testing.assert_allclose(original_output.numpy(), restored_output.numpy())

    def test_integration(self) -> None:
        """Test integration with a simple model."""
        # Create a simple model with the date encoding layer
        inputs = layers.Input(shape=(4,), dtype="int32")
        x = DateEncodingLayer()(inputs)
        outputs = layers.Dense(1)(x)

        model = Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(optimizer="adam", loss="mse")

        # Generate some dummy data
        x_data = tf.constant([[2023, 1, 15, 0], [2022, 12, 31, 6]], dtype=tf.int32)
        y_data = tf.constant([[1.0], [2.0]])

        # Fit the model for one step to ensure everything works
        model.fit(x_data, y_data, epochs=1, verbose=0)

        # Make a prediction to ensure the pipeline works
        pred = model.predict(tf.constant([[2021, 6, 1, 2]], dtype=tf.int32))
        self.assertEqual(pred.shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
