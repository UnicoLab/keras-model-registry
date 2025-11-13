"""Unit tests for the SeasonLayer.

Note: TensorFlow is used in tests for validation purposes only.
The actual layer implementation uses only Keras 3 operations.
"""

import unittest
import numpy as np
import tensorflow as tf  # Used for testing only
from keras import Model, layers
from kerasfactory.layers.SeasonLayer import SeasonLayer


class TestSeasonLayer(unittest.TestCase):
    """Test cases for the SeasonLayer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Using TensorFlow for test data generation only
        # Create test data for all months
        self.test_dates = tf.constant(
            [
                [2023, 1, 15, 0],  # January - Winter
                [2023, 2, 15, 0],  # February - Winter
                [2023, 3, 15, 0],  # March - Spring
                [2023, 4, 15, 0],  # April - Spring
                [2023, 5, 15, 0],  # May - Spring
                [2023, 6, 15, 0],  # June - Summer
                [2023, 7, 15, 0],  # July - Summer
                [2023, 8, 15, 0],  # August - Summer
                [2023, 9, 15, 0],  # September - Fall
                [2023, 10, 15, 0],  # October - Fall
                [2023, 11, 15, 0],  # November - Fall
                [2023, 12, 15, 0],  # December - Winter
            ],
            dtype=tf.int32,
        )

        tf.random.set_seed(42)  # For reproducibility

    def test_initialization(self) -> None:
        """Test layer initialization."""
        layer = SeasonLayer()
        self.assertIsInstance(layer, SeasonLayer)

    def test_season_encoding(self) -> None:
        """Test that seasons are correctly encoded."""
        layer = SeasonLayer()
        output = layer(self.test_dates)

        # Check output shape
        self.assertEqual(output.shape, (12, 8))

        # Extract season encodings (last 4 columns)
        seasons = output.numpy()[:, 4:]

        # Expected seasons for each month
        # Winter: December (12), January (1), February (2)
        # Spring: March (3), April (4), May (5)
        # Summer: June (6), July (7), August (8)
        # Fall: September (9), October (10), November (11)
        expected_seasons = np.array(
            [
                [1, 0, 0, 0],  # January - Winter
                [1, 0, 0, 0],  # February - Winter
                [0, 1, 0, 0],  # March - Spring
                [0, 1, 0, 0],  # April - Spring
                [0, 1, 0, 0],  # May - Spring
                [0, 0, 1, 0],  # June - Summer
                [0, 0, 1, 0],  # July - Summer
                [0, 0, 1, 0],  # August - Summer
                [0, 0, 0, 1],  # September - Fall
                [0, 0, 0, 1],  # October - Fall
                [0, 0, 0, 1],  # November - Fall
                [1, 0, 0, 0],  # December - Winter
            ],
        )

        # Check that seasons match expected
        np.testing.assert_array_equal(seasons, expected_seasons)

    def test_original_data_preservation(self) -> None:
        """Test that original date components are preserved."""
        layer = SeasonLayer()
        output = layer(self.test_dates)

        # Extract original components (first 4 columns)
        original = output.numpy()[:, :4]

        # Check that original components match input
        np.testing.assert_array_equal(original, self.test_dates.numpy())

    def test_output_shape(self) -> None:
        """Test that output shape is correct."""
        layer = SeasonLayer()

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

            # Check that output shape is input_shape[:-1] + (input_shape[-1] + 4,)
            expected_shape = shape[:-1] + (shape[-1] + 4,)
            self.assertEqual(output.shape, expected_shape)

    def test_compute_output_shape(self) -> None:
        """Test the compute_output_shape method."""
        layer = SeasonLayer()

        # Test with various input shapes
        test_shapes = [
            (4,),
            (3, 4),
            (None, 4),
        ]  # 4 components  # 3 dates with 4 components each  # Unknown batch size

        for shape in test_shapes:
            output_shape = layer.compute_output_shape(shape)
            expected_shape = shape[:-1] + (shape[-1] + 4,)
            self.assertEqual(output_shape, expected_shape)

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        original_layer = SeasonLayer()
        config = original_layer.get_config()

        # Create new layer from config
        restored_layer = SeasonLayer.from_config(config)

        # Check that outputs match
        original_output = original_layer(self.test_dates)
        restored_output = restored_layer(self.test_dates)

        np.testing.assert_array_equal(original_output.numpy(), restored_output.numpy())

    def test_integration(self) -> None:
        """Test integration with a simple model."""
        # Create a simple model with the season layer
        inputs = layers.Input(shape=(4,), dtype="int32")
        x = SeasonLayer()(inputs)
        outputs = layers.Dense(1)(x)

        model = Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(optimizer="adam", loss="mse")

        # Generate some dummy data
        x_data = tf.constant([[2023, 1, 15, 0], [2023, 7, 15, 0]], dtype=tf.int32)
        y_data = tf.constant([[1.0], [2.0]])

        # Fit the model for one step to ensure everything works
        model.fit(x_data, y_data, epochs=1, verbose=0)

        # Make a prediction to ensure the pipeline works
        pred = model.predict(tf.constant([[2023, 4, 1, 2]], dtype=tf.int32))
        self.assertEqual(pred.shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
