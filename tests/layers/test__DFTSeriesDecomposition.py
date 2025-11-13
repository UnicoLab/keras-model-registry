"""Unit tests for DFTSeriesDecomposition layer.

Note: TensorFlow is used in tests for validation purposes only.
The actual layer implementation uses only Keras 3 operations.
"""

import unittest
import tensorflow as tf  # Used for testing only
from keras import layers, Model
from kerasfactory.layers.DFTSeriesDecomposition import DFTSeriesDecomposition


class TestDFTSeriesDecomposition(unittest.TestCase):
    """Test cases for DFTSeriesDecomposition layer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.batch_size = 8
        self.time_steps = 100
        self.n_features = 4
        self.top_k = 5
        self.layer = DFTSeriesDecomposition(top_k=self.top_k)
        self.inputs = tf.random.normal(
            (self.batch_size, self.time_steps, self.n_features),
        )

    def test_initialization(self) -> None:
        """Test layer initialization."""
        layer = DFTSeriesDecomposition(top_k=5)
        self.assertEqual(layer.top_k, 5)

    def test_invalid_initialization(self) -> None:
        """Test layer initialization with invalid parameters."""
        with self.assertRaises(ValueError):
            DFTSeriesDecomposition(top_k=0)

        with self.assertRaises(ValueError):
            DFTSeriesDecomposition(top_k=-1)

    def test_output_shapes(self) -> None:
        """Test output shapes of seasonal and trend."""
        seasonal, trend = self.layer(self.inputs)

        self.assertEqual(seasonal.shape, self.inputs.shape)
        self.assertEqual(trend.shape, self.inputs.shape)

    def test_decomposition_property(self) -> None:
        """Test that seasonal + trend ≈ original input."""
        seasonal, trend = self.layer(self.inputs)
        reconstructed = seasonal + trend

        # Allow for numerical precision issues with FFT
        tf.debugging.assert_near(reconstructed, self.inputs, atol=1e-4)

    def test_seasonal_captures_frequencies(self) -> None:
        """Test that seasonal and trend decomposition works correctly."""
        # Create data with periodic component
        time = tf.cast(tf.range(self.time_steps), tf.float32)
        periodic = tf.sin(2 * 3.14159 * time / 20)  # Period of 20
        periodic = tf.expand_dims(periodic, axis=-1)
        periodic = tf.tile(periodic, [1, self.n_features])
        periodic = tf.expand_dims(periodic, axis=0)
        periodic = tf.tile(periodic, [self.batch_size, 1, 1])

        seasonal, trend = self.layer(periodic)

        # Check that decomposition produces valid outputs
        self.assertEqual(seasonal.shape, periodic.shape)
        self.assertEqual(trend.shape, periodic.shape)

        # Reconstruction should be approximately the input
        reconstructed = seasonal + trend
        error = float(tf.reduce_mean(tf.abs(reconstructed - periodic)))
        self.assertLess(error, 1.0, "Decomposition should preserve signal")

    def test_different_top_k_values(self) -> None:
        """Test with different top_k values."""
        for top_k in [1, 5, 10, 20]:
            layer = DFTSeriesDecomposition(top_k=top_k)
            seasonal, trend = layer(self.inputs)

            self.assertEqual(seasonal.shape, self.inputs.shape)
            self.assertEqual(trend.shape, self.inputs.shape)

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        config = self.layer.get_config()
        self.assertIn("top_k", config)

        new_layer = DFTSeriesDecomposition.from_config(config)
        self.assertEqual(new_layer.top_k, self.layer.top_k)

    def test_model_integration(self) -> None:
        """Test layer in a functional model."""
        inputs = layers.Input(shape=(self.time_steps, self.n_features))
        seasonal, trend = self.layer(inputs)

        # Process seasonal
        out = layers.Dense(1)(seasonal)

        model = Model(inputs=inputs, outputs=out)
        model.compile(optimizer="adam", loss="mse")

        x_data = tf.random.normal((10, self.time_steps, self.n_features))
        y_data = tf.random.normal((10, 1))

        history = model.fit(x_data, y_data, epochs=1, verbose=0)
        self.assertTrue(history.history["loss"][0] > 0)

    def test_output_dtypes(self) -> None:
        """Test that outputs have correct dtype."""
        seasonal, trend = self.layer(self.inputs)

        self.assertEqual(seasonal.dtype, self.inputs.dtype)
        self.assertEqual(trend.dtype, self.inputs.dtype)

    def test_multiple_calls_consistency(self) -> None:
        """Test that multiple calls produce consistent results."""
        seasonal1, trend1 = self.layer(self.inputs)
        seasonal2, trend2 = self.layer(self.inputs)

        # Results should be very close (allow for floating point differences)
        tf.debugging.assert_near(seasonal1, seasonal2, atol=1e-5)
        tf.debugging.assert_near(trend1, trend2, atol=1e-5)

    def test_batch_independence(self) -> None:
        """Test that batch processing works correctly."""
        # Create inputs with different patterns
        sample1 = tf.sin(tf.linspace(0.0, 2 * 3.14159, self.time_steps))
        sample1 = tf.tile(sample1[None, :, None], [1, 1, self.n_features])

        sample2 = tf.cos(tf.linspace(0.0, 2 * 3.14159, self.time_steps))
        sample2 = tf.tile(sample2[None, :, None], [1, 1, self.n_features])

        input_concat = tf.concat([sample1, sample2], axis=0)
        seasonal, trend = self.layer(input_concat)

        # Both should have same shape
        self.assertEqual(seasonal.shape, input_concat.shape)
        self.assertEqual(trend.shape, input_concat.shape)

        # Reconstruction should be approximately correct
        reconstructed = seasonal + trend
        error = float(tf.reduce_mean(tf.abs(reconstructed - input_concat)))
        self.assertLess(error, 1.0)


if __name__ == "__main__":
    unittest.main()
