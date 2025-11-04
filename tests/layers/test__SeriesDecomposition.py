"""Unit tests for SeriesDecomposition layer.

Note: TensorFlow is used in tests for validation purposes only.
The actual layer implementation uses only Keras 3 operations.
"""

import unittest
import tensorflow as tf  # Used for testing only
from keras import layers, Model
from kmr.layers.SeriesDecomposition import SeriesDecomposition


class TestSeriesDecomposition(unittest.TestCase):
    """Test cases for SeriesDecomposition layer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.batch_size = 16
        self.time_steps = 100
        self.n_features = 8
        self.kernel_size = 25
        self.layer = SeriesDecomposition(kernel_size=self.kernel_size)
        self.inputs = tf.random.normal(
            (self.batch_size, self.time_steps, self.n_features),
        )

    def test_initialization(self) -> None:
        """Test layer initialization."""
        layer = SeriesDecomposition(kernel_size=25)
        self.assertEqual(layer.kernel_size, 25)

    def test_invalid_initialization(self) -> None:
        """Test layer initialization with invalid parameters."""
        with self.assertRaises(ValueError):
            SeriesDecomposition(kernel_size=0)

        with self.assertRaises(ValueError):
            SeriesDecomposition(kernel_size=-1)

    def test_output_shapes(self) -> None:
        """Test output shapes of both seasonal and trend."""
        seasonal, trend = self.layer(self.inputs)

        # Both should have same shape as input
        self.assertEqual(seasonal.shape, self.inputs.shape)
        self.assertEqual(trend.shape, self.inputs.shape)

    def test_decomposition_property(self) -> None:
        """Test that seasonal + trend ≈ original input."""
        seasonal, trend = self.layer(self.inputs)
        reconstructed = seasonal + trend

        # Reconstructed should equal original (up to numerical precision)
        tf.debugging.assert_near(reconstructed, self.inputs, atol=1e-5)

    def test_trend_is_smooth(self) -> None:
        """Test that trend component is smoother than input."""
        seasonal, trend = self.layer(self.inputs)

        # Compute variance along time axis
        input_var = tf.reduce_mean(tf.math.reduce_variance(self.inputs, axis=1))
        trend_var = tf.reduce_mean(tf.math.reduce_variance(trend, axis=1))

        # Trend should have lower variance (smoother)
        self.assertLess(float(trend_var), float(input_var))

    def test_multiple_decompositions(self) -> None:
        """Test multiple decompositions produce consistent results."""
        seasonal1, trend1 = self.layer(self.inputs)
        seasonal2, trend2 = self.layer(self.inputs)

        # Results should be identical
        tf.debugging.assert_equal(seasonal1, seasonal2)
        tf.debugging.assert_equal(trend1, trend2)

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        config = self.layer.get_config()
        self.assertIn("kernel_size", config)

        new_layer = SeriesDecomposition.from_config(config)
        self.assertEqual(new_layer.kernel_size, self.layer.kernel_size)

    def test_model_integration(self) -> None:
        """Test layer in a functional model."""
        inputs = layers.Input(shape=(self.time_steps, self.n_features))
        seasonal, trend = self.layer(inputs)

        # Process trend
        trend_out = layers.Dense(1)(trend)

        model = Model(inputs=inputs, outputs=trend_out)
        model.compile(optimizer="adam", loss="mse")

        x_data = tf.random.normal((10, self.time_steps, self.n_features))
        y_data = tf.random.normal((10, 1))

        history = model.fit(x_data, y_data, epochs=1, verbose=0)
        self.assertTrue(history.history["loss"][0] > 0)

    def test_different_kernel_sizes(self) -> None:
        """Test with different kernel sizes."""
        for kernel_size in [3, 7, 25, 51]:
            layer = SeriesDecomposition(kernel_size=kernel_size)
            seasonal, trend = layer(self.inputs)

            self.assertEqual(seasonal.shape, self.inputs.shape)
            self.assertEqual(trend.shape, self.inputs.shape)

            # Reconstruction property should hold
            reconstructed = seasonal + trend
            tf.debugging.assert_near(reconstructed, self.inputs, atol=1e-4)

    def test_different_batch_sizes(self) -> None:
        """Test with different batch sizes."""
        for batch_size in [1, 4, 16, 32]:
            inputs = tf.random.normal((batch_size, self.time_steps, self.n_features))
            seasonal, trend = self.layer(inputs)

            self.assertEqual(seasonal.shape[0], batch_size)
            self.assertEqual(trend.shape[0], batch_size)

    def test_seasonal_component_contains_oscillations(self) -> None:
        """Test that seasonal component contains oscillations."""
        # Create synthetic data with clear trend
        time_axis = tf.cast(tf.range(self.time_steps), tf.float32)
        trend_component = tf.expand_dims(
            time_axis / self.time_steps,
            axis=-1,
        )  # Linear trend
        trend_component = tf.tile(trend_component, [1, self.n_features])
        trend_component = tf.expand_dims(trend_component, axis=0)  # Add batch
        trend_component = tf.tile(trend_component, [self.batch_size, 1, 1])

        # Add seasonal oscillation
        seasonal_component = 0.5 * tf.sin(2 * 3.14159 * time_axis / 20)
        seasonal_component = tf.expand_dims(seasonal_component, axis=-1)
        seasonal_component = tf.tile(seasonal_component, [1, self.n_features])
        seasonal_component = tf.expand_dims(seasonal_component, axis=0)
        seasonal_component = tf.tile(seasonal_component, [self.batch_size, 1, 1])

        data = trend_component + seasonal_component

        seasonal_out, trend_out = self.layer(data)

        # Seasonal output should have significant oscillations
        seasonal_var = tf.reduce_mean(tf.math.reduce_variance(seasonal_out, axis=1))
        self.assertGreater(float(seasonal_var), 0.01)


if __name__ == "__main__":
    unittest.main()
