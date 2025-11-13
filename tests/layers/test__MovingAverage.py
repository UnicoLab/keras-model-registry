"""Unit tests for MovingAverage layer.

Note: TensorFlow is used in tests for validation purposes only.
The actual layer implementation uses only Keras 3 operations.
"""

import unittest
import tensorflow as tf  # Used for testing only
from keras import layers, Model
from kerasfactory.layers.MovingAverage import MovingAverage


class TestMovingAverage(unittest.TestCase):
    """Test cases for MovingAverage layer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.batch_size = 16
        self.time_steps = 100
        self.n_features = 8
        self.kernel_size = 25
        self.layer = MovingAverage(kernel_size=self.kernel_size)
        # Using TensorFlow for test data generation only
        self.inputs = tf.random.normal(
            (self.batch_size, self.time_steps, self.n_features),
        )

    def test_initialization(self) -> None:
        """Test layer initialization with various parameters."""
        # Test default initialization
        layer = MovingAverage(kernel_size=25)
        self.assertEqual(layer.kernel_size, 25)

        # Test with different kernel sizes
        for kernel_size in [5, 15, 51]:
            layer = MovingAverage(kernel_size=kernel_size)
            self.assertEqual(layer.kernel_size, kernel_size)

    def test_invalid_initialization(self) -> None:
        """Test layer initialization with invalid parameters."""
        # Test with non-positive kernel size
        with self.assertRaises(ValueError):
            MovingAverage(kernel_size=0)

        with self.assertRaises(ValueError):
            MovingAverage(kernel_size=-1)

        # Test with non-integer kernel size
        with self.assertRaises(ValueError):
            MovingAverage(kernel_size=3.5)

    def test_output_shape(self) -> None:
        """Test output shape of the layer."""
        outputs = self.layer(self.inputs)

        # Check output shape matches input shape
        self.assertEqual(outputs.shape, self.inputs.shape)

        # Test with different shapes
        test_shapes = [
            (8, 50, 4),
            (32, 200, 16),
            (1, 100, 1),
        ]
        for shape in test_shapes:
            inputs = tf.random.normal(shape)
            outputs = self.layer(inputs)
            self.assertEqual(outputs.shape, shape)

    def test_call(self) -> None:
        """Test forward pass of the layer."""
        # Call the layer
        outputs = self.layer(self.inputs)

        # Check that outputs are not None
        self.assertIsNotNone(outputs)

        # Check that outputs have proper dtype
        self.assertEqual(outputs.dtype, self.inputs.dtype)

    def test_smoothing_effect(self) -> None:
        """Test that moving average smooths the signal."""
        # Create a noisy signal
        signal = tf.constant(
            [[[1.0], [2.0], [3.0], [4.0], [5.0], [4.0], [3.0], [2.0], [1.0]]],
            dtype=tf.float32,
        )

        layer = MovingAverage(kernel_size=3)
        smoothed = layer(signal)

        # Check that smoothing reduces variance
        signal_var = tf.math.reduce_variance(signal)
        smoothed_var = tf.math.reduce_variance(smoothed)

        self.assertGreater(float(signal_var), float(smoothed_var))

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        # Get layer config
        config = self.layer.get_config()

        # Check config contains kernel_size
        self.assertIn("kernel_size", config)
        self.assertEqual(config["kernel_size"], self.kernel_size)

        # Create layer from config
        new_layer = MovingAverage.from_config(config)
        self.assertEqual(new_layer.kernel_size, self.layer.kernel_size)

        # Test in a model
        inputs = layers.Input(shape=(self.time_steps, self.n_features))
        outputs = self.layer(inputs)
        model = Model(inputs=inputs, outputs=outputs)

        # Get model config
        model_config = model.get_config()
        reloaded_model = Model.from_config(model_config)

        # Test predictions match
        test_input = tf.random.normal((1, self.time_steps, self.n_features))
        original_output = model(test_input)
        reloaded_output = reloaded_model(test_input)

        tf.debugging.assert_near(original_output, reloaded_output, atol=1e-5)

    def test_model_integration(self) -> None:
        """Test layer integration in a model."""
        # Create a simple model with moving average
        inputs = layers.Input(shape=(self.time_steps, self.n_features))
        x = MovingAverage(kernel_size=11)(inputs)
        outputs = layers.Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)

        # Compile and check it works
        model.compile(optimizer="adam", loss="mse")

        # Generate dummy data
        x_data = tf.random.normal((10, self.time_steps, self.n_features))
        y_data = tf.random.normal((10, 1))

        # Train for one epoch
        history = model.fit(x_data, y_data, epochs=1, verbose=0)
        self.assertTrue(history.history["loss"][0] > 0)

    def test_temporal_preservation(self) -> None:
        """Test that temporal dimension is preserved."""
        for kernel_size in [3, 5, 25, 51]:
            layer = MovingAverage(kernel_size=kernel_size)
            outputs = layer(self.inputs)

            # Temporal dimension should be preserved
            self.assertEqual(outputs.shape[1], self.time_steps)

    def test_different_batch_sizes(self) -> None:
        """Test with different batch sizes."""
        for batch_size in [1, 4, 16, 32]:
            inputs = tf.random.normal((batch_size, self.time_steps, self.n_features))
            outputs = self.layer(inputs)
            self.assertEqual(outputs.shape[0], batch_size)


if __name__ == "__main__":
    unittest.main()
