"""Unit tests for the CastToFloat32Layer.

Note: TensorFlow is used in tests for validation purposes only.
The actual layer implementation uses only Keras 3 operations.
"""

import unittest
import numpy as np
import tensorflow as tf  # Used for testing only
from keras import Model, layers
from kerasfactory.layers.CastToFloat32Layer import CastToFloat32Layer


class TestCastToFloat32Layer(unittest.TestCase):
    """Test cases for the CastToFloat32Layer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Using TensorFlow for test data generation only
        self.int_data = tf.constant([1, 2, 3], dtype=tf.int32)
        self.float64_data = tf.constant([1.1, 2.2, 3.3], dtype=tf.float64)
        self.bool_data = tf.constant([True, False, True])
        tf.random.set_seed(42)  # For reproducibility

    def test_initialization(self) -> None:
        """Test layer initialization."""
        layer = CastToFloat32Layer()
        self.assertIsNotNone(layer)

    def test_cast_int_to_float32(self) -> None:
        """Test casting integer data to float32."""
        layer = CastToFloat32Layer()
        output = layer(self.int_data)

        # Check output dtype
        self.assertEqual(output.dtype, "float32")

        # Check values are preserved
        np.testing.assert_array_equal(
            output.numpy(),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
        )

    def test_cast_float64_to_float32(self) -> None:
        """Test casting float64 data to float32."""
        layer = CastToFloat32Layer()
        output = layer(self.float64_data)

        # Check output dtype
        self.assertEqual(output.dtype, "float32")

        # Check values are preserved (with float32 precision)
        np.testing.assert_array_almost_equal(
            output.numpy(),
            np.array([1.1, 2.2, 3.3], dtype=np.float32),
            decimal=5,
        )

    def test_cast_bool_to_float32(self) -> None:
        """Test casting boolean data to float32."""
        layer = CastToFloat32Layer()
        output = layer(self.bool_data)

        # Check output dtype
        self.assertEqual(output.dtype, "float32")

        # Check values are converted correctly (True -> 1.0, False -> 0.0)
        np.testing.assert_array_equal(
            output.numpy(),
            np.array([1.0, 0.0, 1.0], dtype=np.float32),
        )

    def test_output_shape(self) -> None:
        """Test that output shape matches input shape."""
        layer = CastToFloat32Layer()

        # Test with various shapes
        test_shapes = [(3,), (2, 3), (2, 3, 4)]

        for shape in test_shapes:
            # Create tensor with the given shape
            x = tf.ones(shape, dtype=tf.int32)
            output = layer(x)

            # Check that output shape matches input shape
            self.assertEqual(output.shape, x.shape)

    def test_compute_output_shape(self) -> None:
        """Test the compute_output_shape method."""
        layer = CastToFloat32Layer()

        # Test with various shapes
        test_shapes = [(3,), (2, 3), (2, 3, 4), (None, 5)]

        for shape in test_shapes:
            output_shape = layer.compute_output_shape(shape)
            self.assertEqual(output_shape, shape)

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        original_layer = CastToFloat32Layer()
        config = original_layer.get_config()

        # Create new layer from config
        restored_layer = CastToFloat32Layer.from_config(config)

        # Check that outputs match
        original_output = original_layer(self.int_data)
        restored_output = restored_layer(self.int_data)

        np.testing.assert_array_equal(original_output.numpy(), restored_output.numpy())

    def test_integration(self) -> None:
        """Test integration with a simple model."""
        # Create a simple model with the casting layer
        inputs = layers.Input(shape=(3,), dtype="int32")
        x = CastToFloat32Layer()(inputs)
        outputs = layers.Dense(1)(x)

        model = Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(optimizer="adam", loss="mse")

        # Generate some dummy data
        x_data = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
        y_data = tf.constant([[1.0], [2.0]])

        # Fit the model for one step to ensure everything works
        model.fit(x_data, y_data, epochs=1, verbose=0)

        # Make a prediction to ensure the pipeline works
        pred = model.predict(tf.constant([[7, 8, 9]], dtype=tf.int32))
        self.assertEqual(pred.shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
