"""Unit tests for TokenEmbedding layer.

Note: TensorFlow is used in tests for validation purposes only.
The actual layer implementation uses only Keras 3 operations.
"""

import unittest
import tensorflow as tf  # Used for testing only
from keras import layers, Model
from kmr.layers.TokenEmbedding import TokenEmbedding


class TestTokenEmbedding(unittest.TestCase):
    """Test cases for TokenEmbedding layer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.batch_size = 16
        self.time_steps = 100
        self.c_in = 1
        self.d_model = 64
        self.layer = TokenEmbedding(c_in=self.c_in, d_model=self.d_model)
        self.inputs = tf.random.normal((self.batch_size, self.time_steps, self.c_in))

    def test_initialization(self) -> None:
        """Test layer initialization."""
        layer = TokenEmbedding(c_in=1, d_model=64)
        self.assertEqual(layer.c_in, 1)
        self.assertEqual(layer.d_model, 64)

    def test_invalid_initialization(self) -> None:
        """Test layer initialization with invalid parameters."""
        with self.assertRaises(ValueError):
            TokenEmbedding(c_in=0, d_model=64)

        with self.assertRaises(ValueError):
            TokenEmbedding(c_in=1, d_model=0)

        with self.assertRaises(ValueError):
            TokenEmbedding(c_in=-1, d_model=64)

    def test_output_shape(self) -> None:
        """Test output shape of the layer."""
        outputs = self.layer(self.inputs)
        expected_shape = (self.batch_size, self.time_steps, self.d_model)
        self.assertEqual(tuple(outputs.shape), expected_shape)

    def test_call(self) -> None:
        """Test forward pass of the layer."""
        outputs = self.layer(self.inputs)
        self.assertIsNotNone(outputs)
        self.assertEqual(outputs.dtype, self.inputs.dtype)

    def test_multivariate_input(self) -> None:
        """Test with multivariate input."""
        c_in = 8
        layer = TokenEmbedding(c_in=c_in, d_model=64)
        inputs = tf.random.normal((16, 100, c_in))
        outputs = layer(inputs)

        self.assertEqual(outputs.shape[0], 16)
        self.assertEqual(outputs.shape[1], 100)
        self.assertEqual(outputs.shape[2], 64)

    def test_conv1d_operation(self) -> None:
        """Test that convolution is applied correctly."""
        # Create simple input to verify convolution happens
        layer = TokenEmbedding(c_in=1, d_model=32)
        layer.build((None, 100, 1))

        # Check conv layer was created
        self.assertIsNotNone(layer.conv)
        self.assertEqual(layer.conv.filters, 32)

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        config = self.layer.get_config()
        self.assertIn("c_in", config)
        self.assertIn("d_model", config)

        new_layer = TokenEmbedding.from_config(config)
        self.assertEqual(new_layer.c_in, self.layer.c_in)
        self.assertEqual(new_layer.d_model, self.layer.d_model)

    def test_model_integration(self) -> None:
        """Test layer integration in a model."""
        inputs = layers.Input(shape=(self.time_steps, self.c_in))
        x = TokenEmbedding(c_in=self.c_in, d_model=self.d_model)(inputs)
        outputs = layers.Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer="adam", loss="mse")

        x_data = tf.random.normal((10, self.time_steps, self.c_in))
        y_data = tf.random.normal((10, 1))

        history = model.fit(x_data, y_data, epochs=1, verbose=0)
        self.assertTrue(history.history["loss"][0] > 0)

    def test_different_input_shapes(self) -> None:
        """Test with different input shapes."""
        test_shapes = [
            (8, 50, 1),
            (32, 200, 1),
            (1, 100, 1),
            (16, 100, 4),
        ]

        for shape in test_shapes:
            layer = TokenEmbedding(c_in=shape[2], d_model=64)
            inputs = tf.random.normal(shape)
            outputs = layer(inputs)

            self.assertEqual(outputs.shape[0], shape[0])
            self.assertEqual(outputs.shape[1], shape[1])
            self.assertEqual(outputs.shape[2], 64)

    def test_embedding_dimension_flexibility(self) -> None:
        """Test with different embedding dimensions."""
        for d_model in [32, 64, 128]:
            layer = TokenEmbedding(c_in=1, d_model=d_model)
            outputs = layer(self.inputs)
            self.assertEqual(outputs.shape[2], d_model)


if __name__ == "__main__":
    unittest.main()
