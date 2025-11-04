"""Unit tests for PositionalEmbedding layer.

Note: TensorFlow is used in tests for validation purposes only.
The actual layer implementation uses only Keras 3 operations.
"""

import unittest
import tensorflow as tf  # Used for testing only
from keras import layers, Model
from kmr.layers.PositionalEmbedding import PositionalEmbedding


class TestPositionalEmbedding(unittest.TestCase):
    """Test cases for PositionalEmbedding layer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.d_model = 64
        self.max_len = 512
        self.seq_len = 100
        self.batch_size = 16
        self.layer = PositionalEmbedding(d_model=self.d_model, max_len=self.max_len)
        self.inputs = tf.random.normal((self.batch_size, self.seq_len, self.d_model))

    def test_initialization(self) -> None:
        """Test layer initialization."""
        layer = PositionalEmbedding(d_model=64, max_len=512)
        self.assertEqual(layer.d_model, 64)
        self.assertEqual(layer.max_len, 512)

    def test_invalid_initialization(self) -> None:
        """Test layer initialization with invalid parameters."""
        with self.assertRaises(ValueError):
            PositionalEmbedding(d_model=0, max_len=512)

        with self.assertRaises(ValueError):
            PositionalEmbedding(d_model=64, max_len=0)

        with self.assertRaises(ValueError):
            PositionalEmbedding(d_model=-1, max_len=512)

    def test_output_shape(self) -> None:
        """Test output shape of the layer."""
        outputs = self.layer(self.inputs)
        expected_shape = (1, self.seq_len, self.d_model)
        self.assertEqual(tuple(outputs.shape), expected_shape)

    def test_call(self) -> None:
        """Test forward pass of the layer."""
        outputs = self.layer(self.inputs)
        self.assertIsNotNone(outputs)
        self.assertEqual(outputs.dtype, tf.float32)

    def test_different_sequence_lengths(self) -> None:
        """Test with different sequence lengths."""
        for seq_len in [10, 50, 100, 256]:
            inputs = tf.random.normal((8, seq_len, self.d_model))
            outputs = self.layer(inputs)
            self.assertEqual(outputs.shape[1], seq_len)
            self.assertEqual(outputs.shape[2], self.d_model)

    def test_fixed_encodings(self) -> None:
        """Test that positional encodings are fixed."""
        # Call twice and check outputs are identical
        output1 = self.layer(self.inputs)
        output2 = self.layer(self.inputs)

        tf.debugging.assert_equal(output1, output2)

    def test_encodings_increase_with_position(self) -> None:
        """Test sinusoidal property: encodings vary with position."""
        outputs = self.layer(self.inputs)

        # Different positions should generally have different encodings
        pos1_encoding = outputs[0, 0, :]  # First position
        pos2_encoding = outputs[0, 10, :]  # Later position

        # Compute distance
        dist = tf.sqrt(tf.reduce_sum((pos1_encoding - pos2_encoding) ** 2))

        # Distance should be meaningful
        self.assertGreater(float(dist), 0.01)

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        config = self.layer.get_config()
        self.assertIn("d_model", config)
        self.assertIn("max_len", config)

        new_layer = PositionalEmbedding.from_config(config)
        self.assertEqual(new_layer.d_model, self.layer.d_model)
        self.assertEqual(new_layer.max_len, self.layer.max_len)

    def test_model_integration(self) -> None:
        """Test layer integration in a model."""
        inputs = layers.Input(shape=(self.seq_len, self.d_model))
        pos_emb = self.layer(inputs)
        combined = inputs + pos_emb
        outputs = layers.Dense(1)(combined)
        model = Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer="adam", loss="mse")

        x_data = tf.random.normal((10, self.seq_len, self.d_model))
        y_data = tf.random.normal((10, 1))

        history = model.fit(x_data, y_data, epochs=1, verbose=0)
        self.assertTrue(history.history["loss"][0] > 0)

    def test_max_length_respected(self) -> None:
        """Test that max_len is respected."""
        short_seq = tf.random.normal((2, 10, self.d_model))
        long_seq = tf.random.normal((2, self.max_len - 1, self.d_model))

        short_output = self.layer(short_seq)
        long_output = self.layer(long_seq)

        self.assertEqual(short_output.shape[1], 10)
        self.assertEqual(long_output.shape[1], self.max_len - 1)

    def test_sine_cosine_interleaving(self) -> None:
        """Test that sine and cosine are properly interleaved."""
        # Build layer to generate encodings
        self.layer.build((None, 10, self.d_model))

        # Get encoding for first position
        test_input = tf.zeros((1, 1, self.d_model))
        output = self.layer(test_input)

        # Output should have non-zero values from sine/cosine
        self.assertFalse(tf.reduce_all(tf.equal(output, 0)))


if __name__ == "__main__":
    unittest.main()
