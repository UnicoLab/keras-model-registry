"""Unit tests for MixingLayer.

Note: TensorFlow is used in tests for validation purposes only.
The actual layer implementation uses only Keras 3 operations.
"""

import unittest

import tensorflow as tf

import keras
from kmr.layers.MixingLayer import MixingLayer


class TestMixingLayer(unittest.TestCase):
    """Test cases for MixingLayer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.n_series = 7
        self.input_size = 96
        self.dropout_rate = 0.1
        self.ff_dim = 64
        self.batch_size = 16

        self.layer = MixingLayer(
            n_series=self.n_series,
            input_size=self.input_size,
            dropout=self.dropout_rate,
            ff_dim=self.ff_dim,
        )

    def test_initialization(self) -> None:
        """Test layer initialization."""
        layer = MixingLayer(n_series=7, input_size=96, dropout=0.1, ff_dim=64)
        self.assertEqual(layer.n_series, 7)
        self.assertEqual(layer.input_size, 96)
        self.assertEqual(layer.dropout_rate, 0.1)
        self.assertEqual(layer.ff_dim, 64)

    def test_invalid_parameters(self) -> None:
        """Test layer initialization with invalid parameters."""
        with self.assertRaises(ValueError):
            MixingLayer(n_series=0, input_size=96, dropout=0.1, ff_dim=64)

        with self.assertRaises(ValueError):
            MixingLayer(n_series=7, input_size=0, dropout=0.1, ff_dim=64)

    def test_output_shape(self) -> None:
        """Test output shape matches input shape."""
        x = tf.random.normal((self.batch_size, self.input_size, self.n_series))
        outputs = self.layer(x)
        self.assertEqual(
            tuple(outputs.shape),
            (self.batch_size, self.input_size, self.n_series),
        )

    def test_different_batch_sizes(self) -> None:
        """Test with different batch sizes."""
        for batch_size in [1, 8, 32]:
            x = tf.random.normal((batch_size, self.input_size, self.n_series))
            outputs = self.layer(x)
            self.assertEqual(
                tuple(outputs.shape),
                (batch_size, self.input_size, self.n_series),
            )

    def test_stacked_mixing(self) -> None:
        """Test that mixing layer applies both temporal and feature mixing."""
        x = tf.random.normal((self.batch_size, self.input_size, self.n_series))
        outputs = self.layer(x, training=False)

        # Outputs should differ from inputs due to mixing
        max_diff = tf.reduce_max(tf.abs(outputs - x))
        self.assertGreater(float(max_diff), 0.0)

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        config = self.layer.get_config()

        self.assertIn("n_series", config)
        self.assertIn("input_size", config)
        self.assertIn("dropout", config)
        self.assertIn("ff_dim", config)

        # Recreate layer from config
        new_layer = MixingLayer.from_config(config)
        self.assertEqual(new_layer.n_series, self.layer.n_series)
        self.assertEqual(new_layer.input_size, self.layer.input_size)
        self.assertEqual(new_layer.dropout_rate, self.layer.dropout_rate)
        self.assertEqual(new_layer.ff_dim, self.layer.ff_dim)


if __name__ == "__main__":
    unittest.main()
