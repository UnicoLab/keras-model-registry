"""Unit tests for FeatureMixing layer.

Note: TensorFlow is used in tests for validation purposes only.
The actual layer implementation uses only Keras 3 operations.
"""

import unittest

import tensorflow as tf

import keras
from kerasfactory.layers.FeatureMixing import FeatureMixing


class TestFeatureMixing(unittest.TestCase):
    """Test cases for FeatureMixing layer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.n_series = 7
        self.input_size = 96
        self.dropout_rate = 0.1
        self.ff_dim = 64
        self.batch_size = 16

        self.layer = FeatureMixing(
            n_series=self.n_series,
            input_size=self.input_size,
            dropout=self.dropout_rate,
            ff_dim=self.ff_dim,
        )

    def test_initialization(self) -> None:
        """Test layer initialization."""
        layer = FeatureMixing(n_series=7, input_size=96, dropout=0.1, ff_dim=64)
        self.assertEqual(layer.n_series, 7)
        self.assertEqual(layer.input_size, 96)
        self.assertEqual(layer.dropout_rate, 0.1)
        self.assertEqual(layer.ff_dim, 64)

    def test_invalid_parameters(self) -> None:
        """Test layer initialization with invalid parameters."""
        with self.assertRaises(ValueError):
            FeatureMixing(n_series=0, input_size=96, dropout=0.1, ff_dim=64)

        with self.assertRaises(ValueError):
            FeatureMixing(n_series=7, input_size=0, dropout=0.1, ff_dim=64)

        with self.assertRaises(ValueError):
            FeatureMixing(n_series=7, input_size=96, dropout=-0.1, ff_dim=64)

        with self.assertRaises(ValueError):
            FeatureMixing(n_series=7, input_size=96, dropout=0.1, ff_dim=0)

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

    def test_different_ff_dimensions(self) -> None:
        """Test with different feed-forward dimensions."""
        for ff_dim in [32, 64, 128]:
            layer = FeatureMixing(
                n_series=self.n_series,
                input_size=self.input_size,
                dropout=self.dropout_rate,
                ff_dim=ff_dim,
            )
            x = tf.random.normal((self.batch_size, self.input_size, self.n_series))
            outputs = layer(x)
            self.assertEqual(
                tuple(outputs.shape),
                (self.batch_size, self.input_size, self.n_series),
            )

    def test_residual_connection(self) -> None:
        """Test that output differs from input due to transformations."""
        x = tf.random.normal((self.batch_size, self.input_size, self.n_series))
        outputs = self.layer(x, training=False)

        # Outputs should differ from inputs (not identical)
        max_diff = tf.reduce_max(tf.abs(outputs - x))
        self.assertGreater(float(max_diff), 0.0)

    def test_dropout_effect(self) -> None:
        """Test that dropout affects outputs during training."""
        x = tf.random.normal((self.batch_size, self.input_size, self.n_series))

        # Training mode with dropout
        outputs1 = self.layer(x, training=True)
        outputs2 = self.layer(x, training=True)

        # Outputs should be different due to dropout
        diff = tf.reduce_mean(tf.abs(outputs1 - outputs2))
        self.assertGreater(float(diff), 0.0)

    def test_inference_deterministic(self) -> None:
        """Test that layer is deterministic in inference mode."""
        x = tf.random.normal((self.batch_size, self.input_size, self.n_series))

        outputs1 = self.layer(x, training=False)
        outputs2 = self.layer(x, training=False)

        # Should be identical without dropout
        tf.debugging.assert_near(outputs1, outputs2)

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        config = self.layer.get_config()

        self.assertIn("n_series", config)
        self.assertIn("input_size", config)
        self.assertIn("dropout", config)
        self.assertIn("ff_dim", config)

        # Recreate layer from config
        new_layer = FeatureMixing.from_config(config)
        self.assertEqual(new_layer.n_series, self.layer.n_series)
        self.assertEqual(new_layer.input_size, self.layer.input_size)
        self.assertEqual(new_layer.dropout_rate, self.layer.dropout_rate)
        self.assertEqual(new_layer.ff_dim, self.layer.ff_dim)


if __name__ == "__main__":
    unittest.main()
