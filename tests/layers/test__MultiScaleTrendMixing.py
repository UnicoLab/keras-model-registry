"""Unit tests for MultiScaleTrendMixing layer.

Note: TensorFlow is used in tests for validation purposes only.
The actual layer implementation uses only Keras 3 operations.
"""

import unittest
import tensorflow as tf
import keras
from kerasfactory.layers.MultiScaleTrendMixing import MultiScaleTrendMixing


class TestMultiScaleTrendMixing(unittest.TestCase):
    """Test cases for MultiScaleTrendMixing layer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.seq_len = 96
        self.down_sampling_window = 2
        self.down_sampling_layers = 2
        self.batch_size = 16

        self.layer = MultiScaleTrendMixing(
            seq_len=self.seq_len,
            down_sampling_window=self.down_sampling_window,
            down_sampling_layers=self.down_sampling_layers,
        )

    def test_initialization(self) -> None:
        """Test layer initialization."""
        layer = MultiScaleTrendMixing(
            seq_len=96,
            down_sampling_window=2,
            down_sampling_layers=2,
        )
        self.assertEqual(layer.seq_len, 96)
        self.assertEqual(layer.down_sampling_window, 2)
        self.assertEqual(layer.down_sampling_layers, 2)

    def test_invalid_parameters(self) -> None:
        """Test layer initialization with invalid parameters."""
        with self.assertRaises(ValueError):
            MultiScaleTrendMixing(
                seq_len=0,
                down_sampling_window=2,
                down_sampling_layers=2,
            )

        with self.assertRaises(ValueError):
            MultiScaleTrendMixing(
                seq_len=96,
                down_sampling_window=0,
                down_sampling_layers=2,
            )

    def test_output_type(self) -> None:
        """Test that output is a list."""
        trend_inputs = [tf.random.normal((self.batch_size, self.seq_len, 8))]
        outputs = self.layer(trend_inputs)
        self.assertIsInstance(outputs, list)

    def test_output_shapes(self) -> None:
        """Test output shapes for different upsampling layers."""
        trend_inputs = [tf.random.normal((self.batch_size, self.seq_len, 8))]
        outputs = self.layer(trend_inputs)

        # Should return list of outputs
        self.assertGreater(len(outputs), 0)
        for output in outputs:
            self.assertEqual(len(output.shape), 3)  # (batch, time, channels)

    def test_different_batch_sizes(self) -> None:
        """Test with different batch sizes."""
        for batch_size in [1, 8, 32]:
            trend_inputs = [tf.random.normal((batch_size, self.seq_len, 8))]
            outputs = self.layer(trend_inputs)
            self.assertIsInstance(outputs, list)
            self.assertGreater(len(outputs), 0)

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        config = self.layer.get_config()

        self.assertIn("seq_len", config)
        self.assertIn("down_sampling_window", config)
        self.assertIn("down_sampling_layers", config)

        # Recreate layer from config
        new_layer = MultiScaleTrendMixing.from_config(config)
        self.assertEqual(new_layer.seq_len, self.layer.seq_len)
        self.assertEqual(
            new_layer.down_sampling_window,
            self.layer.down_sampling_window,
        )
        self.assertEqual(
            new_layer.down_sampling_layers,
            self.layer.down_sampling_layers,
        )

    def test_deterministic_output(self) -> None:
        """Test that outputs are deterministic with same seed."""
        trend_inputs = [tf.random.normal((self.batch_size, self.seq_len, 8))]

        outputs1 = self.layer(trend_inputs, training=False)
        outputs2 = self.layer(trend_inputs, training=False)

        # Should be identical during inference
        for o1, o2 in zip(outputs1, outputs2):
            tf.debugging.assert_near(o1, o2)


if __name__ == "__main__":
    unittest.main()
