"""Unit tests for PastDecomposableMixing layer.

Note: TensorFlow is used in tests for validation purposes only.
The actual layer implementation uses only Keras 3 operations.
"""

import unittest
import tensorflow as tf
import keras
from kmr.layers.PastDecomposableMixing import PastDecomposableMixing


class TestPastDecomposableMixing(unittest.TestCase):
    """Test cases for PastDecomposableMixing layer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.seq_len = 96
        self.pred_len = 12
        self.down_sampling_window = 2
        self.down_sampling_layers = 1
        self.d_model = 32
        self.dropout = 0.1
        self.channel_independence = 0
        self.decomp_method = "moving_avg"
        self.d_ff = 128
        self.moving_avg = 25
        self.top_k = 5
        self.batch_size = 16
        self.n_features = 7

        self.layer = PastDecomposableMixing(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            down_sampling_window=self.down_sampling_window,
            down_sampling_layers=self.down_sampling_layers,
            d_model=self.d_model,
            dropout=self.dropout,
            channel_independence=self.channel_independence,
            decomp_method=self.decomp_method,
            d_ff=self.d_ff,
            moving_avg=self.moving_avg,
            top_k=self.top_k,
        )

    def test_initialization(self) -> None:
        """Test layer initialization."""
        layer = PastDecomposableMixing(
            seq_len=96,
            pred_len=12,
            down_sampling_window=2,
            down_sampling_layers=1,
            d_model=32,
            dropout=0.1,
            channel_independence=0,
            decomp_method="moving_avg",
            d_ff=128,
            moving_avg=25,
            top_k=5,
        )
        self.assertEqual(layer.seq_len, 96)
        self.assertEqual(layer.pred_len, 12)
        self.assertEqual(layer.d_model, 32)

    def test_invalid_parameters(self) -> None:
        """Test layer initialization with invalid parameters."""
        # Test with invalid decomp_method
        with self.assertRaises(ValueError):
            PastDecomposableMixing(
                seq_len=96,
                pred_len=12,
                down_sampling_window=2,
                down_sampling_layers=1,
                d_model=32,
                dropout=0.1,
                channel_independence=0,
                decomp_method="invalid_method",
                d_ff=128,
                moving_avg=25,
                top_k=5,
            )

    def test_output_shape_moving_avg(self) -> None:
        """Test output shape with moving average decomposition."""
        x = tf.random.normal((self.batch_size, self.seq_len, self.n_features))
        outputs = self.layer([x])

        # Outputs is a list
        self.assertIsInstance(outputs, list)
        self.assertGreater(len(outputs), 0)

        # First output should have shape (batch, seq_len, d_model)
        output = outputs[0]
        expected_shape = (self.batch_size, self.seq_len, self.d_model)
        self.assertEqual(tuple(output.shape), expected_shape)

    def test_output_shape_dft(self) -> None:
        """Test output shape with DFT decomposition."""
        layer = PastDecomposableMixing(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            down_sampling_window=self.down_sampling_window,
            down_sampling_layers=self.down_sampling_layers,
            d_model=self.d_model,
            dropout=self.dropout,
            channel_independence=self.channel_independence,
            decomp_method="dft_decomp",
            d_ff=self.d_ff,
            moving_avg=self.moving_avg,
            top_k=self.top_k,
        )

        x = tf.random.normal((self.batch_size, self.seq_len, self.n_features))
        outputs = layer([x])

        # Outputs is a list
        self.assertIsInstance(outputs, list)
        self.assertGreater(len(outputs), 0)

        # First output should have shape (batch, seq_len, d_model)
        output = outputs[0]
        expected_shape = (self.batch_size, self.seq_len, self.d_model)
        self.assertEqual(tuple(output.shape), expected_shape)

    def test_different_batch_sizes(self) -> None:
        """Test with different batch sizes."""
        for batch_size in [1, 8, 32]:
            x = tf.random.normal((batch_size, self.seq_len, self.n_features))
            outputs = self.layer([x])

            self.assertIsInstance(outputs, list)
            self.assertGreater(len(outputs), 0)

            output = outputs[0]
            expected_shape = (batch_size, self.seq_len, self.d_model)
            self.assertEqual(tuple(output.shape), expected_shape)

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        config = self.layer.get_config()

        # Check that all config keys are present
        required_keys = [
            "seq_len",
            "pred_len",
            "down_sampling_window",
            "down_sampling_layers",
            "d_model",
            "dropout",
            "channel_independence",
            "decomp_method",
            "d_ff",
            "moving_avg",
            "top_k",
        ]
        for key in required_keys:
            self.assertIn(key, config, f"Missing config key: {key}")

        # Recreate layer from config
        new_layer = PastDecomposableMixing.from_config(config)
        self.assertEqual(new_layer.seq_len, self.layer.seq_len)
        self.assertEqual(new_layer.pred_len, self.layer.pred_len)
        self.assertEqual(new_layer.d_model, self.layer.d_model)

    def test_deterministic_output(self) -> None:
        """Test that outputs are deterministic with same seed."""
        x = tf.random.normal((self.batch_size, self.seq_len, self.n_features))

        outputs1 = self.layer([x], training=False)
        outputs2 = self.layer([x], training=False)

        # Should be identical during inference
        self.assertEqual(len(outputs1), len(outputs2))
        for o1, o2 in zip(outputs1, outputs2):
            tf.debugging.assert_near(o1, o2)


if __name__ == "__main__":
    unittest.main()
