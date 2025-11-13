"""Unit tests for TSMixer model.

Note: TensorFlow is used in tests for validation purposes only.
The actual model implementation uses only Keras 3 operations.
"""

import unittest

import tensorflow as tf

import keras
from kerasfactory.models.TSMixer import TSMixer


class TestTSMixer(unittest.TestCase):
    """Test cases for TSMixer model."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.seq_len = 96
        self.pred_len = 12
        self.n_features = 7
        self.n_blocks = 2
        self.ff_dim = 64
        self.dropout = 0.1
        self.batch_size = 16

        self.model = TSMixer(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            n_features=self.n_features,
            n_blocks=self.n_blocks,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            use_norm=True,
        )
        self.model.compile(optimizer="adam", loss="mse")

    def test_initialization(self) -> None:
        """Test model initialization."""
        model = TSMixer(
            seq_len=96,
            pred_len=12,
            n_features=7,
            n_blocks=2,
            ff_dim=64,
            dropout=0.1,
        )
        self.assertEqual(model.seq_len, 96)
        self.assertEqual(model.pred_len, 12)
        self.assertEqual(model.n_features, 7)
        self.assertEqual(model.n_blocks, 2)
        self.assertEqual(model.ff_dim, 64)

    def test_invalid_parameters(self) -> None:
        """Test model initialization with invalid parameters."""
        with self.assertRaises(ValueError):
            TSMixer(seq_len=0, pred_len=12, n_features=7)

        with self.assertRaises(ValueError):
            TSMixer(seq_len=96, pred_len=0, n_features=7)

        with self.assertRaises(ValueError):
            TSMixer(seq_len=96, pred_len=12, n_features=0)

    def test_output_shape(self) -> None:
        """Test output shape."""
        x = tf.random.normal((self.batch_size, self.seq_len, self.n_features))
        outputs = self.model(x, training=False)

        expected_shape = (self.batch_size, self.pred_len, self.n_features)
        self.assertEqual(tuple(outputs.shape), expected_shape)

    def test_different_batch_sizes(self) -> None:
        """Test with different batch sizes."""
        for batch_size in [1, 8, 32]:
            x = tf.random.normal((batch_size, self.seq_len, self.n_features))
            outputs = self.model(x, training=False)
            expected_shape = (batch_size, self.pred_len, self.n_features)
            self.assertEqual(tuple(outputs.shape), expected_shape)

    def test_with_and_without_normalization(self) -> None:
        """Test model with and without instance normalization."""
        x = tf.random.normal((self.batch_size, self.seq_len, self.n_features))

        # Model with normalization
        model_with_norm = TSMixer(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            n_features=self.n_features,
            use_norm=True,
        )
        model_with_norm.compile(optimizer="adam", loss="mse")
        out_with_norm = model_with_norm(x, training=False)

        # Model without normalization
        model_without_norm = TSMixer(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            n_features=self.n_features,
            use_norm=False,
        )
        model_without_norm.compile(optimizer="adam", loss="mse")
        out_without_norm = model_without_norm(x, training=False)

        # Both should produce valid outputs
        self.assertEqual(
            tuple(out_with_norm.shape),
            (self.batch_size, self.pred_len, self.n_features),
        )
        self.assertEqual(
            tuple(out_without_norm.shape),
            (self.batch_size, self.pred_len, self.n_features),
        )

    def test_different_block_counts(self) -> None:
        """Test with different numbers of mixing blocks."""
        x = tf.random.normal((self.batch_size, self.seq_len, self.n_features))

        for n_blocks in [1, 2, 4]:
            model = TSMixer(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                n_features=self.n_features,
                n_blocks=n_blocks,
            )
            model.compile(optimizer="adam", loss="mse")
            outputs = model(x, training=False)

            expected_shape = (self.batch_size, self.pred_len, self.n_features)
            self.assertEqual(tuple(outputs.shape), expected_shape)

    def test_inference_deterministic(self) -> None:
        """Test that model outputs are deterministic in inference mode."""
        x = tf.random.normal((self.batch_size, self.seq_len, self.n_features))

        outputs1 = self.model(x, training=False)
        outputs2 = self.model(x, training=False)

        tf.debugging.assert_near(outputs1, outputs2)

    def test_training_vs_inference(self) -> None:
        """Test that training and inference produce different outputs due to dropout."""
        x = tf.random.normal((self.batch_size, self.seq_len, self.n_features))

        # Create a model with high dropout to ensure visible effect
        model_high_dropout = TSMixer(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            n_features=self.n_features,
            dropout=0.5,
        )
        model_high_dropout.compile(optimizer="adam", loss="mse")

        outputs_train1 = model_high_dropout(x, training=True)
        outputs_train2 = model_high_dropout(x, training=True)

        # Training outputs should differ due to dropout
        diff = tf.reduce_mean(tf.abs(outputs_train1 - outputs_train2))
        self.assertGreater(float(diff), 0.0)

    def test_serialization(self) -> None:
        """Test model serialization and deserialization."""
        config = self.model.get_config()

        required_keys = [
            "seq_len",
            "pred_len",
            "n_features",
            "n_blocks",
            "ff_dim",
            "dropout",
            "use_norm",
        ]
        for key in required_keys:
            self.assertIn(key, config, f"Missing config key: {key}")

        # Recreate model from config
        new_model = TSMixer.from_config(config)
        self.assertEqual(new_model.seq_len, self.model.seq_len)
        self.assertEqual(new_model.pred_len, self.model.pred_len)
        self.assertEqual(new_model.n_features, self.model.n_features)
        self.assertEqual(new_model.n_blocks, self.model.n_blocks)


if __name__ == "__main__":
    unittest.main()
