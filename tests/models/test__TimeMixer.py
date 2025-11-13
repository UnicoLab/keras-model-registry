"""Unit tests for TimeMixer model.

Note: TensorFlow is used in tests for validation purposes only.
The actual model implementation uses only Keras 3 operations.
"""

import unittest
import tensorflow as tf  # Used for testing only
from keras import Model
from kerasfactory.models.TimeMixer import TimeMixer


class TestTimeMixer(unittest.TestCase):
    """Test cases for TimeMixer model."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.seq_len = 96
        self.pred_len = 12
        self.n_features = 7
        self.batch_size = 32

        self.model = TimeMixer(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            n_features=self.n_features,
            d_model=32,
            d_ff=32,
            e_layers=2,
            dropout=0.1,
            decomp_method="moving_avg",
            moving_avg=25,
            top_k=5,
            channel_independence=0,
            down_sampling_layers=1,
            down_sampling_window=2,
        )

        self.inputs = tf.random.normal((self.batch_size, self.seq_len, self.n_features))
        self.targets = tf.random.normal(
            (self.batch_size, self.pred_len, self.n_features),
        )

    def test_initialization(self) -> None:
        """Test model initialization."""
        self.assertEqual(self.model.seq_len, self.seq_len)
        self.assertEqual(self.model.pred_len, self.pred_len)
        self.assertEqual(self.model.n_features, self.n_features)

    def test_invalid_initialization(self) -> None:
        """Test model initialization with invalid parameters."""
        with self.assertRaises(ValueError):
            TimeMixer(seq_len=100, pred_len=12, n_features=7, decomp_method="invalid")

        with self.assertRaises(ValueError):
            TimeMixer(seq_len=100, pred_len=12, n_features=7, channel_independence=2)

    def test_forward_pass(self) -> None:
        """Test forward pass of the model."""
        outputs = self.model(self.inputs)

        # Check output shape
        self.assertEqual(outputs.shape[0], self.batch_size)
        self.assertEqual(outputs.shape[1], self.pred_len)
        self.assertEqual(outputs.shape[2], self.n_features)

    def test_compilation_and_training(self) -> None:
        """Test model compilation and training."""
        self.model.compile(optimizer="adam", loss="mse")

        history = self.model.fit(
            self.inputs,
            self.targets,
            epochs=1,
            verbose=0,
            batch_size=16,
        )

        self.assertTrue(history.history["loss"][0] > 0)

    def test_prediction(self) -> None:
        """Test model prediction."""
        self.model.compile(optimizer="adam", loss="mse")
        predictions = self.model.predict(self.inputs[:4], verbose=0)

        self.assertEqual(predictions.shape[0], 4)
        self.assertEqual(predictions.shape[1], self.pred_len)
        self.assertEqual(predictions.shape[2], self.n_features)

    def test_different_batch_sizes(self) -> None:
        """Test with different batch sizes."""
        for batch_size in [1, 8, 16, 32]:
            inputs = tf.random.normal((batch_size, self.seq_len, self.n_features))
            outputs = self.model(inputs)

            self.assertEqual(outputs.shape[0], batch_size)
            self.assertEqual(outputs.shape[1], self.pred_len)
            self.assertEqual(outputs.shape[2], self.n_features)

    def test_dft_decomposition_method(self) -> None:
        """Test model with DFT decomposition."""
        model = TimeMixer(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            n_features=self.n_features,
            decomp_method="dft_decomp",
            top_k=5,
        )

        outputs = model(self.inputs)

        self.assertEqual(outputs.shape[0], self.batch_size)
        self.assertEqual(outputs.shape[1], self.pred_len)
        self.assertEqual(outputs.shape[2], self.n_features)

    def test_channel_independence(self) -> None:
        """Test model with channel independence."""
        model_independent = TimeMixer(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            n_features=self.n_features,
            channel_independence=1,
        )

        outputs = model_independent(self.inputs)

        self.assertEqual(outputs.shape[0], self.batch_size)
        self.assertEqual(outputs.shape[1], self.pred_len)
        self.assertEqual(outputs.shape[2], self.n_features)

    def test_different_architectures(self) -> None:
        """Test with different model architectures."""
        configs = [
            {"d_model": 16, "e_layers": 1},
            {"d_model": 64, "e_layers": 4},
            {"d_model": 32, "e_layers": 2, "down_sampling_layers": 2},
        ]

        for config in configs:
            model = TimeMixer(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                n_features=self.n_features,
                **config,
            )

            outputs = model(self.inputs)
            self.assertEqual(
                tuple(outputs.shape),
                (self.batch_size, self.pred_len, self.n_features),
            )

    def test_serialization(self) -> None:
        """Test model serialization and deserialization."""
        config = self.model.get_config()

        self.assertIn("seq_len", config)
        self.assertIn("pred_len", config)
        self.assertIn("n_features", config)
        self.assertIn("decomp_method", config)

        new_model = TimeMixer.from_config(config)
        self.assertEqual(new_model.seq_len, self.model.seq_len)
        self.assertEqual(new_model.pred_len, self.model.pred_len)
        self.assertEqual(new_model.n_features, self.model.n_features)

    def test_decoder_input_multiplier_validation(self) -> None:
        """Test decoder input size multiplier validation."""
        with self.assertRaises(ValueError):
            TimeMixer(
                seq_len=100,
                pred_len=12,
                n_features=7,
                decoder_input_size_multiplier=1.5,  # > 1
            )

        with self.assertRaises(ValueError):
            TimeMixer(
                seq_len=100,
                pred_len=12,
                n_features=7,
                decoder_input_size_multiplier=0,  # <= 0
            )

    def test_multivariate_features(self) -> None:
        """Test with different numbers of features."""
        for n_features in [1, 5, 10, 20]:
            model = TimeMixer(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                n_features=n_features,
            )

            inputs = tf.random.normal((self.batch_size, self.seq_len, n_features))
            outputs = model(inputs)

            self.assertEqual(outputs.shape[2], n_features)

    def test_model_with_temporal_features(self) -> None:
        """Test model with optional temporal features."""
        x = tf.random.normal((self.batch_size, self.seq_len, self.n_features))
        # Temporal marks: [month(0-12), day(0-31), weekday(0-6), hour(0-23), minute(0-59)]
        x_mark = tf.stack(
            [
                tf.random.uniform(
                    (self.batch_size, self.seq_len),
                    minval=0,
                    maxval=13,
                    dtype=tf.int32,
                ),  # month
                tf.random.uniform(
                    (self.batch_size, self.seq_len),
                    minval=0,
                    maxval=32,
                    dtype=tf.int32,
                ),  # day
                tf.random.uniform(
                    (self.batch_size, self.seq_len),
                    minval=0,
                    maxval=7,
                    dtype=tf.int32,
                ),  # weekday
                tf.random.uniform(
                    (self.batch_size, self.seq_len),
                    minval=0,
                    maxval=24,
                    dtype=tf.int32,
                ),  # hour
                tf.random.uniform(
                    (self.batch_size, self.seq_len),
                    minval=0,
                    maxval=60,
                    dtype=tf.int32,
                ),  # minute
            ],
            axis=-1,
        )

        outputs = self.model([x, x_mark])

        self.assertEqual(outputs.shape[0], self.batch_size)
        self.assertEqual(outputs.shape[1], self.pred_len)
        self.assertEqual(outputs.shape[2], self.n_features)


if __name__ == "__main__":
    unittest.main()
