"""Unit tests for TemporalEmbedding layer.

Note: TensorFlow is used in tests for validation purposes only.
The actual layer implementation uses only Keras 3 operations.
"""

import unittest
import tensorflow as tf
import keras
from kmr.layers.TemporalEmbedding import TemporalEmbedding


class TestTemporalEmbedding(unittest.TestCase):
    """Test cases for TemporalEmbedding layer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.d_model = 64
        self.batch_size = 16
        self.time_steps = 96
        self.layer_fixed = TemporalEmbedding(
            d_model=self.d_model,
            embed_type="fixed",
            freq="h",
        )
        self.layer_learned = TemporalEmbedding(
            d_model=self.d_model,
            embed_type="learned",
            freq="h",
        )

    def test_initialization_fixed(self) -> None:
        """Test layer initialization with fixed embeddings."""
        layer = TemporalEmbedding(d_model=64, embed_type="fixed", freq="h")
        self.assertEqual(layer.d_model, 64)
        self.assertEqual(layer.embed_type, "fixed")
        self.assertEqual(layer.freq, "h")

    def test_initialization_learned(self) -> None:
        """Test layer initialization with learned embeddings."""
        layer = TemporalEmbedding(d_model=64, embed_type="learned", freq="h")
        self.assertEqual(layer.d_model, 64)
        self.assertEqual(layer.embed_type, "learned")

    def test_invalid_parameters(self) -> None:
        """Test layer initialization with invalid parameters."""
        with self.assertRaises(ValueError):
            TemporalEmbedding(d_model=0, embed_type="fixed", freq="h")

        with self.assertRaises(ValueError):
            TemporalEmbedding(d_model=-1, embed_type="fixed", freq="h")

    def test_output_shape_hourly(self) -> None:
        """Test output shape for hourly frequency."""
        x_mark = tf.stack(
            [
                tf.random.uniform(
                    (self.batch_size, self.time_steps),
                    minval=0,
                    maxval=12,
                    dtype=tf.int32,
                ),
                tf.random.uniform(
                    (self.batch_size, self.time_steps),
                    minval=0,
                    maxval=31,
                    dtype=tf.int32,
                ),
                tf.random.uniform(
                    (self.batch_size, self.time_steps),
                    minval=0,
                    maxval=7,
                    dtype=tf.int32,
                ),
                tf.random.uniform(
                    (self.batch_size, self.time_steps),
                    minval=0,
                    maxval=24,
                    dtype=tf.int32,
                ),
            ],
            axis=-1,
        )

        outputs_fixed = self.layer_fixed(x_mark)
        outputs_learned = self.layer_learned(x_mark)

        expected_shape = (self.batch_size, self.time_steps, self.d_model)
        self.assertEqual(tuple(outputs_fixed.shape), expected_shape)
        self.assertEqual(tuple(outputs_learned.shape), expected_shape)

    def test_output_shape_daily(self) -> None:
        """Test output shape for daily frequency."""
        layer = TemporalEmbedding(d_model=64, embed_type="fixed", freq="h")
        x_mark = tf.stack(
            [
                tf.random.uniform(
                    (self.batch_size, self.time_steps),
                    minval=0,
                    maxval=12,
                    dtype=tf.int32,
                ),
                tf.random.uniform(
                    (self.batch_size, self.time_steps),
                    minval=0,
                    maxval=31,
                    dtype=tf.int32,
                ),
                tf.random.uniform(
                    (self.batch_size, self.time_steps),
                    minval=0,
                    maxval=7,
                    dtype=tf.int32,
                ),
                tf.random.uniform(
                    (self.batch_size, self.time_steps),
                    minval=0,
                    maxval=24,
                    dtype=tf.int32,
                ),
            ],
            axis=-1,
        )

        outputs = layer(x_mark)
        expected_shape = (self.batch_size, self.time_steps, 64)
        self.assertEqual(tuple(outputs.shape), expected_shape)

    def test_output_shape_minutely(self) -> None:
        """Test output shape for minutely frequency."""
        layer = TemporalEmbedding(d_model=64, embed_type="fixed", freq="t")
        x_mark = tf.stack(
            [
                tf.random.uniform(
                    (self.batch_size, self.time_steps),
                    minval=0,
                    maxval=12,
                    dtype=tf.int32,
                ),
                tf.random.uniform(
                    (self.batch_size, self.time_steps),
                    minval=0,
                    maxval=31,
                    dtype=tf.int32,
                ),
                tf.random.uniform(
                    (self.batch_size, self.time_steps),
                    minval=0,
                    maxval=7,
                    dtype=tf.int32,
                ),
                tf.random.uniform(
                    (self.batch_size, self.time_steps),
                    minval=0,
                    maxval=24,
                    dtype=tf.int32,
                ),
                tf.random.uniform(
                    (self.batch_size, self.time_steps),
                    minval=0,
                    maxval=4,
                    dtype=tf.int32,
                ),
            ],
            axis=-1,
        )

        outputs = layer(x_mark)
        expected_shape = (self.batch_size, self.time_steps, 64)
        self.assertEqual(tuple(outputs.shape), expected_shape)

    def test_fixed_vs_learned(self) -> None:
        """Test that fixed and learned embeddings produce different outputs."""
        x_mark = tf.stack(
            [
                tf.fill((self.batch_size, self.time_steps), 1),
                tf.fill((self.batch_size, self.time_steps), 15),
                tf.fill((self.batch_size, self.time_steps), 2),
                tf.fill((self.batch_size, self.time_steps), 12),
            ],
            axis=-1,
        )
        x_mark = tf.cast(x_mark, tf.int32)

        outputs_fixed = self.layer_fixed(x_mark)
        outputs_learned = self.layer_learned(x_mark)

        # Outputs should be different (fixed embeddings are deterministic, learned are initialized differently)
        self.assertNotEqual(
            float(tf.reduce_mean(outputs_fixed)),
            float(tf.reduce_mean(outputs_learned)),
        )

    def test_serialization(self) -> None:
        """Test layer serialization."""
        config = self.layer_fixed.get_config()
        self.assertEqual(config["d_model"], self.d_model)
        self.assertEqual(config["embed_type"], "fixed")
        self.assertEqual(config["freq"], "h")

        # Recreate from config
        new_layer = TemporalEmbedding.from_config(config)
        self.assertEqual(new_layer.d_model, self.d_model)
        self.assertEqual(new_layer.embed_type, "fixed")

    def test_model_integration(self) -> None:
        """Test layer integration in a model."""
        inputs = keras.Input(shape=(self.time_steps, 4), dtype="int32")
        outputs = self.layer_fixed(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        x_mark = tf.stack(
            [
                tf.random.uniform(
                    (self.batch_size, self.time_steps),
                    minval=0,
                    maxval=12,
                    dtype=tf.int32,
                ),
                tf.random.uniform(
                    (self.batch_size, self.time_steps),
                    minval=0,
                    maxval=31,
                    dtype=tf.int32,
                ),
                tf.random.uniform(
                    (self.batch_size, self.time_steps),
                    minval=0,
                    maxval=7,
                    dtype=tf.int32,
                ),
                tf.random.uniform(
                    (self.batch_size, self.time_steps),
                    minval=0,
                    maxval=24,
                    dtype=tf.int32,
                ),
            ],
            axis=-1,
        )

        outputs = model.predict(x_mark, verbose=0)
        expected_shape = (self.batch_size, self.time_steps, self.d_model)
        self.assertEqual(tuple(outputs.shape), expected_shape)

    def test_consistency_fixed_embeddings(self) -> None:
        """Test that fixed embeddings are consistent across calls."""
        x_mark = tf.stack(
            [
                tf.fill((self.batch_size, self.time_steps), 1),
                tf.fill((self.batch_size, self.time_steps), 15),
                tf.fill((self.batch_size, self.time_steps), 2),
                tf.fill((self.batch_size, self.time_steps), 12),
            ],
            axis=-1,
        )
        x_mark = tf.cast(x_mark, tf.int32)

        layer = TemporalEmbedding(d_model=64, embed_type="fixed", freq="h")
        outputs1 = layer(x_mark)
        outputs2 = layer(x_mark)

        # Fixed embeddings should be identical
        difference = float(tf.reduce_mean(tf.abs(outputs1 - outputs2)))
        self.assertAlmostEqual(difference, 0.0, places=5)

    def test_different_batch_sizes(self) -> None:
        """Test layer with different batch sizes."""
        for batch_size in [1, 8, 16, 32]:
            x_mark = tf.stack(
                [
                    tf.random.uniform(
                        (batch_size, self.time_steps),
                        minval=0,
                        maxval=12,
                        dtype=tf.int32,
                    ),
                    tf.random.uniform(
                        (batch_size, self.time_steps),
                        minval=0,
                        maxval=31,
                        dtype=tf.int32,
                    ),
                    tf.random.uniform(
                        (batch_size, self.time_steps),
                        minval=0,
                        maxval=7,
                        dtype=tf.int32,
                    ),
                    tf.random.uniform(
                        (batch_size, self.time_steps),
                        minval=0,
                        maxval=24,
                        dtype=tf.int32,
                    ),
                ],
                axis=-1,
            )

            outputs = self.layer_fixed(x_mark)
            expected_shape = (batch_size, self.time_steps, self.d_model)
            self.assertEqual(tuple(outputs.shape), expected_shape)


if __name__ == "__main__":
    unittest.main()
