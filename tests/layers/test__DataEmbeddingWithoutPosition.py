"""Unit tests for DataEmbeddingWithoutPosition layer.

Note: TensorFlow is used in tests for validation purposes only.
The actual layer implementation uses only Keras 3 operations.
"""

import unittest
import tensorflow as tf
import keras
from kerasfactory.layers.DataEmbeddingWithoutPosition import (
    DataEmbeddingWithoutPosition,
)


class TestDataEmbeddingWithoutPosition(unittest.TestCase):
    """Test cases for DataEmbeddingWithoutPosition layer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.c_in = 8
        self.d_model = 64
        self.dropout_rate = 0.1
        self.batch_size = 16
        self.time_steps = 100
        self.layer = DataEmbeddingWithoutPosition(
            c_in=self.c_in,
            d_model=self.d_model,
            embed_type="fixed",
            freq="h",
            dropout=self.dropout_rate,
        )

    def test_initialization(self) -> None:
        """Test layer initialization."""
        layer = DataEmbeddingWithoutPosition(
            c_in=8,
            d_model=64,
            embed_type="fixed",
            freq="h",
            dropout=0.1,
        )
        self.assertEqual(layer.c_in, 8)
        self.assertEqual(layer.d_model, 64)
        self.assertEqual(layer.dropout_rate, 0.1)

    def test_invalid_parameters(self) -> None:
        """Test layer initialization with invalid parameters."""
        with self.assertRaises(ValueError):
            DataEmbeddingWithoutPosition(
                c_in=0,
                d_model=64,
                embed_type="fixed",
                freq="h",
            )

        with self.assertRaises(ValueError):
            DataEmbeddingWithoutPosition(
                c_in=8,
                d_model=0,
                embed_type="fixed",
                freq="h",
            )

    def test_output_shape_without_temporal(self) -> None:
        """Test output shape without temporal features."""
        x = tf.random.normal((self.batch_size, self.time_steps, self.c_in))
        outputs = self.layer(x)

        expected_shape = (self.batch_size, self.time_steps, self.d_model)
        self.assertEqual(tuple(outputs.shape), expected_shape)

    def test_output_shape_with_temporal(self) -> None:
        """Test output shape with temporal features."""
        x = tf.random.normal((self.batch_size, self.time_steps, self.c_in))
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

        outputs = self.layer([x, x_mark])
        expected_shape = (self.batch_size, self.time_steps, self.d_model)
        self.assertEqual(tuple(outputs.shape), expected_shape)

    def test_dropout_enabled(self) -> None:
        """Test dropout functionality."""
        x = tf.random.normal((self.batch_size, self.time_steps, self.c_in))

        # Training mode with dropout
        outputs1 = self.layer(x, training=True)
        outputs2 = self.layer(x, training=True)

        # Outputs should be different due to dropout
        difference = float(tf.reduce_mean(tf.abs(outputs1 - outputs2)))
        self.assertGreater(difference, 0.0)

    def test_dropout_disabled(self) -> None:
        """Test no dropout during inference."""
        x = tf.random.normal((self.batch_size, self.time_steps, self.c_in))

        # Inference mode without dropout
        outputs1 = self.layer(x, training=False)
        outputs2 = self.layer(x, training=False)

        # Outputs should be identical without dropout
        difference = float(tf.reduce_mean(tf.abs(outputs1 - outputs2)))
        self.assertAlmostEqual(difference, 0.0, places=5)

    def test_combined_embeddings(self) -> None:
        """Test combination of token and temporal embeddings."""
        x = tf.random.normal((self.batch_size, self.time_steps, self.c_in))
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

        outputs = self.layer([x, x_mark])
        expected_shape = (self.batch_size, self.time_steps, self.d_model)
        self.assertEqual(tuple(outputs.shape), expected_shape)

    def test_serialization(self) -> None:
        """Test layer serialization."""
        config = self.layer.get_config()
        self.assertEqual(config["c_in"], self.c_in)
        self.assertEqual(config["d_model"], self.d_model)
        self.assertEqual(config["dropout"], self.dropout_rate)

        # Recreate from config
        new_layer = DataEmbeddingWithoutPosition.from_config(config)
        self.assertEqual(new_layer.c_in, self.c_in)
        self.assertEqual(new_layer.d_model, self.d_model)

    def test_model_integration(self) -> None:
        """Test layer integration in a model."""
        inputs_x = keras.Input(shape=(self.time_steps, self.c_in), dtype="float32")
        inputs_x_mark = keras.Input(shape=(self.time_steps, 4), dtype="int32")

        outputs = self.layer([inputs_x, inputs_x_mark])
        model = keras.Model(inputs=[inputs_x, inputs_x_mark], outputs=outputs)

        x = tf.random.normal((self.batch_size, self.time_steps, self.c_in))
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

        predictions = model.predict([x, x_mark], verbose=0)
        expected_shape = (self.batch_size, self.time_steps, self.d_model)
        self.assertEqual(tuple(predictions.shape), expected_shape)

    def test_different_batch_sizes(self) -> None:
        """Test with different batch sizes."""
        for batch_size in [1, 8, 32]:
            x = tf.random.normal((batch_size, self.time_steps, self.c_in))
            outputs = self.layer(x)
            expected_shape = (batch_size, self.time_steps, self.d_model)
            self.assertEqual(tuple(outputs.shape), expected_shape)

    def test_different_embedding_types(self) -> None:
        """Test with different embedding types."""
        x = tf.random.normal((self.batch_size, self.time_steps, self.c_in))

        # Test with learned embedding
        layer_learned = DataEmbeddingWithoutPosition(
            c_in=self.c_in,
            d_model=self.d_model,
            embed_type="learned",
            freq="h",
            dropout=self.dropout_rate,
        )
        outputs = layer_learned(x)
        expected_shape = (self.batch_size, self.time_steps, self.d_model)
        self.assertEqual(tuple(outputs.shape), expected_shape)


if __name__ == "__main__":
    unittest.main()
