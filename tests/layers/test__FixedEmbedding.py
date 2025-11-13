"""Unit tests for FixedEmbedding layer.

Note: TensorFlow is used in tests for validation purposes only.
The actual layer implementation uses only Keras 3 operations.
"""

import unittest
import tensorflow as tf  # Used for testing only
from keras import layers, Model
from kerasfactory.layers.FixedEmbedding import FixedEmbedding


class TestFixedEmbedding(unittest.TestCase):
    """Test cases for FixedEmbedding layer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.batch_size = 16
        self.seq_len = 50
        self.n_features = 32
        self.d_model = 64
        self.layer = FixedEmbedding(n_features=self.n_features, d_model=self.d_model)
        # Using TensorFlow for test data generation only
        self.inputs = tf.random.uniform(
            (self.batch_size, self.seq_len),
            minval=0,
            maxval=self.n_features,
            dtype=tf.int32,
        )

    def test_initialization(self) -> None:
        """Test layer initialization."""
        layer = FixedEmbedding(n_features=32, d_model=64)
        self.assertEqual(layer.n_features, 32)
        self.assertEqual(layer.d_model, 64)

    def test_invalid_initialization(self) -> None:
        """Test layer initialization with invalid parameters."""
        with self.assertRaises(ValueError):
            FixedEmbedding(n_features=0, d_model=64)

        with self.assertRaises(ValueError):
            FixedEmbedding(n_features=32, d_model=0)

        with self.assertRaises(ValueError):
            FixedEmbedding(n_features=-1, d_model=64)

    def test_output_shape(self) -> None:
        """Test output shape of the layer."""
        outputs = self.layer(self.inputs)
        expected_shape = (self.batch_size, self.seq_len, self.d_model)
        self.assertEqual(tuple(outputs.shape), expected_shape)

    def test_call(self) -> None:
        """Test forward pass of the layer."""
        outputs = self.layer(self.inputs)
        self.assertIsNotNone(outputs)
        self.assertEqual(outputs.dtype, tf.float32)

    def test_embedding_values_fixed(self) -> None:
        """Test that embeddings are fixed (non-trainable)."""
        # Call layer twice with same input
        output1 = self.layer(self.inputs)
        output2 = self.layer(self.inputs)

        # Outputs should be identical
        tf.debugging.assert_equal(output1, output2)

    def test_different_indices(self) -> None:
        """Test that different indices produce different embeddings."""
        # Create two inputs with different indices
        input1 = tf.constant([[0, 1, 2]], dtype=tf.int32)
        input2 = tf.constant([[3, 4, 5]], dtype=tf.int32)

        layer = FixedEmbedding(n_features=10, d_model=32)
        output1 = layer(input1)
        output2 = layer(input2)

        # Outputs should be different
        self.assertFalse(tf.reduce_all(tf.equal(output1, output2)))

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        config = self.layer.get_config()
        self.assertIn("n_features", config)
        self.assertIn("d_model", config)

        new_layer = FixedEmbedding.from_config(config)
        self.assertEqual(new_layer.n_features, self.layer.n_features)
        self.assertEqual(new_layer.d_model, self.layer.d_model)

    def test_model_integration(self) -> None:
        """Test layer integration in a model."""
        inputs = layers.Input(shape=(self.seq_len,), dtype=tf.int32)
        x = FixedEmbedding(n_features=self.n_features, d_model=self.d_model)(inputs)
        outputs = layers.Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer="adam", loss="mse")

        # Generate dummy data
        x_data = tf.random.uniform(
            (10, self.seq_len),
            minval=0,
            maxval=self.n_features,
            dtype=tf.int32,
        )
        y_data = tf.random.normal((10, 1))

        history = model.fit(x_data, y_data, epochs=1, verbose=0)
        self.assertTrue(history.history["loss"][0] > 0)

    def test_embedding_orthogonality(self) -> None:
        """Test sinusoidal embeddings have good properties."""
        # Build the layer
        self.layer.build((None, self.seq_len))

        # Get embeddings for consecutive positions
        input1 = tf.constant([[0]], dtype=tf.int32)
        input2 = tf.constant([[1]], dtype=tf.int32)

        emb1 = self.layer(input1)  # (1, 1, d_model)
        emb2 = self.layer(input2)  # (1, 1, d_model)

        # Compute distance
        dist = tf.sqrt(tf.reduce_sum((emb1 - emb2) ** 2))

        # Distance should be reasonable (not too close, not too far)
        self.assertGreater(float(dist), 0.1)
        self.assertLess(float(dist), float(tf.sqrt(tf.cast(self.d_model, tf.float32))))

    def test_different_batch_sizes(self) -> None:
        """Test with different batch sizes."""
        for batch_size in [1, 8, 16, 32]:
            inputs = tf.random.uniform(
                (batch_size, self.seq_len),
                minval=0,
                maxval=self.n_features,
                dtype=tf.int32,
            )
            outputs = self.layer(inputs)
            self.assertEqual(outputs.shape[0], batch_size)
            self.assertEqual(outputs.shape[1], self.seq_len)
            self.assertEqual(outputs.shape[2], self.d_model)


if __name__ == "__main__":
    unittest.main()
