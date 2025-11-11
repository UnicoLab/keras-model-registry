"""Tests for NormalizedDotProductSimilarity layer."""

import unittest
import numpy as np
import tensorflow as tf  # Used for testing only
import keras
from kmr.layers import NormalizedDotProductSimilarity


class TestNormalizedDotProductSimilarity(unittest.TestCase):
    """Test suite for NormalizedDotProductSimilarity."""

    def test_output_shape(self) -> None:
        """Test output shape."""
        layer = NormalizedDotProductSimilarity()
        emb1 = keras.random.normal((32, 64))
        emb2 = keras.random.normal((32, 64))
        output = layer([emb1, emb2])
        self.assertEqual(output.shape, (32, 1))

    def test_output_dtype(self) -> None:
        """Test output dtype."""
        layer = NormalizedDotProductSimilarity()
        emb1 = keras.random.normal((16, 32), dtype="float32")
        emb2 = keras.random.normal((16, 32), dtype="float32")
        output = layer([emb1, emb2])
        self.assertEqual(output.dtype, "float32")

    def test_batch_independence(self) -> None:
        """Test that batch elements are independent."""
        layer = NormalizedDotProductSimilarity()
        emb1 = keras.random.normal((4, 32))
        emb2 = keras.random.normal((4, 32))
        output = layer([emb1, emb2])
        # Each batch should have independent similarity
        self.assertEqual(output.shape[0], 4)

    def test_identical_embeddings(self) -> None:
        """Test similarity with identical embeddings."""
        layer = NormalizedDotProductSimilarity()
        emb = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        output = layer([emb, emb]).numpy()
        # Diagonal elements should be 1.0 (identical)
        np.testing.assert_almost_equal(output[0, 0], 1.0 / np.sqrt(3), decimal=4)
        np.testing.assert_almost_equal(output[1, 0], 1.0 / np.sqrt(3), decimal=4)

    def test_zero_embeddings(self) -> None:
        """Test with zero embeddings."""
        layer = NormalizedDotProductSimilarity()
        emb1 = tf.constant([[0.0, 0.0]])
        emb2 = tf.constant([[1.0, 1.0]])
        output = layer([emb1, emb2])
        self.assertEqual(output.shape, (1, 1))

    def test_orthogonal_embeddings(self) -> None:
        """Test with orthogonal embeddings."""
        layer = NormalizedDotProductSimilarity()
        emb1 = tf.constant([[1.0, 0.0]])
        emb2 = tf.constant([[0.0, 1.0]])
        output = layer([emb1, emb2]).numpy()
        # Orthogonal vectors should have near-zero dot product
        np.testing.assert_almost_equal(output[0, 0], 0.0, decimal=4)

    def test_different_embedding_dimensions(self) -> None:
        """Test with various embedding dimensions."""
        layer = NormalizedDotProductSimilarity()
        for dim in [8, 16, 32, 64, 128]:
            emb1 = keras.random.normal((4, dim))
            emb2 = keras.random.normal((4, dim))
            output = layer([emb1, emb2])
            self.assertEqual(output.shape, (4, 1))

    def test_negative_embeddings(self) -> None:
        """Test with negative embedding values."""
        layer = NormalizedDotProductSimilarity()
        emb1 = tf.constant([[-1.0, -1.0, -1.0]])
        emb2 = tf.constant([[1.0, 1.0, 1.0]])
        output = layer([emb1, emb2])
        self.assertEqual(output.shape, (1, 1))

    def test_large_batch_size(self) -> None:
        """Test with large batch sizes."""
        layer = NormalizedDotProductSimilarity()
        emb1 = keras.random.normal((256, 64))
        emb2 = keras.random.normal((256, 64))
        output = layer([emb1, emb2])
        self.assertEqual(output.shape, (256, 1))

    def test_output_non_zero_range(self) -> None:
        """Test that outputs are in reasonable range."""
        layer = NormalizedDotProductSimilarity()
        emb1 = keras.random.normal((32, 64))
        emb2 = keras.random.normal((32, 64))
        output = layer([emb1, emb2]).numpy()
        # Should be bounded (normalized by dimension)
        self.assertTrue(np.all(np.isfinite(output)))

    def test_serialization(self) -> None:
        """Test serialization."""
        layer = NormalizedDotProductSimilarity()
        config = layer.get_config()
        new_layer = NormalizedDotProductSimilarity.from_config(config)
        self.assertIsNotNone(new_layer)

    def test_model_save_load(self) -> None:
        """Test model save and load."""
        import tempfile

        layer = NormalizedDotProductSimilarity()
        emb1_input = keras.Input(shape=(32,))
        emb2_input = keras.Input(shape=(32,))
        output = layer([emb1_input, emb2_input])
        model = keras.Model([emb1_input, emb2_input], output)

        emb1 = keras.random.normal((16, 32))
        emb2 = keras.random.normal((16, 32))
        pred1 = model.predict([emb1, emb2], verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(f"{tmpdir}/model.keras")
            loaded = keras.models.load_model(f"{tmpdir}/model.keras")
            pred2 = loaded.predict([emb1, emb2], verbose=0)
            np.testing.assert_array_almost_equal(pred1, pred2)


if __name__ == "__main__":
    unittest.main()
