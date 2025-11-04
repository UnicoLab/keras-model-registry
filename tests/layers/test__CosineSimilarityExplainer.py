"""Tests for CosineSimilarityExplainer layer."""

import unittest
import numpy as np
import tensorflow as tf  # Used for testing only
import keras
from kmr.layers import CosineSimilarityExplainer


class TestCosineSimilarityExplainer(unittest.TestCase):
    """Test suite for CosineSimilarityExplainer."""

    def test_output_shape(self) -> None:
        """Test output shape."""
        layer = CosineSimilarityExplainer()
        user_emb = keras.random.normal((8, 32))
        item_emb = keras.random.normal((16, 32))
        output = layer([user_emb, item_emb])
        self.assertEqual(output.shape, (8, 16))

    def test_output_range(self) -> None:
        """Test that output is in valid cosine range [-1, 1]."""
        layer = CosineSimilarityExplainer()
        user_emb = keras.random.normal((8, 32))
        item_emb = keras.random.normal((16, 32))
        output = layer([user_emb, item_emb]).numpy()
        self.assertTrue(np.all(output >= -1.0))
        self.assertTrue(np.all(output <= 1.0))

    def test_self_similarity_high(self) -> None:
        """Test that similar embeddings have high similarity."""
        layer = CosineSimilarityExplainer()
        # Same embedding should have high similarity
        emb = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
        output = layer([emb, emb]).numpy()
        # Diagonal should be 1.0 (perfect similarity)
        np.testing.assert_almost_equal(output[0, 0], 1.0, decimal=4)
        np.testing.assert_almost_equal(output[1, 1], 1.0, decimal=4)

    def test_symmetry_of_similarity(self) -> None:
        """Test that similarity is symmetric."""
        layer = CosineSimilarityExplainer()
        user_emb = keras.random.normal((4, 32))
        item_emb = keras.random.normal((8, 32))

        output1 = layer([user_emb, item_emb]).numpy()
        output2 = layer([item_emb, user_emb]).numpy()

        # Transpose of output1 should equal output2
        np.testing.assert_almost_equal(output1, output2.T)

    def test_orthogonal_vectors(self) -> None:
        """Test cosine similarity with orthogonal vectors."""
        layer = CosineSimilarityExplainer()
        # Create orthogonal vectors
        emb1 = tf.constant([[1.0, 0.0, 0.0]])
        emb2 = tf.constant([[0.0, 1.0, 0.0]])
        output = layer([emb1, emb2]).numpy()
        # Should be approximately 0
        np.testing.assert_almost_equal(output[0, 0], 0.0, decimal=4)

    def test_parallel_vectors(self) -> None:
        """Test cosine similarity with parallel vectors."""
        layer = CosineSimilarityExplainer()
        emb = tf.constant([[1.0, 0.0, 0.0]], dtype="float32")
        emb_scaled = tf.constant([[2.0, 0.0, 0.0]], dtype="float32")
        output = layer([emb, emb_scaled]).numpy()
        # Should be 1.0 (perfectly aligned)
        np.testing.assert_almost_equal(output[0, 0], 1.0, decimal=4)

    def test_antiparallel_vectors(self) -> None:
        """Test cosine similarity with antiparallel vectors."""
        layer = CosineSimilarityExplainer()
        emb1 = tf.constant([[1.0, 0.0, 0.0]], dtype="float32")
        emb2 = tf.constant([[-1.0, 0.0, 0.0]], dtype="float32")
        output = layer([emb1, emb2]).numpy()
        # Should be -1.0 (perfectly opposite)
        np.testing.assert_almost_equal(output[0, 0], -1.0, decimal=4)

    def test_output_range_bounds(self) -> None:
        """Test that output is strictly within [-1, 1]."""
        layer = CosineSimilarityExplainer()
        for _ in range(5):
            user_emb = keras.random.normal((8, 32))
            item_emb = keras.random.normal((16, 32))
            output = layer([user_emb, item_emb]).numpy()
            self.assertTrue(np.all(output >= -1.0 - 1e-5))
            self.assertTrue(np.all(output <= 1.0 + 1e-5))

    def test_different_embedding_dims(self) -> None:
        """Test with various embedding dimensions."""
        layer = CosineSimilarityExplainer()
        for dim in [8, 16, 32, 64]:
            user_emb = keras.random.normal((4, dim))
            item_emb = keras.random.normal((8, dim))
            output = layer([user_emb, item_emb])
            self.assertEqual(output.shape, (4, 8))

    def test_batch_independence(self) -> None:
        """Test that batch elements are independent."""
        layer = CosineSimilarityExplainer()
        user_emb = keras.random.normal((4, 32))
        item_emb = keras.random.normal((8, 32))
        output = layer([user_emb, item_emb])

        # Each user should have similarity with all items
        self.assertEqual(output.shape, (4, 8))

    def test_zero_embeddings(self) -> None:
        """Test behavior with zero embeddings."""
        layer = CosineSimilarityExplainer()
        user_emb = keras.ops.zeros((2, 32))
        item_emb = keras.random.normal((4, 32))
        output = layer([user_emb, item_emb]).numpy()
        # Zero embeddings should have zero similarity
        self.assertTrue(np.all(np.isfinite(output)))

    def test_large_batch_sizes(self) -> None:
        """Test with large batch sizes."""
        layer = CosineSimilarityExplainer()
        user_emb = keras.random.normal((128, 64))
        item_emb = keras.random.normal((256, 64))
        output = layer([user_emb, item_emb])
        self.assertEqual(output.shape, (128, 256))

    def test_serialization(self) -> None:
        """Test serialization."""
        layer = CosineSimilarityExplainer()
        config = layer.get_config()
        new_layer = CosineSimilarityExplainer.from_config(config)
        self.assertIsNotNone(new_layer)

    def test_model_save_load(self) -> None:
        """Test model save and load."""
        import tempfile

        layer = CosineSimilarityExplainer()
        user_input = keras.Input(shape=(32,))
        item_input = keras.Input(shape=(32,))
        output = layer([user_input, item_input])
        model = keras.Model([user_input, item_input], output)

        user_emb = keras.random.normal((8, 32))
        item_emb = keras.random.normal((8, 32))  # Match batch size
        pred1 = model.predict([user_emb, item_emb], verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(f"{tmpdir}/model.keras")
            loaded = keras.models.load_model(f"{tmpdir}/model.keras")
            pred2 = loaded.predict([user_emb, item_emb], verbose=0)
            np.testing.assert_array_almost_equal(pred1, pred2)


if __name__ == "__main__":
    unittest.main()
