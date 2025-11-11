"""Tests for CollaborativeUserItemEmbedding layer."""

import unittest
import numpy as np
import tensorflow as tf  # Used for testing only
import keras
from kmr.layers import CollaborativeUserItemEmbedding


class TestCollaborativeUserItemEmbedding(unittest.TestCase):
    """Test suite for CollaborativeUserItemEmbedding."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.layer = CollaborativeUserItemEmbedding(
            num_users=100,
            num_items=50,
            embedding_dim=32,
            l2_reg=1e-6,
        )

    def test_initialization_default(self) -> None:
        """Test layer initialization with default parameters."""
        layer = CollaborativeUserItemEmbedding(num_users=100, num_items=50)
        self.assertEqual(layer.num_users, 100)
        self.assertEqual(layer.num_items, 50)
        self.assertEqual(layer.embedding_dim, 32)
        self.assertEqual(layer.l2_reg, 1e-6)

    def test_initialization_custom(self) -> None:
        """Test with custom parameters."""
        layer = CollaborativeUserItemEmbedding(
            num_users=200,
            num_items=100,
            embedding_dim=64,
            l2_reg=1e-5,
        )
        self.assertEqual(layer.num_users, 200)
        self.assertEqual(layer.num_items, 100)
        self.assertEqual(layer.embedding_dim, 64)
        self.assertEqual(layer.l2_reg, 1e-5)

    def test_invalid_num_users(self) -> None:
        """Test that invalid num_users raises error."""
        with self.assertRaises(ValueError):
            CollaborativeUserItemEmbedding(num_users=0, num_items=50)

    def test_invalid_num_items(self) -> None:
        """Test that invalid num_items raises error."""
        with self.assertRaises(ValueError):
            CollaborativeUserItemEmbedding(num_users=100, num_items=-1)

    def test_invalid_embedding_dim(self) -> None:
        """Test that invalid embedding_dim raises error."""
        with self.assertRaises(ValueError):
            CollaborativeUserItemEmbedding(num_users=100, num_items=50, embedding_dim=0)

    def test_invalid_l2_reg(self) -> None:
        """Test that invalid l2_reg raises error."""
        with self.assertRaises(ValueError):
            CollaborativeUserItemEmbedding(num_users=100, num_items=50, l2_reg=-0.1)

    def test_output_shapes(self) -> None:
        """Test that output shapes are correct."""
        user_ids = tf.constant([1, 5, 10, 3])
        item_ids = tf.constant([2, 8, 15, 7])
        user_emb, item_emb = self.layer([user_ids, item_ids])

        self.assertEqual(user_emb.shape, (4, 32))
        self.assertEqual(item_emb.shape, (4, 32))

    def test_output_dtype(self) -> None:
        """Test output dtype matches input dtype."""
        user_ids = tf.constant([1, 5], dtype="int32")
        item_ids = tf.constant([2, 8], dtype="int32")
        user_emb, item_emb = self.layer([user_ids, item_ids])

        self.assertEqual(user_emb.dtype, keras.backend.floatx())
        self.assertEqual(item_emb.dtype, keras.backend.floatx())

    def test_different_batch_sizes(self) -> None:
        """Test with different batch sizes."""
        for batch_size in [1, 16, 32, 64]:
            user_ids = tf.constant(np.random.randint(0, 100, batch_size))
            item_ids = tf.constant(np.random.randint(0, 50, batch_size))
            user_emb, item_emb = self.layer([user_ids, item_ids])

            self.assertEqual(user_emb.shape[0], batch_size)
            self.assertEqual(item_emb.shape[0], batch_size)

    def test_embedding_values_different(self) -> None:
        """Test that different IDs produce different embeddings."""
        user_ids1 = tf.constant([1, 1])
        user_ids2 = tf.constant([1, 2])
        item_ids = tf.constant([5, 5])

        user_emb1, _ = self.layer([user_ids1, item_ids])
        user_emb2, _ = self.layer([user_ids2, item_ids])

        # First user should be same, second should differ
        np.testing.assert_array_almost_equal(user_emb1[0].numpy(), user_emb2[0].numpy())
        self.assertFalse(np.allclose(user_emb1[1].numpy(), user_emb2[1].numpy()))

    def test_embedding_dimension_consistency(self) -> None:
        """Test that embedding dimension is consistent across calls."""
        for embedding_dim in [8, 16, 32, 64]:
            layer = CollaborativeUserItemEmbedding(
                num_users=100,
                num_items=50,
                embedding_dim=embedding_dim,
            )
            user_ids = tf.constant([1, 5, 10])
            item_ids = tf.constant([2, 8, 15])
            user_emb, item_emb = layer([user_ids, item_ids])
            self.assertEqual(user_emb.shape[1], embedding_dim)
            self.assertEqual(item_emb.shape[1], embedding_dim)

    def test_l2_regularization_weights(self) -> None:
        """Test that L2 regularization is applied to weights."""
        layer = CollaborativeUserItemEmbedding(
            num_users=100,
            num_items=50,
            embedding_dim=32,
            l2_reg=0.01,
        )
        user_ids = tf.constant([1, 5])
        item_ids = tf.constant([2, 8])
        _ = layer([user_ids, item_ids])

        # Check that layer has losses (from L2 regularization)
        self.assertGreater(len(layer.losses), 0)

    def test_boundary_id_values(self) -> None:
        """Test with boundary ID values."""
        # Test with 0 (first valid ID)
        user_ids = tf.constant([0, 99])
        item_ids = tf.constant([0, 49])
        user_emb, item_emb = self.layer([user_ids, item_ids])
        self.assertEqual(user_emb.shape, (2, 32))
        self.assertEqual(item_emb.shape, (2, 32))

    def test_repeated_ids(self) -> None:
        """Test that repeated IDs produce identical embeddings."""
        user_ids = tf.constant([5, 5, 5])
        item_ids = tf.constant([10, 10, 10])
        user_emb, item_emb = self.layer([user_ids, item_ids])

        # All embeddings should be identical
        np.testing.assert_array_almost_equal(user_emb[0].numpy(), user_emb[1].numpy())
        np.testing.assert_array_almost_equal(user_emb[1].numpy(), user_emb[2].numpy())
        np.testing.assert_array_almost_equal(item_emb[0].numpy(), item_emb[1].numpy())
        np.testing.assert_array_almost_equal(item_emb[1].numpy(), item_emb[2].numpy())

    def test_embedding_non_zero(self) -> None:
        """Test that embeddings are non-zero."""
        user_ids = tf.constant([1, 5, 10, 3])
        item_ids = tf.constant([2, 8, 15, 7])
        user_emb, item_emb = self.layer([user_ids, item_ids])

        # Check that embeddings have non-zero values
        self.assertGreater(np.abs(user_emb.numpy()).max(), 0)
        self.assertGreater(np.abs(item_emb.numpy()).max(), 0)

    def test_serialization_get_config(self) -> None:
        """Test layer serialization."""
        config = self.layer.get_config()
        self.assertEqual(config["num_users"], 100)
        self.assertEqual(config["num_items"], 50)
        self.assertEqual(config["embedding_dim"], 32)
        self.assertEqual(config["l2_reg"], 1e-6)

    def test_deserialization_from_config(self) -> None:
        """Test layer deserialization."""
        config = self.layer.get_config()
        new_layer = CollaborativeUserItemEmbedding.from_config(config)
        self.assertEqual(new_layer.num_users, 100)
        self.assertEqual(new_layer.num_items, 50)
        self.assertEqual(new_layer.embedding_dim, 32)

    def test_model_save_load(self) -> None:
        """Test model save and load with layer."""
        import tempfile

        user_ids_input = keras.Input(shape=(), dtype="int32")
        item_ids_input = keras.Input(shape=(), dtype="int32")
        user_emb, item_emb = self.layer([user_ids_input, item_ids_input])
        model = keras.Model([user_ids_input, item_ids_input], [user_emb, item_emb])

        user_ids = np.array([1, 5, 10, 3], dtype="int32")
        item_ids = np.array([2, 8, 15, 7], dtype="int32")
        pred1_u, pred1_i = model.predict([user_ids, item_ids], verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = f"{tmpdir}/model.keras"
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)
            pred2_u, pred2_i = loaded_model.predict([user_ids, item_ids], verbose=0)

            np.testing.assert_array_almost_equal(pred1_u, pred2_u)
            np.testing.assert_array_almost_equal(pred1_i, pred2_i)


if __name__ == "__main__":
    unittest.main()
