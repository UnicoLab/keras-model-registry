"""Tests for UnifiedRecommendationModel."""

import unittest
import numpy as np
import tensorflow as tf
import keras
from kmr.models import UnifiedRecommendationModel


class TestUnifiedRecommendationModel(unittest.TestCase):
    """Test suite for UnifiedRecommendationModel."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.num_users = 100
        self.num_items = 50
        self.user_feature_dim = 32
        self.item_feature_dim = 32
        self.batch_size = 16
        self.model = UnifiedRecommendationModel(
            num_users=self.num_users,
            num_items=self.num_items,
            user_feature_dim=self.user_feature_dim,
            item_feature_dim=self.item_feature_dim,
            embedding_dim=32,
            top_k=10,
        )

    def test_initialization_default(self) -> None:
        """Test model initialization with default parameters."""
        model = UnifiedRecommendationModel(
            num_users=100,
            num_items=50,
            user_feature_dim=32,
            item_feature_dim=32,
        )
        self.assertEqual(model.num_users, 100)
        self.assertEqual(model.num_items, 50)
        self.assertEqual(model.top_k, 10)

    def test_invalid_parameters(self) -> None:
        """Test that invalid parameters raise errors."""
        with self.assertRaises(ValueError):
            UnifiedRecommendationModel(
                num_users=0,
                num_items=50,
                user_feature_dim=32,
                item_feature_dim=32,
            )

    def test_forward_pass_output_shapes(self) -> None:
        """Test that forward pass produces correct output shapes."""
        user_ids = tf.constant(np.random.randint(0, self.num_users, (self.batch_size,)))
        user_features = tf.constant(
            np.random.randn(self.batch_size, self.user_feature_dim).astype(np.float32),
        )
        item_ids = tf.constant(
            np.random.randint(0, self.num_items, (self.batch_size, self.num_items)),
        )
        item_features = tf.constant(
            np.random.randn(
                self.batch_size,
                self.num_items,
                self.item_feature_dim,
            ).astype(np.float32),
        )

        rec_indices, rec_scores = self.model(
            [user_ids, user_features, item_ids, item_features],
            training=False,
        )

        self.assertEqual(rec_indices.shape, (self.batch_size, self.model.top_k))
        self.assertEqual(rec_scores.shape, (self.batch_size, self.model.top_k))

    def test_output_indices_valid(self) -> None:
        """Test that output indices are valid."""
        user_ids = tf.constant(np.random.randint(0, self.num_users, (self.batch_size,)))
        user_features = tf.constant(
            np.random.randn(self.batch_size, self.user_feature_dim).astype(np.float32),
        )
        item_ids = tf.constant(
            np.random.randint(0, self.num_items, (self.batch_size, self.num_items)),
        )
        item_features = tf.constant(
            np.random.randn(
                self.batch_size,
                self.num_items,
                self.item_feature_dim,
            ).astype(np.float32),
        )

        rec_indices, _ = self.model(
            [user_ids, user_features, item_ids, item_features],
            training=False,
        )

        max_idx = tf.reduce_max(rec_indices).numpy()
        min_idx = tf.reduce_min(rec_indices).numpy()
        self.assertGreaterEqual(min_idx, 0)
        self.assertLess(max_idx, self.num_items)

    def test_different_batch_sizes(self) -> None:
        """Test with different batch sizes."""
        for batch_size in [1, 8, 16]:
            user_ids = tf.constant(np.random.randint(0, self.num_users, (batch_size,)))
            user_features = tf.constant(
                np.random.randn(batch_size, self.user_feature_dim).astype(np.float32),
            )
            item_ids = tf.constant(
                np.random.randint(0, self.num_items, (batch_size, self.num_items)),
            )
            item_features = tf.constant(
                np.random.randn(
                    batch_size,
                    self.num_items,
                    self.item_feature_dim,
                ).astype(np.float32),
            )

            rec_indices, rec_scores = self.model(
                [user_ids, user_features, item_ids, item_features],
                training=False,
            )

            self.assertEqual(rec_indices.shape[0], batch_size)
            self.assertEqual(rec_scores.shape[0], batch_size)

    def test_serialization(self) -> None:
        """Test model serialization."""
        config = self.model.get_config()
        self.assertEqual(config["num_users"], self.num_users)
        self.assertEqual(config["num_items"], self.num_items)

    def test_deserialization(self) -> None:
        """Test model deserialization."""
        config = self.model.get_config()
        new_model = UnifiedRecommendationModel.from_config(config)
        self.assertEqual(new_model.num_users, self.model.num_users)

    def test_inference_consistency(self) -> None:
        """Test inference consistency."""
        user_ids = tf.constant(np.random.randint(0, self.num_users, (self.batch_size,)))
        user_features = tf.constant(
            np.random.randn(self.batch_size, self.user_feature_dim).astype(np.float32),
        )
        item_ids = tf.constant(
            np.random.randint(0, self.num_items, (self.batch_size, self.num_items)),
        )
        item_features = tf.constant(
            np.random.randn(
                self.batch_size,
                self.num_items,
                self.item_feature_dim,
            ).astype(np.float32),
        )

        idx1, scores1 = self.model(
            [user_ids, user_features, item_ids, item_features],
            training=False,
        )
        idx2, scores2 = self.model(
            [user_ids, user_features, item_ids, item_features],
            training=False,
        )

        np.testing.assert_array_equal(idx1.numpy(), idx2.numpy())


if __name__ == "__main__":
    unittest.main()
