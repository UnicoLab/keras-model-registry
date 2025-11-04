"""Tests for ExplainableRecommendationModel."""

import unittest
import numpy as np
import tensorflow as tf
import keras
from kmr.models import ExplainableRecommendationModel


class TestExplainableRecommendationModel(unittest.TestCase):
    """Test suite for ExplainableRecommendationModel."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.num_users = 100
        self.num_items = 50
        self.batch_size = 16
        self.model = ExplainableRecommendationModel(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=32,
            top_k=10,
        )

    def test_initialization_default(self) -> None:
        """Test model initialization with default parameters."""
        model = ExplainableRecommendationModel(
            num_users=100,
            num_items=50,
        )
        self.assertEqual(model.num_users, 100)
        self.assertEqual(model.num_items, 50)
        self.assertEqual(model.embedding_dim, 32)

    def test_initialization_custom(self) -> None:
        """Test custom initialization."""
        model = ExplainableRecommendationModel(
            num_users=200,
            num_items=100,
            embedding_dim=64,
            feedback_weight=0.7,
        )
        self.assertEqual(model.embedding_dim, 64)
        self.assertEqual(model.feedback_weight, 0.7)

    def test_invalid_feedback_weight(self) -> None:
        """Test that invalid feedback_weight raises error."""
        with self.assertRaises(ValueError):
            ExplainableRecommendationModel(
                num_users=100,
                num_items=50,
                feedback_weight=1.5,
            )

    def test_forward_pass_without_feedback(self) -> None:
        """Test forward pass without feedback."""
        user_ids = tf.constant(np.random.randint(0, self.num_users, (self.batch_size,)))
        item_ids = tf.constant(
            np.random.randint(0, self.num_items, (self.batch_size, self.num_items)),
        )

        rec_indices, rec_scores, sim_matrix = self.model(
            [user_ids, item_ids],
            training=False,
        )

        self.assertEqual(rec_indices.shape, (self.batch_size, self.model.top_k))
        self.assertEqual(rec_scores.shape, (self.batch_size, self.model.top_k))
        self.assertEqual(sim_matrix.shape, (self.batch_size, self.num_items))

    def test_similarity_matrix_valid_range(self) -> None:
        """Test that similarity matrix is in valid range [-1, 1]."""
        user_ids = tf.constant(np.random.randint(0, self.num_users, (self.batch_size,)))
        item_ids = tf.constant(
            np.random.randint(0, self.num_items, (self.batch_size, self.num_items)),
        )

        _, _, sim_matrix = self.model([user_ids, item_ids], training=False)

        min_sim = tf.reduce_min(sim_matrix).numpy()
        max_sim = tf.reduce_max(sim_matrix).numpy()
        self.assertGreaterEqual(min_sim, -1.1)
        self.assertLessEqual(max_sim, 1.1)

    def test_output_indices_valid(self) -> None:
        """Test that output indices are valid."""
        user_ids = tf.constant(np.random.randint(0, self.num_users, (self.batch_size,)))
        item_ids = tf.constant(
            np.random.randint(0, self.num_items, (self.batch_size, self.num_items)),
        )

        rec_indices, _, _ = self.model([user_ids, item_ids], training=False)

        max_idx = tf.reduce_max(rec_indices).numpy()
        min_idx = tf.reduce_min(rec_indices).numpy()
        self.assertGreaterEqual(min_idx, 0)
        self.assertLess(max_idx, self.num_items)

    def test_different_batch_sizes(self) -> None:
        """Test with different batch sizes."""
        for batch_size in [1, 8, 16]:
            user_ids = tf.constant(np.random.randint(0, self.num_users, (batch_size,)))
            item_ids = tf.constant(
                np.random.randint(0, self.num_items, (batch_size, self.num_items)),
            )

            rec_indices, rec_scores, sim_matrix = self.model(
                [user_ids, item_ids],
                training=False,
            )

            self.assertEqual(rec_indices.shape[0], batch_size)
            self.assertEqual(rec_scores.shape[0], batch_size)
            self.assertEqual(sim_matrix.shape[0], batch_size)

    def test_serialization(self) -> None:
        """Test model serialization."""
        config = self.model.get_config()
        self.assertEqual(config["num_users"], self.num_users)
        self.assertEqual(config["num_items"], self.num_items)

    def test_deserialization(self) -> None:
        """Test model deserialization."""
        config = self.model.get_config()
        new_model = ExplainableRecommendationModel.from_config(config)
        self.assertEqual(new_model.num_users, self.model.num_users)

    def test_inference_consistency(self) -> None:
        """Test inference consistency."""
        user_ids = tf.constant(np.random.randint(0, self.num_users, (self.batch_size,)))
        item_ids = tf.constant(
            np.random.randint(0, self.num_items, (self.batch_size, self.num_items)),
        )

        idx1, scores1, sim1 = self.model([user_ids, item_ids], training=False)
        idx2, scores2, sim2 = self.model([user_ids, item_ids], training=False)

        np.testing.assert_array_equal(idx1.numpy(), idx2.numpy())
        np.testing.assert_array_almost_equal(sim1.numpy(), sim2.numpy())


if __name__ == "__main__":
    unittest.main()
