"""Tests for DeepRankingModel."""

import unittest
import numpy as np
import tensorflow as tf
import keras
from kmr.models import DeepRankingModel


class TestDeepRankingModel(unittest.TestCase):
    """Test suite for DeepRankingModel."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.num_items = 50
        self.user_feature_dim = 32
        self.item_feature_dim = 32
        self.batch_size = 16
        self.model = DeepRankingModel(
            user_feature_dim=self.user_feature_dim,
            item_feature_dim=self.item_feature_dim,
            num_items=self.num_items,
            hidden_units=[128, 64, 32],
            top_k=10,
        )

    def test_initialization_default(self) -> None:
        """Test model initialization with default parameters."""
        model = DeepRankingModel(
            user_feature_dim=32,
            item_feature_dim=32,
            num_items=50,
        )
        self.assertEqual(model.num_items, 50)
        self.assertEqual(model.top_k, 10)

    def test_initialization_custom(self) -> None:
        """Test model initialization with custom parameters."""
        model = DeepRankingModel(
            user_feature_dim=64,
            item_feature_dim=64,
            num_items=100,
            hidden_units=[256, 128, 64],
            dropout_rate=0.4,
            batch_norm=False,
            top_k=20,
        )
        self.assertEqual(model.user_feature_dim, 64)
        self.assertEqual(model.dropout_rate, 0.4)
        self.assertEqual(model.batch_norm, False)

    def test_invalid_user_feature_dim(self) -> None:
        """Test that invalid user_feature_dim raises error."""
        with self.assertRaises(ValueError):
            DeepRankingModel(
                user_feature_dim=-1,
                item_feature_dim=32,
                num_items=50,
            )

    def test_invalid_dropout_rate(self) -> None:
        """Test that invalid dropout_rate raises error."""
        with self.assertRaises(ValueError):
            DeepRankingModel(
                user_feature_dim=32,
                item_feature_dim=32,
                num_items=50,
                dropout_rate=2.0,
            )

    def test_forward_pass_output_shapes(self) -> None:
        """Test that forward pass produces correct output shapes."""
        user_features = tf.constant(
            np.random.randn(self.batch_size, self.user_feature_dim).astype(np.float32),
        )
        item_features = tf.constant(
            np.random.randn(
                self.batch_size,
                self.num_items,
                self.item_feature_dim,
            ).astype(np.float32),
        )

        rec_indices, rec_scores = self.model(
            [user_features, item_features],
            training=False,
        )

        self.assertEqual(rec_indices.shape, (self.batch_size, self.model.top_k))
        self.assertEqual(rec_scores.shape, (self.batch_size, self.model.top_k))

    def test_output_indices_valid(self) -> None:
        """Test that output indices are valid item indices."""
        user_features = tf.constant(
            np.random.randn(self.batch_size, self.user_feature_dim).astype(np.float32),
        )
        item_features = tf.constant(
            np.random.randn(
                self.batch_size,
                self.num_items,
                self.item_feature_dim,
            ).astype(np.float32),
        )

        rec_indices, _ = self.model([user_features, item_features], training=False)

        max_idx = tf.reduce_max(rec_indices).numpy()
        min_idx = tf.reduce_min(rec_indices).numpy()
        self.assertGreaterEqual(min_idx, 0)
        self.assertLess(max_idx, self.num_items)

    def test_output_scores_in_valid_range(self) -> None:
        """Test that output scores are in valid range [0, 1] (sigmoid)."""
        user_features = tf.constant(
            np.random.randn(self.batch_size, self.user_feature_dim).astype(np.float32),
        )
        item_features = tf.constant(
            np.random.randn(
                self.batch_size,
                self.num_items,
                self.item_feature_dim,
            ).astype(np.float32),
        )

        _, rec_scores = self.model([user_features, item_features], training=False)

        min_score = tf.reduce_min(rec_scores).numpy()
        max_score = tf.reduce_max(rec_scores).numpy()
        self.assertGreaterEqual(min_score, 0)
        self.assertLessEqual(max_score, 1)

    def test_different_batch_sizes(self) -> None:
        """Test with different batch sizes."""
        for batch_size in [1, 8, 16, 32]:
            user_features = tf.constant(
                np.random.randn(batch_size, self.user_feature_dim).astype(np.float32),
            )
            item_features = tf.constant(
                np.random.randn(
                    batch_size,
                    self.num_items,
                    self.item_feature_dim,
                ).astype(np.float32),
            )

            rec_indices, rec_scores = self.model(
                [user_features, item_features],
                training=False,
            )

            self.assertEqual(rec_indices.shape[0], batch_size)
            self.assertEqual(rec_scores.shape[0], batch_size)

    def test_serialization(self) -> None:
        """Test model serialization."""
        config = self.model.get_config()

        self.assertEqual(config["user_feature_dim"], self.user_feature_dim)
        self.assertEqual(config["item_feature_dim"], self.item_feature_dim)
        self.assertEqual(config["top_k"], 10)

    def test_deserialization(self) -> None:
        """Test model deserialization from config."""
        config = self.model.get_config()
        new_model = DeepRankingModel.from_config(config)

        self.assertEqual(new_model.user_feature_dim, self.model.user_feature_dim)
        self.assertEqual(new_model.item_feature_dim, self.model.item_feature_dim)

    def test_model_save_load(self) -> None:
        """Test model save and load."""
        import tempfile

        user_features = tf.constant(
            np.random.randn(self.batch_size, self.user_feature_dim).astype(np.float32),
        )
        item_features = tf.constant(
            np.random.randn(
                self.batch_size,
                self.num_items,
                self.item_feature_dim,
            ).astype(np.float32),
        )

        pred1_idx, pred1_scores = self.model(
            [user_features, item_features],
            training=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = f"{tmpdir}/model.keras"
            self.model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

            pred2_idx, pred2_scores = loaded_model(
                [user_features, item_features],
                training=False,
            )

            np.testing.assert_array_equal(pred1_idx.numpy(), pred2_idx.numpy())

    def test_training_vs_inference_mode(self) -> None:
        """Test that shapes are consistent between training and inference."""
        user_features = tf.constant(
            np.random.randn(self.batch_size, self.user_feature_dim).astype(np.float32),
        )
        item_features = tf.constant(
            np.random.randn(
                self.batch_size,
                self.num_items,
                self.item_feature_dim,
            ).astype(np.float32),
        )

        idx_train, scores_train = self.model(
            [user_features, item_features],
            training=True,
        )
        idx_infer, scores_infer = self.model(
            [user_features, item_features],
            training=False,
        )

        self.assertEqual(idx_train.shape, idx_infer.shape)
        self.assertEqual(scores_train.shape, scores_infer.shape)

    def test_inference_consistency(self) -> None:
        """Test that inference mode produces consistent results."""
        user_features = tf.constant(
            np.random.randn(self.batch_size, self.user_feature_dim).astype(np.float32),
        )
        item_features = tf.constant(
            np.random.randn(
                self.batch_size,
                self.num_items,
                self.item_feature_dim,
            ).astype(np.float32),
        )

        idx1, scores1 = self.model([user_features, item_features], training=False)
        idx2, scores2 = self.model([user_features, item_features], training=False)

        np.testing.assert_array_equal(idx1.numpy(), idx2.numpy())
        np.testing.assert_array_almost_equal(scores1.numpy(), scores2.numpy())


if __name__ == "__main__":
    unittest.main()
