"""Tests for GeospatialClusteringModel."""

import unittest
import numpy as np
import tensorflow as tf
import keras
from kmr.models import GeospatialClusteringModel


class TestGeospatialClusteringModel(unittest.TestCase):
    """Test suite for GeospatialClusteringModel."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.batch_size = 16
        self.num_items = 50
        self.model = GeospatialClusteringModel(
            num_items=self.num_items,
            embedding_dim=32,
            num_clusters=8,
            top_k=10,
            threshold=0.1,
        )

    def test_initialization_default(self) -> None:
        """Test model initialization with default parameters."""
        model = GeospatialClusteringModel(num_items=100)
        self.assertEqual(model.num_items, 100)
        self.assertEqual(model.embedding_dim, 32)
        self.assertEqual(model.num_clusters, 8)
        self.assertEqual(model.top_k, 10)
        self.assertEqual(model.threshold, 0.1)

    def test_initialization_custom(self) -> None:
        """Test model initialization with custom parameters."""
        model = GeospatialClusteringModel(
            num_items=200,
            embedding_dim=64,
            num_clusters=16,
            top_k=20,
            threshold=0.2,
            entropy_weight=0.15,
            variance_weight=0.08,
        )
        self.assertEqual(model.num_items, 200)
        self.assertEqual(model.embedding_dim, 64)
        self.assertEqual(model.num_clusters, 16)
        self.assertEqual(model.top_k, 20)
        self.assertEqual(model.threshold, 0.2)

    def test_invalid_num_items(self) -> None:
        """Test that invalid num_items raises error."""
        with self.assertRaises(ValueError):
            GeospatialClusteringModel(num_items=0)

    def test_invalid_embedding_dim(self) -> None:
        """Test that invalid embedding_dim raises error."""
        with self.assertRaises(ValueError):
            GeospatialClusteringModel(num_items=100, embedding_dim=-1)

    def test_invalid_num_clusters(self) -> None:
        """Test that invalid num_clusters raises error."""
        with self.assertRaises(ValueError):
            GeospatialClusteringModel(num_items=100, num_clusters=0)

    def test_invalid_top_k(self) -> None:
        """Test that invalid top_k raises error."""
        with self.assertRaises(ValueError):
            GeospatialClusteringModel(num_items=100, top_k=101)

    def test_invalid_threshold(self) -> None:
        """Test that invalid threshold raises error."""
        with self.assertRaises(ValueError):
            GeospatialClusteringModel(num_items=100, threshold=1.5)

    def test_forward_pass_output_shapes(self) -> None:
        """Test that forward pass produces correct output shapes."""
        user_lat = tf.constant(np.random.uniform(-90, 90, (self.batch_size,)))
        user_lon = tf.constant(np.random.uniform(-180, 180, (self.batch_size,)))
        item_lats = tf.constant(
            np.random.uniform(-90, 90, (self.batch_size, self.num_items)),
        )
        item_lons = tf.constant(
            np.random.uniform(-180, 180, (self.batch_size, self.num_items)),
        )

        rec_indices, rec_scores = self.model(
            [user_lat, user_lon, item_lats, item_lons],
            training=False,
        )

        self.assertEqual(rec_indices.shape, (self.batch_size, self.model.top_k))
        self.assertEqual(rec_scores.shape, (self.batch_size, self.model.top_k))

    def test_output_indices_valid(self) -> None:
        """Test that output indices are valid item indices."""
        user_lat = tf.constant(np.random.uniform(-90, 90, (self.batch_size,)))
        user_lon = tf.constant(np.random.uniform(-180, 180, (self.batch_size,)))
        item_lats = tf.constant(
            np.random.uniform(-90, 90, (self.batch_size, self.num_items)),
        )
        item_lons = tf.constant(
            np.random.uniform(-180, 180, (self.batch_size, self.num_items)),
        )

        rec_indices, _ = self.model(
            [user_lat, user_lon, item_lats, item_lons],
            training=False,
        )

        # All indices should be within [0, num_items)
        max_idx = tf.reduce_max(rec_indices).numpy()
        min_idx = tf.reduce_min(rec_indices).numpy()
        self.assertGreaterEqual(min_idx, 0)
        self.assertLess(max_idx, self.num_items)

    def test_output_scores_non_negative(self) -> None:
        """Test that output scores are non-negative."""
        user_lat = tf.constant(np.random.uniform(-90, 90, (self.batch_size,)))
        user_lon = tf.constant(np.random.uniform(-180, 180, (self.batch_size,)))
        item_lats = tf.constant(
            np.random.uniform(-90, 90, (self.batch_size, self.num_items)),
        )
        item_lons = tf.constant(
            np.random.uniform(-180, 180, (self.batch_size, self.num_items)),
        )

        _, rec_scores = self.model(
            [user_lat, user_lon, item_lats, item_lons],
            training=False,
        )

        min_score = tf.reduce_min(rec_scores).numpy()
        self.assertGreaterEqual(min_score, 0)

    def test_inference_vs_training_mode(self) -> None:
        """Test different behavior in training vs inference mode."""
        user_lat = tf.constant(np.random.uniform(-90, 90, (self.batch_size,)))
        user_lon = tf.constant(np.random.uniform(-180, 180, (self.batch_size,)))
        item_lats = tf.constant(
            np.random.uniform(-90, 90, (self.batch_size, self.num_items)),
        )
        item_lons = tf.constant(
            np.random.uniform(-180, 180, (self.batch_size, self.num_items)),
        )

        # Training mode
        idx_train, scores_train = self.model(
            [user_lat, user_lon, item_lats, item_lons],
            training=True,
        )

        # Inference mode
        idx_infer, scores_infer = self.model(
            [user_lat, user_lon, item_lats, item_lons],
            training=False,
        )

        # Shapes should match
        self.assertEqual(idx_train.shape, idx_infer.shape)
        self.assertEqual(scores_train.shape, scores_infer.shape)

    def test_different_batch_sizes(self) -> None:
        """Test with different batch sizes."""
        for batch_size in [1, 8, 16, 32]:
            user_lat = tf.constant(np.random.uniform(-90, 90, (batch_size,)))
            user_lon = tf.constant(np.random.uniform(-180, 180, (batch_size,)))
            item_lats = tf.constant(
                np.random.uniform(-90, 90, (batch_size, self.num_items)),
            )
            item_lons = tf.constant(
                np.random.uniform(-180, 180, (batch_size, self.num_items)),
            )

            rec_indices, rec_scores = self.model(
                [user_lat, user_lon, item_lats, item_lons],
                training=False,
            )

            self.assertEqual(rec_indices.shape[0], batch_size)
            self.assertEqual(rec_scores.shape[0], batch_size)

    def test_serialization(self) -> None:
        """Test model serialization."""
        config = self.model.get_config()

        self.assertEqual(config["num_items"], self.num_items)
        self.assertEqual(config["embedding_dim"], 32)
        self.assertEqual(config["num_clusters"], 8)
        self.assertEqual(config["top_k"], 10)

    def test_deserialization(self) -> None:
        """Test model deserialization from config."""
        config = self.model.get_config()
        new_model = GeospatialClusteringModel.from_config(config)

        self.assertEqual(new_model.num_items, self.model.num_items)
        self.assertEqual(new_model.embedding_dim, self.model.embedding_dim)
        self.assertEqual(new_model.num_clusters, self.model.num_clusters)

    def test_model_save_load(self) -> None:
        """Test model save and load."""
        import tempfile

        user_lat = tf.constant(np.random.uniform(-90, 90, (self.batch_size,)))
        user_lon = tf.constant(np.random.uniform(-180, 180, (self.batch_size,)))
        item_lats = tf.constant(
            np.random.uniform(-90, 90, (self.batch_size, self.num_items)),
        )
        item_lons = tf.constant(
            np.random.uniform(-180, 180, (self.batch_size, self.num_items)),
        )

        pred1_idx, pred1_scores = self.model(
            [user_lat, user_lon, item_lats, item_lons],
            training=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = f"{tmpdir}/model.keras"
            self.model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

            pred2_idx, pred2_scores = loaded_model(
                [user_lat, user_lon, item_lats, item_lons],
                training=False,
            )

            np.testing.assert_array_equal(pred1_idx.numpy(), pred2_idx.numpy())
            np.testing.assert_array_almost_equal(
                pred1_scores.numpy(),
                pred2_scores.numpy(),
            )


if __name__ == "__main__":
    unittest.main()
