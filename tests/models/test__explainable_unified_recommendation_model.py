"""Comprehensive unit tests for ExplainableUnifiedRecommendationModel."""

import unittest
import numpy as np
import tensorflow as tf
import keras

from kmr.models import ExplainableUnifiedRecommendationModel
from kmr.losses import ImprovedMarginRankingLoss
from kmr.metrics import AccuracyAtK


class TestExplainableUnifiedInit(unittest.TestCase):
    """Test initialization."""

    def test_default_params(self):
        """Test model initialization with default parameters."""
        model = ExplainableUnifiedRecommendationModel(
            num_users=100,
            num_items=50,
            user_feature_dim=32,
            item_feature_dim=32,
        )
        self.assertEqual(model.num_users, 100)
        self.assertEqual(model.embedding_dim, 32)

    def test_custom_params(self):
        """Test model initialization with custom parameters."""
        model = ExplainableUnifiedRecommendationModel(
            num_users=500,
            num_items=200,
            user_feature_dim=64,
            item_feature_dim=64,
            embedding_dim=48,
            tower_dim=48,
            top_k=20,
        )
        self.assertEqual(model.num_users, 500)
        self.assertEqual(model.top_k, 20)

    def test_invalid_params(self):
        """Test model initialization with invalid parameters raises error."""
        with self.assertRaises(ValueError):
            ExplainableUnifiedRecommendationModel(
                num_users=0,
                num_items=50,
                user_feature_dim=32,
                item_feature_dim=32,
            )


class TestExplainableUnifiedCall(unittest.TestCase):
    """Test call method."""

    def setUp(self):
        """Set up test fixtures for call method tests."""
        self.model = ExplainableUnifiedRecommendationModel(
            num_users=100,
            num_items=50,
            user_feature_dim=32,
            item_feature_dim=32,
            top_k=10,
        )
        self.batch_size = 16
        self.user_ids = tf.constant(
            np.random.randint(0, 100, self.batch_size),
            dtype=tf.int32,
        )
        self.user_features = tf.constant(
            np.random.randn(self.batch_size, 32).astype(np.float32),
        )
        self.item_ids = tf.constant(
            np.random.randint(0, 50, (self.batch_size, 50)),
            dtype=tf.int32,
        )
        self.item_features = tf.constant(
            np.random.randn(self.batch_size, 50, 32).astype(np.float32),
        )

    def test_training_returns_scores(self):
        """Test that training mode returns combined scores."""
        (
            combined_scores,
            rec_indices,
            rec_scores,
            cf_similarities,
            cb_similarities,
            weights,
            raw_cf_scores,
        ) = self.model(
            [self.user_ids, self.user_features, self.item_ids, self.item_features],
            training=True,
        )
        self.assertEqual(combined_scores.shape, (self.batch_size, 50))

    def test_inference_returns_explanations(self):
        """Test that inference mode returns explanations tuple."""
        result = self.model(
            [self.user_ids, self.user_features, self.item_ids, self.item_features],
            training=False,
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 7)
        (
            combined_scores,
            rec_indices,
            rec_scores,
            cf_sims,
            cb_sims,
            weights,
            raw_cf_scores,
        ) = result
        self.assertEqual(rec_indices.shape, (self.batch_size, 10))
        self.assertEqual(cf_sims.shape, (self.batch_size, 50))
        self.assertEqual(len(weights), 2)  # weights is a list of 2 tensors


class TestExplainableUnifiedCompile(unittest.TestCase):
    """Test compilation."""

    def test_compile_with_loss(self):
        """Test model compilation with loss function."""
        model = ExplainableUnifiedRecommendationModel(
            num_users=100,
            num_items=50,
            user_feature_dim=32,
            item_feature_dim=32,
        )
        # For models with 7 outputs, use list format but only provide loss for first output
        # Keras will handle the tuple output correctly
        model.compile(optimizer="adam", loss=ImprovedMarginRankingLoss())
        self.assertIsNotNone(model.optimizer)


class TestExplainableUnifiedTraining(unittest.TestCase):
    """Test training."""

    def setUp(self):
        """Set up test fixtures for training tests."""
        self.model = ExplainableUnifiedRecommendationModel(
            num_users=100,
            num_items=50,
            user_feature_dim=32,
            item_feature_dim=32,
            embedding_dim=16,
            tower_dim=16,
            top_k=10,
        )
        self.batch_size = 16
        self.user_ids = np.random.randint(0, 100, self.batch_size)
        self.user_features = np.random.randn(self.batch_size, 32).astype(np.float32)
        self.item_ids = np.random.randint(0, 50, (self.batch_size, 50))
        self.item_features = np.random.randn(self.batch_size, 50, 32).astype(np.float32)
        self.labels = np.random.randint(0, 2, (self.batch_size, 50)).astype(np.float32)

        # Build model by calling it first (like e2e test)
        _ = self.model(
            [self.user_ids, self.user_features, self.item_ids, self.item_features],
        )

        # Use exact same format as e2e test - 7 outputs with list mapping
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=[ImprovedMarginRankingLoss(), None, None, None, None, None, None],
            metrics=[
                [AccuracyAtK(k=5)],
                None,
                None,
                None,
                None,
                None,
                None,
            ],
        )

    def test_fit(self):
        """Test model training with fit method."""
        history = self.model.fit(
            x=[self.user_ids, self.user_features, self.item_ids, self.item_features],
            y=self.labels,
            epochs=2,
            batch_size=8,
            verbose=0,
        )
        self.assertIn("loss", history.history)


class TestExplainableUnifiedPredict(unittest.TestCase):
    """Test prediction."""

    def setUp(self):
        """Set up test fixtures for prediction tests."""
        self.model = ExplainableUnifiedRecommendationModel(
            num_users=100,
            num_items=50,
            user_feature_dim=32,
            item_feature_dim=32,
            top_k=10,
        )

    def test_predict_shapes(self):
        """Test that predict returns correct output shapes."""
        batch_size = 8
        user_ids = np.random.randint(0, 100, batch_size)
        user_features = np.random.randn(batch_size, 32).astype(np.float32)
        item_ids = np.random.randint(0, 50, (batch_size, 50))
        item_features = np.random.randn(batch_size, 50, 32).astype(np.float32)

        (
            combined_scores,
            rec_indices,
            rec_scores,
            cf_sims,
            cb_sims,
            weights,
            raw_cf_scores,
        ) = self.model.predict(
            [user_ids, user_features, item_ids, item_features],
        )
        self.assertEqual(rec_indices.shape, (batch_size, 10))
        self.assertEqual(cf_sims.shape, (batch_size, 50))
        self.assertEqual(len(weights), 2)  # weights is a list of 2 tensors

    def test_predict_indices_valid(self):
        """Test that predicted indices are within valid range."""
        batch_size = 8
        user_ids = np.random.randint(0, 100, batch_size)
        user_features = np.random.randn(batch_size, 32).astype(np.float32)
        item_ids = np.random.randint(0, 50, (batch_size, 50))
        item_features = np.random.randn(batch_size, 50, 32).astype(np.float32)

        (
            combined_scores,
            rec_indices,
            rec_scores,
            cf_sims,
            cb_sims,
            weights,
            raw_cf_scores,
        ) = self.model.predict(
            [user_ids, user_features, item_ids, item_features],
        )
        self.assertTrue(np.all(rec_indices >= 0))
        self.assertTrue(np.all(rec_indices < 50))


class TestExplainableUnifiedSerialization(unittest.TestCase):
    """Test serialization."""

    def test_get_config(self):
        """Test model configuration retrieval."""
        model = ExplainableUnifiedRecommendationModel(
            num_users=100,
            num_items=50,
            user_feature_dim=32,
            item_feature_dim=32,
            embedding_dim=16,
            tower_dim=16,
            top_k=10,
        )
        config = model.get_config()
        self.assertEqual(config["num_users"], 100)
        self.assertEqual(config["embedding_dim"], 16)


class TestExplainableUnifiedEdgeCases(unittest.TestCase):
    """Test edge cases."""

    def test_single_batch(self):
        """Test model with single batch size."""
        model = ExplainableUnifiedRecommendationModel(
            num_users=100,
            num_items=50,
            user_feature_dim=32,
            item_feature_dim=32,
        )
        user_ids = np.array([0])
        user_features = np.random.randn(1, 32).astype(np.float32)
        item_ids = np.random.randint(0, 50, (1, 50))
        item_features = np.random.randn(1, 50, 32).astype(np.float32)

        (
            combined_scores,
            rec_indices,
            rec_scores,
            cf_similarities,
            cb_similarities,
            weights,
            raw_cf_scores,
        ) = model(
            [user_ids, user_features, item_ids, item_features],
        )
        self.assertEqual(combined_scores.shape, (1, 50))

    def test_keras_model(self):
        """Test that model is an instance of keras.Model."""
        model = ExplainableUnifiedRecommendationModel(
            num_users=100,
            num_items=50,
            user_feature_dim=32,
            item_feature_dim=32,
        )
        self.assertIsInstance(model, keras.Model)


if __name__ == "__main__":
    unittest.main()
