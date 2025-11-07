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
        model = ExplainableUnifiedRecommendationModel(
            num_users=100,
            num_items=50,
            user_feature_dim=32,
            item_feature_dim=32,
        )
        self.assertEqual(model.num_users, 100)
        self.assertEqual(model.embedding_dim, 32)

    def test_custom_params(self):
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
        scores = self.model(
            [self.user_ids, self.user_features, self.item_ids, self.item_features],
            training=True,
        )
        self.assertEqual(scores.shape, (self.batch_size, 50))

    def test_inference_returns_explanations(self):
        result = self.model(
            [self.user_ids, self.user_features, self.item_ids, self.item_features],
            training=False,
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 5)
        rec_indices, rec_scores, cf_sims, cb_sims, weights = result
        self.assertEqual(rec_indices.shape, (self.batch_size, 10))
        self.assertEqual(cf_sims.shape, (self.batch_size, 50))
        self.assertEqual(weights.shape, (3,))


class TestExplainableUnifiedCompile(unittest.TestCase):
    """Test compilation."""

    def test_compile_with_loss(self):
        model = ExplainableUnifiedRecommendationModel(
            num_users=100,
            num_items=50,
            user_feature_dim=32,
            item_feature_dim=32,
        )
        model.compile(optimizer="adam", loss=ImprovedMarginRankingLoss())
        self.assertIsNotNone(model.optimizer)


class TestExplainableUnifiedTraining(unittest.TestCase):
    """Test training."""

    def setUp(self):
        self.model = ExplainableUnifiedRecommendationModel(
            num_users=100,
            num_items=50,
            user_feature_dim=32,
            item_feature_dim=32,
            embedding_dim=16,
            tower_dim=16,
            top_k=10,
        )
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=ImprovedMarginRankingLoss(),
            metrics=[AccuracyAtK(k=5, name="acc@5")],
        )
        self.batch_size = 16
        self.user_ids = np.random.randint(0, 100, self.batch_size)
        self.user_features = np.random.randn(self.batch_size, 32).astype(np.float32)
        self.item_ids = np.random.randint(0, 50, (self.batch_size, 50))
        self.item_features = np.random.randn(self.batch_size, 50, 32).astype(np.float32)
        self.labels = np.random.randint(0, 2, (self.batch_size, 50)).astype(np.float32)

    def test_fit(self):
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
        self.model = ExplainableUnifiedRecommendationModel(
            num_users=100,
            num_items=50,
            user_feature_dim=32,
            item_feature_dim=32,
            top_k=10,
        )

    def test_predict_shapes(self):
        batch_size = 8
        user_ids = np.random.randint(0, 100, batch_size)
        user_features = np.random.randn(batch_size, 32).astype(np.float32)
        item_ids = np.random.randint(0, 50, (batch_size, 50))
        item_features = np.random.randn(batch_size, 50, 32).astype(np.float32)

        rec_indices, rec_scores, cf_sims, cb_sims, weights = self.model.predict(
            [user_ids, user_features, item_ids, item_features],
        )
        self.assertEqual(rec_indices.shape, (batch_size, 10))
        self.assertEqual(cf_sims.shape, (batch_size, 50))
        self.assertEqual(weights.shape, (3,))

    def test_predict_indices_valid(self):
        batch_size = 8
        user_ids = np.random.randint(0, 100, batch_size)
        user_features = np.random.randn(batch_size, 32).astype(np.float32)
        item_ids = np.random.randint(0, 50, (batch_size, 50))
        item_features = np.random.randn(batch_size, 50, 32).astype(np.float32)

        rec_indices, _, _, _, _ = self.model.predict(
            [user_ids, user_features, item_ids, item_features],
        )
        self.assertTrue(np.all(rec_indices >= 0))
        self.assertTrue(np.all(rec_indices < 50))


class TestExplainableUnifiedSerialization(unittest.TestCase):
    """Test serialization."""

    def test_get_config(self):
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

        scores = model.compute_similarities(
            [user_ids, user_features, item_ids, item_features],
        )
        self.assertEqual(scores.shape, (1, 50))

    def test_keras_model(self):
        model = ExplainableUnifiedRecommendationModel(
            num_users=100,
            num_items=50,
            user_feature_dim=32,
            item_feature_dim=32,
        )
        self.assertIsInstance(model, keras.Model)


if __name__ == "__main__":
    unittest.main()
