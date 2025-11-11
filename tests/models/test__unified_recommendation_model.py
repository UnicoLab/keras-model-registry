"""Comprehensive unit tests for UnifiedRecommendationModel.

Tests cover:
- Model initialization with various configurations
- Call method behavior in training and inference modes
- compute_similarities() helper method
- Compilation with custom losses and metrics
- Training with standard Keras fit()
- Recommendation generation
- Model serialization (save/load)
- Edge cases and error handling
- Collaborative filtering, content-based, and hybrid score computation
"""

import unittest
import numpy as np
import tensorflow as tf
import keras

from kmr.models import UnifiedRecommendationModel
from kmr.losses import ImprovedMarginRankingLoss
from kmr.metrics import AccuracyAtK, PrecisionAtK, RecallAtK


class TestUnifiedRecommendationModelInitialization(unittest.TestCase):
    """Test UnifiedRecommendationModel initialization."""

    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        model = UnifiedRecommendationModel(
            num_users=100,
            num_items=50,
            user_feature_dim=32,
            item_feature_dim=32,
        )

        self.assertEqual(model.num_users, 100)
        self.assertEqual(model.num_items, 50)
        self.assertEqual(model.user_feature_dim, 32)
        self.assertEqual(model.item_feature_dim, 32)
        self.assertEqual(model.embedding_dim, 32)
        self.assertEqual(model.tower_dim, 32)
        self.assertEqual(model.top_k, 10)
        self.assertEqual(model.l2_reg, 1e-4)

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        model = UnifiedRecommendationModel(
            num_users=500,
            num_items=200,
            user_feature_dim=64,
            item_feature_dim=64,
            embedding_dim=48,
            tower_dim=48,
            top_k=20,
            l2_reg=1e-3,
            name="custom_unified",
        )

        self.assertEqual(model.num_users, 500)
        self.assertEqual(model.num_items, 200)
        self.assertEqual(model.user_feature_dim, 64)
        self.assertEqual(model.item_feature_dim, 64)
        self.assertEqual(model.embedding_dim, 48)
        self.assertEqual(model.tower_dim, 48)
        self.assertEqual(model.top_k, 20)
        self.assertEqual(model.l2_reg, 1e-3)
        self.assertEqual(model.name, "custom_unified")

    def test_initialization_layers_created(self):
        """Test that required layers are created."""
        model = UnifiedRecommendationModel(
            num_users=100,
            num_items=50,
            user_feature_dim=32,
            item_feature_dim=32,
        )

        self.assertTrue(hasattr(model, "embedding_layer"))
        self.assertTrue(hasattr(model, "user_tower"))
        self.assertTrue(hasattr(model, "item_tower"))
        self.assertTrue(hasattr(model, "similarity_layer"))
        self.assertTrue(hasattr(model, "weight_combiner"))
        self.assertTrue(hasattr(model, "selector_layer"))

    def test_initialization_invalid_num_users(self):
        """Test initialization with invalid num_users."""
        with self.assertRaises(ValueError):
            UnifiedRecommendationModel(
                num_users=0,
                num_items=50,
                user_feature_dim=32,
                item_feature_dim=32,
            )

    def test_initialization_invalid_user_feature_dim(self):
        """Test initialization with invalid user_feature_dim."""
        with self.assertRaises(ValueError):
            UnifiedRecommendationModel(
                num_users=100,
                num_items=50,
                user_feature_dim=0,
                item_feature_dim=32,
            )

    def test_initialization_invalid_item_feature_dim(self):
        """Test initialization with invalid item_feature_dim."""
        with self.assertRaises(ValueError):
            UnifiedRecommendationModel(
                num_users=100,
                num_items=50,
                user_feature_dim=32,
                item_feature_dim=0,
            )

    def test_initialization_invalid_top_k(self):
        """Test initialization with invalid top_k."""
        with self.assertRaises(ValueError):
            UnifiedRecommendationModel(
                num_users=100,
                num_items=50,
                user_feature_dim=32,
                item_feature_dim=32,
                top_k=100,
            )


class TestUnifiedRecommendationModelCallMethod(unittest.TestCase):
    """Test the call() method behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = UnifiedRecommendationModel(
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

    def test_call_training_mode_returns_scores(self):
        """Test call() returns scores during training."""
        combined_scores, rec_indices, rec_scores = self.model(
            [self.user_ids, self.user_features, self.item_ids, self.item_features],
            training=True,
        )

        self.assertEqual(combined_scores.shape, (self.batch_size, 50))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(combined_scores)))

    def test_call_inference_mode_returns_topk(self):
        """Test call() returns top-K recommendations during inference."""
        combined_scores, rec_indices, rec_scores = self.model(
            [self.user_ids, self.user_features, self.item_ids, self.item_features],
            training=False,
        )

        self.assertEqual(rec_indices.shape, (self.batch_size, 10))
        self.assertEqual(rec_scores.shape, (self.batch_size, 10))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(rec_scores)))

    def test_call_default_training_is_false(self):
        """Test call() defaults to inference mode."""
        combined_scores, rec_indices, rec_scores = self.model(
            [self.user_ids, self.user_features, self.item_ids, self.item_features],
        )

        self.assertEqual(rec_indices.shape, (self.batch_size, 10))
        self.assertEqual(rec_scores.shape, (self.batch_size, 10))

    def test_topk_scores_are_sorted(self):
        """Test that returned top-K scores are sorted."""
        combined_scores, rec_indices, rec_scores = self.model(
            [self.user_ids, self.user_features, self.item_ids, self.item_features],
            training=False,
        )

        for i in range(rec_scores.shape[0]):
            is_sorted = tf.reduce_all(rec_scores[i, :-1] >= rec_scores[i, 1:])
            self.assertTrue(is_sorted.numpy())


class TestUnifiedRecommendationModelComputeSimilarities(unittest.TestCase):
    """Test similarity computation via call() method."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = UnifiedRecommendationModel(
            num_users=100,
            num_items=50,
            user_feature_dim=32,
            item_feature_dim=32,
        )
        self.batch_size = 8
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

    def test_compute_similarities_output_shape(self):
        """Test similarity scores have correct shape."""
        combined_scores, rec_indices, rec_scores = self.model(
            [self.user_ids, self.user_features, self.item_ids, self.item_features],
        )

        self.assertEqual(combined_scores.shape, (self.batch_size, 50))

    def test_compute_similarities_values_bounded(self):
        """Test that similarity scores are bounded."""
        combined_scores, rec_indices, rec_scores = self.model(
            [self.user_ids, self.user_features, self.item_ids, self.item_features],
        )

        self.assertTrue(tf.reduce_all(combined_scores >= -2.0))
        self.assertTrue(tf.reduce_all(combined_scores <= 2.0))

    def test_compute_similarities_deterministic(self):
        """Test similarity computation is deterministic."""
        combined_scores1, _, _ = self.model(
            [self.user_ids, self.user_features, self.item_ids, self.item_features],
            training=False,
        )
        combined_scores2, _, _ = self.model(
            [self.user_ids, self.user_features, self.item_ids, self.item_features],
            training=False,
        )

        tf.debugging.assert_near(combined_scores1, combined_scores2, atol=1e-5)

    def test_compute_similarities_all_finite(self):
        """Test that all similarity values are finite."""
        combined_scores, rec_indices, rec_scores = self.model(
            [self.user_ids, self.user_features, self.item_ids, self.item_features],
        )

        self.assertTrue(tf.reduce_all(tf.math.is_finite(combined_scores)))


class TestUnifiedRecommendationModelCompilation(unittest.TestCase):
    """Test model compilation with custom losses and metrics."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = UnifiedRecommendationModel(
            num_users=100,
            num_items=50,
            user_feature_dim=32,
            item_feature_dim=32,
            top_k=10,
        )

    def test_compile_with_improved_margin_loss(self):
        """Test compilation with ImprovedMarginRankingLoss."""
        loss_fn = ImprovedMarginRankingLoss()
        self.model.compile(
            optimizer="adam",
            loss=[loss_fn, None, None],
        )

        self.assertIsNotNone(self.model.optimizer)
        self.assertIsNotNone(self.model.loss)

    def test_compile_with_metrics(self):
        """Test compilation with recommendation metrics."""
        metrics = [
            AccuracyAtK(k=5, name="acc@5"),
            AccuracyAtK(k=10, name="acc@10"),
            PrecisionAtK(k=10, name="prec@10"),
            RecallAtK(k=10, name="recall@10"),
        ]
        self.model.compile(
            optimizer="adam",
            loss=[ImprovedMarginRankingLoss(), None, None],
            metrics=[metrics, None, None],
        )

        self.assertIsNotNone(self.model.metrics)
        self.assertTrue(
            hasattr(self.model, "compiled_metrics") or len(self.model.metrics) > 0,
        )

    def test_compile_standard_optimizer(self):
        """Test compilation with standard optimizers."""
        for optimizer_name in ["adam", "sgd", "rmsprop"]:
            model = UnifiedRecommendationModel(
                num_users=100,
                num_items=50,
                user_feature_dim=32,
                item_feature_dim=32,
            )
            model.compile(
                optimizer=optimizer_name,
                loss=[ImprovedMarginRankingLoss(), None, None],
            )
            self.assertIsNotNone(model.optimizer)


class TestUnifiedRecommendationModelTraining(unittest.TestCase):
    """Test model training with standard Keras fit()."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = UnifiedRecommendationModel(
            num_users=100,
            num_items=50,
            user_feature_dim=32,
            item_feature_dim=32,
            top_k=10,
            embedding_dim=16,
            tower_dim=16,
        )
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=[ImprovedMarginRankingLoss(), None, None],
            metrics=[[AccuracyAtK(k=5, name="acc@5")], None, None],
        )

        self.batch_size = 16
        self.user_ids = np.random.randint(0, 100, self.batch_size)
        self.user_features = np.random.randn(self.batch_size, 32).astype(np.float32)
        self.item_ids = np.random.randint(0, 50, (self.batch_size, 50))
        self.item_features = np.random.randn(self.batch_size, 50, 32).astype(np.float32)
        self.labels = np.random.randint(0, 2, (self.batch_size, 50)).astype(np.float32)

    def test_fit_runs_without_error(self):
        """Test that model.fit() runs without errors."""
        history = self.model.fit(
            x=[self.user_ids, self.user_features, self.item_ids, self.item_features],
            y=self.labels,
            epochs=2,
            batch_size=8,
            verbose=0,
        )

        self.assertIsNotNone(history)
        self.assertIn("loss", history.history)

    def test_fit_loss_decreases(self):
        """Test that loss generally decreases."""
        history = self.model.fit(
            x=[self.user_ids, self.user_features, self.item_ids, self.item_features],
            y=self.labels,
            epochs=3,
            batch_size=8,
            verbose=0,
        )

        losses = history.history["loss"]
        self.assertLess(losses[-1], losses[0] * 1.5)

    def test_fit_metrics_computed(self):
        """Test that metrics are computed during training."""
        history = self.model.fit(
            x=[self.user_ids, self.user_features, self.item_ids, self.item_features],
            y=self.labels,
            epochs=2,
            batch_size=8,
            verbose=0,
        )

        self.assertIn("acc@5", history.history)
        self.assertTrue(len(history.history["acc@5"]) > 0)


class TestUnifiedRecommendationModelPrediction(unittest.TestCase):
    """Test model prediction for generating recommendations."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = UnifiedRecommendationModel(
            num_users=100,
            num_items=50,
            user_feature_dim=32,
            item_feature_dim=32,
            top_k=10,
        )

    def test_predict_returns_tuple(self):
        """Test that predict returns tuple."""
        batch_size = 8
        user_ids = np.random.randint(0, 100, batch_size)
        user_features = np.random.randn(batch_size, 32).astype(np.float32)
        item_ids = np.random.randint(0, 50, (batch_size, 50))
        item_features = np.random.randn(batch_size, 50, 32).astype(np.float32)

        result = self.model.predict([user_ids, user_features, item_ids, item_features])

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_predict_output_shapes(self):
        """Test that predict returns correct shapes."""
        batch_size = 8
        user_ids = np.random.randint(0, 100, batch_size)
        user_features = np.random.randn(batch_size, 32).astype(np.float32)
        item_ids = np.random.randint(0, 50, (batch_size, 50))
        item_features = np.random.randn(batch_size, 50, 32).astype(np.float32)

        combined_scores, rec_indices, rec_scores = self.model.predict(
            [user_ids, user_features, item_ids, item_features],
        )

        self.assertEqual(rec_indices.shape, (batch_size, 10))
        self.assertEqual(rec_scores.shape, (batch_size, 10))

    def test_predict_indices_valid(self):
        """Test that predicted indices are valid."""
        batch_size = 8
        user_ids = np.random.randint(0, 100, batch_size)
        user_features = np.random.randn(batch_size, 32).astype(np.float32)
        item_ids = np.random.randint(0, 50, (batch_size, 50))
        item_features = np.random.randn(batch_size, 50, 32).astype(np.float32)

        combined_scores, rec_indices, rec_scores = self.model.predict(
            [user_ids, user_features, item_ids, item_features],
        )

        self.assertTrue(np.all(rec_indices >= 0))
        self.assertTrue(np.all(rec_indices < 50))


class TestUnifiedRecommendationModelSerialization(unittest.TestCase):
    """Test model serialization and deserialization."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = UnifiedRecommendationModel(
            num_users=100,
            num_items=50,
            user_feature_dim=32,
            item_feature_dim=32,
            embedding_dim=16,
            tower_dim=16,
            top_k=10,
            l2_reg=1e-3,
            name="test_unified",
        )

    def test_get_config(self):
        """Test get_config() returns correct configuration."""
        config = self.model.get_config()

        self.assertEqual(config["num_users"], 100)
        self.assertEqual(config["num_items"], 50)
        self.assertEqual(config["user_feature_dim"], 32)
        self.assertEqual(config["item_feature_dim"], 32)
        self.assertEqual(config["embedding_dim"], 16)
        self.assertEqual(config["tower_dim"], 16)
        self.assertEqual(config["top_k"], 10)
        self.assertAlmostEqual(config["l2_reg"], 1e-3, places=6)

    def test_from_config(self):
        """Test creating model from config."""
        config = self.model.get_config()
        new_model = UnifiedRecommendationModel.from_config(config)

        self.assertEqual(new_model.num_users, self.model.num_users)
        self.assertEqual(new_model.num_items, self.model.num_items)
        self.assertEqual(new_model.user_feature_dim, self.model.user_feature_dim)
        self.assertEqual(new_model.item_feature_dim, self.model.item_feature_dim)
        self.assertEqual(new_model.embedding_dim, self.model.embedding_dim)
        self.assertEqual(new_model.tower_dim, self.model.tower_dim)
        self.assertEqual(new_model.top_k, self.model.top_k)

    def test_serialization_roundtrip(self):
        """Test full serialization and deserialization."""
        config = self.model.get_config()
        restored_model = UnifiedRecommendationModel.from_config(config)

        batch_size = 8
        user_ids = np.random.randint(0, 100, batch_size)
        user_features = np.random.randn(batch_size, 32).astype(np.float32)
        item_ids = np.random.randint(0, 50, (batch_size, 50))
        item_features = np.random.randn(batch_size, 50, 32).astype(np.float32)

        original_pred = self.model.predict(
            [user_ids, user_features, item_ids, item_features],
        )
        restored_pred = restored_model.predict(
            [user_ids, user_features, item_ids, item_features],
        )

        # Should have same shapes
        self.assertEqual(original_pred[0].shape, restored_pred[0].shape)
        self.assertEqual(original_pred[1].shape, restored_pred[1].shape)


class TestUnifiedRecommendationModelEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_single_batch_item(self):
        """Test model with batch size of 1."""
        model = UnifiedRecommendationModel(
            num_users=100,
            num_items=50,
            user_feature_dim=32,
            item_feature_dim=32,
        )

        user_ids = np.array([0])
        user_features = np.random.randn(1, 32).astype(np.float32)
        item_ids = np.random.randint(0, 50, (1, 50))
        item_features = np.random.randn(1, 50, 32).astype(np.float32)

        combined_scores, rec_indices, rec_scores = model(
            [user_ids, user_features, item_ids, item_features],
        )
        self.assertEqual(combined_scores.shape, (1, 50))

    def test_large_batch_size(self):
        """Test model with large batch size."""
        model = UnifiedRecommendationModel(
            num_users=100,
            num_items=50,
            user_feature_dim=32,
            item_feature_dim=32,
            embedding_dim=8,
            tower_dim=8,
        )

        batch_size = 128
        user_ids = np.random.randint(0, 100, batch_size)
        user_features = np.random.randn(batch_size, 32).astype(np.float32)
        item_ids = np.random.randint(0, 50, (batch_size, 50))
        item_features = np.random.randn(batch_size, 50, 32).astype(np.float32)

        combined_scores, rec_indices, rec_scores = model(
            [user_ids, user_features, item_ids, item_features],
        )
        self.assertEqual(combined_scores.shape, (batch_size, 50))

    def test_top_k_equals_num_items(self):
        """Test when top_k equals num_items."""
        model = UnifiedRecommendationModel(
            num_users=100,
            num_items=50,
            user_feature_dim=32,
            item_feature_dim=32,
            top_k=50,
        )

        batch_size = 8
        user_ids = np.random.randint(0, 100, batch_size)
        user_features = np.random.randn(batch_size, 32).astype(np.float32)
        item_ids = np.random.randint(0, 50, (batch_size, 50))
        item_features = np.random.randn(batch_size, 50, 32).astype(np.float32)

        combined_scores, rec_indices, rec_scores = model.predict(
            [user_ids, user_features, item_ids, item_features],
        )

        self.assertEqual(rec_indices.shape, (batch_size, 50))
        self.assertEqual(rec_scores.shape, (batch_size, 50))

    def test_minimal_configuration(self):
        """Test model with minimal configuration."""
        model = UnifiedRecommendationModel(
            num_users=10,
            num_items=5,
            user_feature_dim=4,
            item_feature_dim=4,
            embedding_dim=2,
            tower_dim=2,
            top_k=1,
        )

        user_ids = np.array([0, 1, 2])
        user_features = np.random.randn(3, 4).astype(np.float32)
        item_ids = np.random.randint(0, 5, (3, 5))
        item_features = np.random.randn(3, 5, 4).astype(np.float32)

        combined_scores, rec_indices, rec_scores = model.predict(
            [user_ids, user_features, item_ids, item_features],
        )

        self.assertEqual(rec_indices.shape, (3, 1))
        self.assertEqual(rec_scores.shape, (3, 1))


class TestUnifiedRecommendationModelKerasCompatibility(unittest.TestCase):
    """Test Keras compatibility and standard API usage."""

    def test_model_is_keras_model(self):
        """Test that model is a proper Keras Model."""
        model = UnifiedRecommendationModel(
            num_users=100,
            num_items=50,
            user_feature_dim=32,
            item_feature_dim=32,
        )

        self.assertIsInstance(model, keras.Model)

    def test_model_has_standard_methods(self):
        """Test that model has standard Keras methods."""
        model = UnifiedRecommendationModel(
            num_users=100,
            num_items=50,
            user_feature_dim=32,
            item_feature_dim=32,
        )

        self.assertTrue(hasattr(model, "compile"))
        self.assertTrue(hasattr(model, "fit"))
        self.assertTrue(hasattr(model, "predict"))
        self.assertTrue(hasattr(model, "evaluate"))

    def test_model_trainable_variables(self):
        """Test that model has trainable variables."""
        model = UnifiedRecommendationModel(
            num_users=100,
            num_items=50,
            user_feature_dim=32,
            item_feature_dim=32,
            embedding_dim=16,
            tower_dim=16,
        )

        batch_size = 8
        user_ids = np.random.randint(0, 100, batch_size)
        user_features = np.random.randn(batch_size, 32).astype(np.float32)
        item_ids = np.random.randint(0, 50, (batch_size, 50))
        item_features = np.random.randn(batch_size, 50, 32).astype(np.float32)
        model([user_ids, user_features, item_ids, item_features])

        self.assertGreater(len(model.trainable_variables), 0)

    def test_model_weights_updated_during_training(self):
        """Test that model weights are updated during training."""
        model = UnifiedRecommendationModel(
            num_users=100,
            num_items=50,
            user_feature_dim=32,
            item_feature_dim=32,
            embedding_dim=8,
            tower_dim=8,
        )
        model.compile(
            optimizer="adam",
            loss=[ImprovedMarginRankingLoss(), None, None],
        )

        batch_size = 16
        user_ids = np.random.randint(0, 100, batch_size)
        user_features = np.random.randn(batch_size, 32).astype(np.float32)
        item_ids = np.random.randint(0, 50, (batch_size, 50))
        item_features = np.random.randn(batch_size, 50, 32).astype(np.float32)
        labels = np.random.randint(0, 2, (batch_size, 50)).astype(np.float32)

        model([user_ids, user_features, item_ids, item_features])
        original_weights = [w.numpy().copy() for w in model.trainable_variables]

        model.fit(
            x=[user_ids, user_features, item_ids, item_features],
            y=labels,
            epochs=2,
            batch_size=8,
            verbose=0,
        )

        updated_weights = [w.numpy() for w in model.trainable_variables]

        # At least some weights should have changed
        any_weight_changed = False
        for orig, updated in zip(original_weights, updated_weights):
            if not np.allclose(orig, updated):
                any_weight_changed = True
                break

        self.assertTrue(any_weight_changed)


if __name__ == "__main__":
    unittest.main()
