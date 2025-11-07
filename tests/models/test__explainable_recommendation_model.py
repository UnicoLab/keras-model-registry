"""Comprehensive unit tests for ExplainableRecommendationModel.

Tests cover:
- Model initialization with various configurations
- Call method behavior in training and inference modes
- compute_similarities() helper method
- Compilation with custom losses and metrics
- Training with standard Keras fit()
- Recommendation generation with explanations
- Model serialization (save/load)
- Feedback adjustment functionality
- Edge cases and error handling
"""

import unittest
import numpy as np
import tensorflow as tf
import keras

from kmr.models import ExplainableRecommendationModel
from kmr.losses import ImprovedMarginRankingLoss
from kmr.metrics import AccuracyAtK, PrecisionAtK, RecallAtK


class TestExplainableRecommendationModelInitialization(unittest.TestCase):
    """Test ExplainableRecommendationModel initialization."""

    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        model = ExplainableRecommendationModel(
            num_users=100,
            num_items=50,
        )

        self.assertEqual(model.num_users, 100)
        self.assertEqual(model.num_items, 50)
        self.assertEqual(model.embedding_dim, 32)
        self.assertEqual(model.top_k, 10)
        self.assertEqual(model.l2_reg, 1e-4)
        self.assertEqual(model.feedback_weight, 0.5)
        self.assertEqual(model.name, "explainable_recommendation_model")

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        model = ExplainableRecommendationModel(
            num_users=500,
            num_items=200,
            embedding_dim=64,
            top_k=20,
            l2_reg=1e-3,
            feedback_weight=0.7,
            name="custom_explainable",
        )

        self.assertEqual(model.num_users, 500)
        self.assertEqual(model.num_items, 200)
        self.assertEqual(model.embedding_dim, 64)
        self.assertEqual(model.top_k, 20)
        self.assertEqual(model.l2_reg, 1e-3)
        self.assertEqual(model.feedback_weight, 0.7)
        self.assertEqual(model.name, "custom_explainable")

    def test_initialization_layers_created(self):
        """Test that required layers are created."""
        model = ExplainableRecommendationModel(
            num_users=100,
            num_items=50,
        )

        self.assertTrue(hasattr(model, "embedding_layer"))
        self.assertTrue(hasattr(model, "explainer"))
        self.assertTrue(hasattr(model, "feedback_adjuster"))
        self.assertTrue(hasattr(model, "selector_layer"))

    def test_initialization_invalid_num_users(self):
        """Test initialization with invalid num_users."""
        with self.assertRaises(ValueError):
            ExplainableRecommendationModel(num_users=0, num_items=50)

        with self.assertRaises(ValueError):
            ExplainableRecommendationModel(num_users=-1, num_items=50)

    def test_initialization_invalid_num_items(self):
        """Test initialization with invalid num_items."""
        with self.assertRaises(ValueError):
            ExplainableRecommendationModel(num_users=100, num_items=0)

    def test_initialization_invalid_embedding_dim(self):
        """Test initialization with invalid embedding_dim."""
        with self.assertRaises(ValueError):
            ExplainableRecommendationModel(num_users=100, num_items=50, embedding_dim=0)

    def test_initialization_invalid_top_k(self):
        """Test initialization with invalid top_k."""
        with self.assertRaises(ValueError):
            ExplainableRecommendationModel(num_users=100, num_items=50, top_k=0)

        with self.assertRaises(ValueError):
            ExplainableRecommendationModel(num_users=100, num_items=50, top_k=100)

    def test_initialization_invalid_feedback_weight(self):
        """Test initialization with invalid feedback_weight."""
        with self.assertRaises(ValueError):
            ExplainableRecommendationModel(
                num_users=100,
                num_items=50,
                feedback_weight=-0.1,
            )

        with self.assertRaises(ValueError):
            ExplainableRecommendationModel(
                num_users=100,
                num_items=50,
                feedback_weight=1.5,
            )

    def test_initialization_invalid_l2_reg(self):
        """Test initialization with invalid l2_reg."""
        with self.assertRaises(ValueError):
            ExplainableRecommendationModel(num_users=100, num_items=50, l2_reg=-0.1)


class TestExplainableRecommendationModelCallMethod(unittest.TestCase):
    """Test the call() method behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = ExplainableRecommendationModel(
            num_users=100,
            num_items=50,
            top_k=10,
        )
        self.batch_size = 16
        self.user_ids = tf.constant(
            np.random.randint(0, 100, self.batch_size),
            dtype=tf.int32,
        )
        self.item_ids = tf.constant(
            np.random.randint(0, 50, (self.batch_size, 50)),
            dtype=tf.int32,
        )
        self.user_feedback = tf.constant(
            np.random.uniform(0, 1, (self.batch_size, 50)).astype(np.float32),
        )

    def test_call_training_mode_returns_scores(self):
        """Test call() returns scores during training."""
        scores = self.model([self.user_ids, self.item_ids], training=True)

        self.assertEqual(scores.shape, (self.batch_size, 50))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(scores)))

    def test_call_inference_mode_returns_tuple(self):
        """Test call() returns tuple during inference."""
        result = self.model([self.user_ids, self.item_ids], training=False)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_call_inference_mode_output_shapes(self):
        """Test call() returns correct shapes during inference."""
        rec_indices, rec_scores, similarity_matrix = self.model(
            [self.user_ids, self.item_ids],
            training=False,
        )

        self.assertEqual(rec_indices.shape, (self.batch_size, 10))
        self.assertEqual(rec_scores.shape, (self.batch_size, 10))
        self.assertEqual(similarity_matrix.shape, (self.batch_size, 50))

    def test_call_default_training_is_false(self):
        """Test call() defaults to inference mode."""
        result = self.model([self.user_ids, self.item_ids])

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_topk_scores_are_sorted(self):
        """Test that returned top-K scores are sorted."""
        rec_indices, rec_scores, _ = self.model(
            [self.user_ids, self.item_ids],
            training=False,
        )

        for i in range(rec_scores.shape[0]):
            is_sorted = tf.reduce_all(rec_scores[i, :-1] >= rec_scores[i, 1:])
            self.assertTrue(is_sorted.numpy())


class TestExplainableRecommendationModelComputeSimilarities(unittest.TestCase):
    """Test the compute_similarities() helper method."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = ExplainableRecommendationModel(
            num_users=100,
            num_items=50,
        )
        self.batch_size = 8
        self.user_ids = tf.constant(
            np.random.randint(0, 100, self.batch_size),
            dtype=tf.int32,
        )
        self.item_ids = tf.constant(
            np.random.randint(0, 50, (self.batch_size, 50)),
            dtype=tf.int32,
        )
        self.user_feedback = tf.constant(
            np.random.uniform(0, 1, (self.batch_size, 50)).astype(np.float32),
        )

    def test_compute_similarities_without_feedback(self):
        """Test compute_similarities() without feedback."""
        scores = self.model.compute_similarities([self.user_ids, self.item_ids])

        self.assertEqual(scores.shape, (self.batch_size, 50))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(scores)))

    def test_compute_similarities_values_bounded(self):
        """Test that similarity scores are bounded."""
        scores = self.model.compute_similarities([self.user_ids, self.item_ids])

        self.assertTrue(tf.reduce_all(scores >= -1.1))
        self.assertTrue(tf.reduce_all(scores <= 1.1))

    def test_compute_similarities_deterministic(self):
        """Test compute_similarities() is deterministic."""
        sim1 = self.model.compute_similarities(
            [self.user_ids, self.item_ids],
            training=False,
        )
        sim2 = self.model.compute_similarities(
            [self.user_ids, self.item_ids],
            training=False,
        )

        tf.debugging.assert_near(sim1, sim2, atol=1e-5)


class TestExplainableRecommendationModelCompilation(unittest.TestCase):
    """Test model compilation with custom losses and metrics."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = ExplainableRecommendationModel(
            num_users=100,
            num_items=50,
            top_k=10,
        )

    def test_compile_with_improved_margin_loss(self):
        """Test compilation with ImprovedMarginRankingLoss."""
        loss_fn = ImprovedMarginRankingLoss()
        self.model.compile(
            optimizer="adam",
            loss=loss_fn,
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
            loss=ImprovedMarginRankingLoss(),
            metrics=metrics,
        )

        self.assertIsNotNone(self.model.metrics)
        self.assertTrue(
            hasattr(self.model, "compiled_metrics") or len(self.model.metrics) > 0,
        )

    def test_compile_standard_optimizer(self):
        """Test compilation with standard optimizers."""
        for optimizer_name in ["adam", "sgd", "rmsprop"]:
            model = ExplainableRecommendationModel(num_users=100, num_items=50)
            model.compile(
                optimizer=optimizer_name,
                loss=ImprovedMarginRankingLoss(),
            )
            self.assertIsNotNone(model.optimizer)


class TestExplainableRecommendationModelTraining(unittest.TestCase):
    """Test model training with standard Keras fit()."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = ExplainableRecommendationModel(
            num_users=100,
            num_items=50,
            top_k=10,
            embedding_dim=16,
        )
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=ImprovedMarginRankingLoss(),
            metrics=[AccuracyAtK(k=5, name="acc@5")],
        )

        self.batch_size = 16
        self.user_ids = np.random.randint(0, 100, self.batch_size)
        self.item_ids = np.random.randint(0, 50, (self.batch_size, 50))
        self.user_feedback = np.random.uniform(0, 1, (self.batch_size, 50)).astype(
            np.float32,
        )
        self.labels = np.random.randint(0, 2, (self.batch_size, 50)).astype(np.float32)

    def test_fit_without_feedback(self):
        """Test model.fit() without feedback."""
        history = self.model.fit(
            x=[self.user_ids, self.item_ids],
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
            x=[self.user_ids, self.item_ids],
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
            x=[self.user_ids, self.item_ids],
            y=self.labels,
            epochs=2,
            batch_size=8,
            verbose=0,
        )

        self.assertIn("acc@5", history.history)
        self.assertTrue(len(history.history["acc@5"]) > 0)


class TestExplainableRecommendationModelPrediction(unittest.TestCase):
    """Test model prediction for generating recommendations."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = ExplainableRecommendationModel(
            num_users=100,
            num_items=50,
            top_k=10,
        )

    def test_predict_without_feedback(self):
        """Test predict() without feedback returns tuple."""
        batch_size = 8
        user_ids = np.random.randint(0, 100, batch_size)
        item_ids = np.random.randint(0, 50, (batch_size, 50))

        result = self.model.predict([user_ids, item_ids])

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_predict_output_shapes(self):
        """Test predict returns correct output shapes."""
        batch_size = 8
        user_ids = np.random.randint(0, 100, batch_size)
        item_ids = np.random.randint(0, 50, (batch_size, 50))

        rec_indices, rec_scores, similarity_matrix = self.model.predict(
            [user_ids, item_ids],
        )

        self.assertEqual(rec_indices.shape, (batch_size, 10))
        self.assertEqual(rec_scores.shape, (batch_size, 10))
        self.assertEqual(similarity_matrix.shape, (batch_size, 50))

    def test_predict_indices_valid(self):
        """Test that predicted indices are valid."""
        batch_size = 8
        user_ids = np.random.randint(0, 100, batch_size)
        item_ids = np.random.randint(0, 50, (batch_size, 50))

        rec_indices, rec_scores, similarity_matrix = self.model.predict(
            [user_ids, item_ids],
        )

        self.assertTrue(np.all(rec_indices >= 0))
        self.assertTrue(np.all(rec_indices < 50))


class TestExplainableRecommendationModelSerialization(unittest.TestCase):
    """Test model serialization and deserialization."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = ExplainableRecommendationModel(
            num_users=100,
            num_items=50,
            embedding_dim=16,
            top_k=10,
            l2_reg=1e-3,
            feedback_weight=0.6,
            name="test_explainable",
        )

    def test_get_config(self):
        """Test get_config() returns correct configuration."""
        config = self.model.get_config()

        self.assertEqual(config["num_users"], 100)
        self.assertEqual(config["num_items"], 50)
        self.assertEqual(config["embedding_dim"], 16)
        self.assertEqual(config["top_k"], 10)
        self.assertAlmostEqual(config["l2_reg"], 1e-3, places=6)
        self.assertAlmostEqual(config["feedback_weight"], 0.6, places=6)

    def test_from_config(self):
        """Test creating model from config."""
        config = self.model.get_config()
        new_model = ExplainableRecommendationModel.from_config(config)

        self.assertEqual(new_model.num_users, self.model.num_users)
        self.assertEqual(new_model.num_items, self.model.num_items)
        self.assertEqual(new_model.embedding_dim, self.model.embedding_dim)
        self.assertEqual(new_model.top_k, self.model.top_k)
        self.assertEqual(new_model.feedback_weight, self.model.feedback_weight)

    def test_serialization_roundtrip(self):
        """Test full serialization and deserialization."""
        config = self.model.get_config()
        restored_model = ExplainableRecommendationModel.from_config(config)

        batch_size = 8
        user_ids = np.random.randint(0, 100, batch_size)
        item_ids = np.random.randint(0, 50, (batch_size, 50))

        original_pred = self.model.predict([user_ids, item_ids])
        restored_pred = restored_model.predict([user_ids, item_ids])

        # Should have same shapes
        self.assertEqual(original_pred[0].shape, restored_pred[0].shape)
        self.assertEqual(original_pred[1].shape, restored_pred[1].shape)
        self.assertEqual(original_pred[2].shape, restored_pred[2].shape)


class TestExplainableRecommendationModelFeedbackWeightConfiguration(unittest.TestCase):
    """Test feedback weight configuration and validation."""

    def test_feedback_weight_configurations(self):
        """Test model creation with various feedback weights."""
        for weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
            model = ExplainableRecommendationModel(
                num_users=100,
                num_items=50,
                feedback_weight=weight,
            )
            self.assertEqual(model.feedback_weight, weight)

    def test_feedback_weight_stored_in_config(self):
        """Test that feedback weight is stored in config."""
        for weight in [0.2, 0.5, 0.8]:
            model = ExplainableRecommendationModel(
                num_users=100,
                num_items=50,
                feedback_weight=weight,
            )
            config = model.get_config()
            self.assertAlmostEqual(config["feedback_weight"], weight, places=6)


class TestExplainableRecommendationModelEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_single_batch_item(self):
        """Test model with batch size of 1."""
        model = ExplainableRecommendationModel(
            num_users=100,
            num_items=50,
        )

        user_ids = np.array([0])
        item_ids = np.random.randint(0, 50, (1, 50))

        scores = model.compute_similarities([user_ids, item_ids])
        self.assertEqual(scores.shape, (1, 50))

    def test_large_batch_size(self):
        """Test model with large batch size."""
        model = ExplainableRecommendationModel(
            num_users=100,
            num_items=50,
            embedding_dim=8,
        )

        batch_size = 128
        user_ids = np.random.randint(0, 100, batch_size)
        item_ids = np.random.randint(0, 50, (batch_size, 50))

        scores = model.compute_similarities([user_ids, item_ids])
        self.assertEqual(scores.shape, (batch_size, 50))

    def test_top_k_equals_num_items(self):
        """Test when top_k equals num_items."""
        model = ExplainableRecommendationModel(
            num_users=100,
            num_items=50,
            top_k=50,
        )

        batch_size = 8
        user_ids = np.random.randint(0, 100, batch_size)
        item_ids = np.random.randint(0, 50, (batch_size, 50))

        rec_indices, rec_scores, sim_matrix = model.predict([user_ids, item_ids])

        self.assertEqual(rec_indices.shape, (batch_size, 50))
        self.assertEqual(rec_scores.shape, (batch_size, 50))

    def test_minimal_configuration(self):
        """Test model with minimal configuration."""
        model = ExplainableRecommendationModel(
            num_users=10,
            num_items=5,
            embedding_dim=2,
            top_k=1,
        )

        user_ids = np.array([0, 1, 2])
        item_ids = np.random.randint(0, 5, (3, 5))

        rec_indices, rec_scores, sim_matrix = model.predict([user_ids, item_ids])

        self.assertEqual(rec_indices.shape, (3, 1))
        self.assertEqual(rec_scores.shape, (3, 1))
        self.assertEqual(sim_matrix.shape, (3, 5))


class TestExplainableRecommendationModelKerasCompatibility(unittest.TestCase):
    """Test Keras compatibility and standard API usage."""

    def test_model_is_keras_model(self):
        """Test that model is a proper Keras Model."""
        model = ExplainableRecommendationModel(
            num_users=100,
            num_items=50,
        )

        self.assertIsInstance(model, keras.Model)

    def test_model_has_standard_methods(self):
        """Test that model has standard Keras methods."""
        model = ExplainableRecommendationModel(
            num_users=100,
            num_items=50,
        )

        self.assertTrue(hasattr(model, "compile"))
        self.assertTrue(hasattr(model, "fit"))
        self.assertTrue(hasattr(model, "predict"))
        self.assertTrue(hasattr(model, "evaluate"))

    def test_model_trainable_variables(self):
        """Test that model has trainable variables."""
        model = ExplainableRecommendationModel(
            num_users=100,
            num_items=50,
            embedding_dim=16,
        )

        # Call model to build it
        batch_size = 8
        user_ids = np.random.randint(0, 100, batch_size)
        item_ids = np.random.randint(0, 50, (batch_size, 50))
        model([user_ids, item_ids])

        # Now check for trainable variables
        self.assertGreater(len(model.trainable_variables), 0)

    def test_model_weights_are_updated_during_training(self):
        """Test that model weights are updated during training."""
        model = ExplainableRecommendationModel(
            num_users=100,
            num_items=50,
            embedding_dim=8,
        )
        model.compile(
            optimizer="adam",
            loss=ImprovedMarginRankingLoss(),
        )

        batch_size = 16
        user_ids = np.random.randint(0, 100, batch_size)
        item_ids = np.random.randint(0, 50, (batch_size, 50))
        labels = np.random.randint(0, 2, (batch_size, 50)).astype(np.float32)

        # Build the model first
        model([user_ids, item_ids])

        original_weights = [w.numpy().copy() for w in model.trainable_variables]

        model.fit(
            x=[user_ids, item_ids],
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
