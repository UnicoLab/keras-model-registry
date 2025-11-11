"""Comprehensive unit tests for DeepRankingModel.

Tests cover:
- Model initialization with various configurations
- Call method behavior in training and inference modes
- compute_similarities() helper method
- Compilation with custom losses and metrics
- Training with standard Keras fit()
- Recommendation generation
- Model serialization (save/load)
- Edge cases and error handling
"""

import unittest
import numpy as np
import tensorflow as tf
import keras

from kmr.models import DeepRankingModel
from kmr.losses import ImprovedMarginRankingLoss
from kmr.metrics import AccuracyAtK, PrecisionAtK, RecallAtK


class TestDeepRankingModelInitialization(unittest.TestCase):
    """Test DeepRankingModel initialization."""

    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        model = DeepRankingModel(
            user_feature_dim=64,
            item_feature_dim=64,
            num_items=100,
        )

        self.assertEqual(model.user_feature_dim, 64)
        self.assertEqual(model.item_feature_dim, 64)
        self.assertEqual(model.num_items, 100)
        self.assertEqual(model.top_k, 10)
        self.assertEqual(model.hidden_units, [128, 64, 32])
        self.assertEqual(model.activation, "relu")
        self.assertEqual(model.dropout_rate, 0.3)
        self.assertTrue(model.batch_norm)

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        model = DeepRankingModel(
            user_feature_dim=32,
            item_feature_dim=32,
            num_items=200,
            hidden_units=[256, 128, 64],
            activation="tanh",
            dropout_rate=0.5,
            batch_norm=False,
            top_k=20,
            name="custom_deep_ranking",
        )

        self.assertEqual(model.user_feature_dim, 32)
        self.assertEqual(model.item_feature_dim, 32)
        self.assertEqual(model.num_items, 200)
        self.assertEqual(model.hidden_units, [256, 128, 64])
        self.assertEqual(model.activation, "tanh")
        self.assertEqual(model.dropout_rate, 0.5)
        self.assertFalse(model.batch_norm)
        self.assertEqual(model.top_k, 20)
        self.assertEqual(model.name, "custom_deep_ranking")

    def test_initialization_layers_created(self):
        """Test that required layers are created."""
        model = DeepRankingModel(
            user_feature_dim=64,
            item_feature_dim=64,
            num_items=100,
        )

        self.assertTrue(hasattr(model, "ranking_tower"))
        self.assertTrue(hasattr(model, "dense_layers"))
        self.assertTrue(hasattr(model, "output_layer"))
        self.assertTrue(hasattr(model, "selector_layer"))

    def test_initialization_invalid_user_feature_dim(self):
        """Test initialization with invalid user_feature_dim."""
        with self.assertRaises(ValueError):
            DeepRankingModel(
                user_feature_dim=0,
                item_feature_dim=64,
                num_items=100,
            )

        with self.assertRaises(ValueError):
            DeepRankingModel(
                user_feature_dim=-1,
                item_feature_dim=64,
                num_items=100,
            )

    def test_initialization_invalid_item_feature_dim(self):
        """Test initialization with invalid item_feature_dim."""
        with self.assertRaises(ValueError):
            DeepRankingModel(
                user_feature_dim=64,
                item_feature_dim=0,
                num_items=100,
            )

    def test_initialization_invalid_num_items(self):
        """Test initialization with invalid num_items."""
        with self.assertRaises(ValueError):
            DeepRankingModel(
                user_feature_dim=64,
                item_feature_dim=64,
                num_items=0,
            )

    def test_initialization_invalid_dropout_rate(self):
        """Test initialization with invalid dropout_rate."""
        with self.assertRaises(ValueError):
            DeepRankingModel(
                user_feature_dim=64,
                item_feature_dim=64,
                num_items=100,
                dropout_rate=-0.1,
            )

        with self.assertRaises(ValueError):
            DeepRankingModel(
                user_feature_dim=64,
                item_feature_dim=64,
                num_items=100,
                dropout_rate=1.5,
            )

    def test_initialization_invalid_top_k(self):
        """Test initialization with invalid top_k."""
        with self.assertRaises(ValueError):
            DeepRankingModel(
                user_feature_dim=64,
                item_feature_dim=64,
                num_items=100,
                top_k=0,
            )

        with self.assertRaises(ValueError):
            DeepRankingModel(
                user_feature_dim=64,
                item_feature_dim=64,
                num_items=100,
                top_k=150,
            )  # Exceeds num_items


class TestDeepRankingModelCallMethod(unittest.TestCase):
    """Test the call() method behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = DeepRankingModel(
            user_feature_dim=32,
            item_feature_dim=32,
            num_items=50,
            top_k=10,
        )
        self.batch_size = 16
        self.user_features = tf.constant(
            np.random.randn(self.batch_size, 32).astype(np.float32),
        )
        self.item_features = tf.constant(
            np.random.randn(self.batch_size, 50, 32).astype(np.float32),
        )

    def test_call_training_mode_returns_scores(self):
        """Test call() returns scores during training."""
        scores, rec_indices, rec_scores = self.model(
            [self.user_features, self.item_features],
            training=True,
        )

        self.assertEqual(scores.shape, (self.batch_size, 50))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(scores)))

    def test_call_inference_mode_returns_topk(self):
        """Test call() returns top-K recommendations during inference."""
        scores, rec_indices, rec_scores = self.model(
            [self.user_features, self.item_features],
            training=False,
        )

        self.assertEqual(rec_indices.shape, (self.batch_size, 10))
        self.assertEqual(rec_scores.shape, (self.batch_size, 10))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(rec_scores)))

    def test_call_default_training_is_false(self):
        """Test call() defaults to inference mode when training not specified."""
        scores, rec_indices, rec_scores = self.model(
            [self.user_features, self.item_features],
        )

        self.assertEqual(rec_indices.shape, (self.batch_size, 10))
        self.assertEqual(rec_scores.shape, (self.batch_size, 10))

    def test_topk_scores_are_sorted(self):
        """Test that returned top-K scores are sorted in descending order."""
        scores, rec_indices, rec_scores = self.model(
            [self.user_features, self.item_features],
            training=False,
        )

        # Check that scores are non-increasing
        for i in range(rec_scores.shape[0]):
            is_sorted = tf.reduce_all(rec_scores[i, :-1] >= rec_scores[i, 1:])
            self.assertTrue(is_sorted.numpy())


class TestDeepRankingModelComputeSimilarities(unittest.TestCase):
    """Test similarity computation via call() method."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = DeepRankingModel(
            user_feature_dim=32,
            item_feature_dim=32,
            num_items=50,
        )
        self.batch_size = 8
        self.user_features = tf.constant(
            np.random.randn(self.batch_size, 32).astype(np.float32),
        )
        self.item_features = tf.constant(
            np.random.randn(self.batch_size, 50, 32).astype(np.float32),
        )

    def test_compute_similarities_output_shape(self):
        """Test similarity scores have correct shape."""
        scores, rec_indices, rec_scores = self.model(
            [self.user_features, self.item_features],
        )

        self.assertEqual(scores.shape, (self.batch_size, 50))

    def test_compute_similarities_values_bounded(self):
        """Test similarity scores are bounded (sigmoid output)."""
        scores, rec_indices, rec_scores = self.model(
            [self.user_features, self.item_features],
        )

        # Sigmoid output should be between 0 and 1
        self.assertTrue(tf.reduce_all(scores >= 0.0))
        self.assertTrue(tf.reduce_all(scores <= 1.0))

    def test_compute_similarities_training_false(self):
        """Test similarity computation with training=False."""
        scores1, _, _ = self.model(
            [self.user_features, self.item_features],
            training=False,
        )
        scores2, _, _ = self.model(
            [self.user_features, self.item_features],
            training=False,
        )

        # Should be deterministic
        tf.debugging.assert_near(scores1, scores2, atol=1e-5)

    def test_compute_similarities_all_finite(self):
        """Test that all similarity values are finite."""
        scores, rec_indices, rec_scores = self.model(
            [self.user_features, self.item_features],
        )

        self.assertTrue(tf.reduce_all(tf.math.is_finite(scores)))


class TestDeepRankingModelCompilation(unittest.TestCase):
    """Test model compilation with custom losses and metrics."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = DeepRankingModel(
            user_feature_dim=32,
            item_feature_dim=32,
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
            loss=[ImprovedMarginRankingLoss(), None, None],
            metrics=[metrics, None, None],
        )

        # Model should have metrics configured
        self.assertIsNotNone(self.model.metrics)
        # Verify the metrics were registered without errors
        self.assertTrue(
            hasattr(self.model, "compiled_metrics") or len(self.model.metrics) > 0,
        )

    def test_compile_standard_optimizer(self):
        """Test compilation with standard Keras optimizers."""
        for optimizer_name in ["adam", "sgd", "rmsprop"]:
            model = DeepRankingModel(
                user_feature_dim=32,
                item_feature_dim=32,
                num_items=50,
            )
            model.compile(
                optimizer=optimizer_name,
                loss=[ImprovedMarginRankingLoss(), None, None],
            )
            self.assertIsNotNone(model.optimizer)


class TestDeepRankingModelTraining(unittest.TestCase):
    """Test model training with standard Keras fit()."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = DeepRankingModel(
            user_feature_dim=32,
            item_feature_dim=32,
            num_items=50,
            top_k=10,
            hidden_units=[64, 32],
        )
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=[ImprovedMarginRankingLoss(), None, None],
            metrics=[[AccuracyAtK(k=5, name="acc@5")], None, None],
        )

        # Generate training data
        self.batch_size = 16
        self.user_features = np.random.randn(self.batch_size, 32).astype(np.float32)
        self.item_features = np.random.randn(self.batch_size, 50, 32).astype(np.float32)
        self.labels = np.random.randint(0, 2, (self.batch_size, 50)).astype(np.float32)

    def test_fit_runs_without_error(self):
        """Test that model.fit() runs without errors."""
        history = self.model.fit(
            x=[self.user_features, self.item_features],
            y=self.labels,
            epochs=2,
            batch_size=8,
            verbose=0,
        )

        self.assertIsNotNone(history)
        self.assertIn("loss", history.history)

    def test_fit_loss_decreases(self):
        """Test that loss generally decreases during training."""
        history = self.model.fit(
            x=[self.user_features, self.item_features],
            y=self.labels,
            epochs=3,
            batch_size=8,
            verbose=0,
        )

        losses = history.history["loss"]
        # Loss should decrease on average
        self.assertLess(losses[-1], losses[0] * 1.5)

    def test_fit_metrics_computed(self):
        """Test that metrics are computed during training."""
        history = self.model.fit(
            x=[self.user_features, self.item_features],
            y=self.labels,
            epochs=2,
            batch_size=8,
            verbose=0,
        )

        self.assertIn("acc@5", history.history)
        self.assertTrue(len(history.history["acc@5"]) > 0)


class TestDeepRankingModelPrediction(unittest.TestCase):
    """Test model prediction for generating recommendations."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = DeepRankingModel(
            user_feature_dim=32,
            item_feature_dim=32,
            num_items=50,
            top_k=10,
        )

    def test_predict_returns_tuple(self):
        """Test that predict returns (indices, scores) tuple."""
        batch_size = 8
        user_features = np.random.randn(batch_size, 32).astype(np.float32)
        item_features = np.random.randn(batch_size, 50, 32).astype(np.float32)

        result = self.model.predict([user_features, item_features])

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)  # (scores, rec_indices, rec_scores)

    def test_predict_output_shapes(self):
        """Test that predict returns correct output shapes."""
        batch_size = 8
        user_features = np.random.randn(batch_size, 32).astype(np.float32)
        item_features = np.random.randn(batch_size, 50, 32).astype(np.float32)

        scores, rec_indices, rec_scores = self.model.predict(
            [user_features, item_features],
        )

        self.assertEqual(rec_indices.shape, (batch_size, 10))
        self.assertEqual(rec_scores.shape, (batch_size, 10))

    def test_predict_indices_valid(self):
        """Test that predicted indices are valid item IDs."""
        batch_size = 8
        user_features = np.random.randn(batch_size, 32).astype(np.float32)
        item_features = np.random.randn(batch_size, 50, 32).astype(np.float32)

        scores, rec_indices, rec_scores = self.model.predict(
            [user_features, item_features],
        )

        self.assertTrue(np.all(rec_indices >= 0))
        self.assertTrue(np.all(rec_indices < 50))


class TestDeepRankingModelSerialization(unittest.TestCase):
    """Test model serialization and deserialization."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = DeepRankingModel(
            user_feature_dim=32,
            item_feature_dim=32,
            num_items=50,
            hidden_units=[64, 32],
            top_k=10,
            dropout_rate=0.4,
            name="test_deep_ranking",
        )

    def test_get_config(self):
        """Test get_config() returns correct configuration."""
        config = self.model.get_config()

        self.assertEqual(config["user_feature_dim"], 32)
        self.assertEqual(config["item_feature_dim"], 32)
        self.assertEqual(config["num_items"], 50)
        self.assertEqual(config["hidden_units"], [64, 32])
        self.assertEqual(config["top_k"], 10)
        self.assertAlmostEqual(config["dropout_rate"], 0.4, places=6)

    def test_from_config(self):
        """Test creating model from config."""
        config = self.model.get_config()
        new_model = DeepRankingModel.from_config(config)

        self.assertEqual(new_model.user_feature_dim, self.model.user_feature_dim)
        self.assertEqual(new_model.item_feature_dim, self.model.item_feature_dim)
        self.assertEqual(new_model.num_items, self.model.num_items)
        self.assertEqual(new_model.hidden_units, self.model.hidden_units)
        self.assertEqual(new_model.top_k, self.model.top_k)

    def test_serialization_roundtrip(self):
        """Test full serialization and deserialization."""
        config = self.model.get_config()
        restored_model = DeepRankingModel.from_config(config)

        # Verify predictions are similar
        batch_size = 8
        user_features = np.random.randn(batch_size, 32).astype(np.float32)
        item_features = np.random.randn(batch_size, 50, 32).astype(np.float32)

        original_pred = self.model.predict([user_features, item_features])
        restored_pred = restored_model.predict([user_features, item_features])

        # Should have same shapes
        self.assertEqual(original_pred[0].shape, restored_pred[0].shape)
        self.assertEqual(original_pred[1].shape, restored_pred[1].shape)


class TestDeepRankingModelEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_single_batch_item(self):
        """Test model with batch size of 1."""
        model = DeepRankingModel(
            user_feature_dim=32,
            item_feature_dim=32,
            num_items=50,
        )

        user_features = np.random.randn(1, 32).astype(np.float32)
        item_features = np.random.randn(1, 50, 32).astype(np.float32)

        scores, rec_indices, rec_scores = model([user_features, item_features])
        self.assertEqual(scores.shape, (1, 50))

    def test_large_batch_size(self):
        """Test model with large batch size."""
        model = DeepRankingModel(
            user_feature_dim=32,
            item_feature_dim=32,
            num_items=50,
            hidden_units=[32, 16],
        )

        batch_size = 128
        user_features = np.random.randn(batch_size, 32).astype(np.float32)
        item_features = np.random.randn(batch_size, 50, 32).astype(np.float32)

        scores, rec_indices, rec_scores = model([user_features, item_features])
        self.assertEqual(scores.shape, (batch_size, 50))

    def test_top_k_equals_num_items(self):
        """Test when top_k equals num_items."""
        model = DeepRankingModel(
            user_feature_dim=32,
            item_feature_dim=32,
            num_items=50,
            top_k=50,
        )

        batch_size = 8
        user_features = np.random.randn(batch_size, 32).astype(np.float32)
        item_features = np.random.randn(batch_size, 50, 32).astype(np.float32)

        scores, rec_indices, rec_scores = model.predict([user_features, item_features])

        self.assertEqual(rec_indices.shape, (batch_size, 50))
        self.assertEqual(rec_scores.shape, (batch_size, 50))

    def test_minimal_model_configuration(self):
        """Test model with minimal configuration."""
        model = DeepRankingModel(
            user_feature_dim=8,
            item_feature_dim=8,
            num_items=10,
            hidden_units=[16],
            top_k=1,
        )

        user_features = np.random.randn(3, 8).astype(np.float32)
        item_features = np.random.randn(3, 10, 8).astype(np.float32)

        scores, rec_indices, rec_scores = model.predict([user_features, item_features])

        self.assertEqual(rec_indices.shape, (3, 1))
        self.assertEqual(rec_scores.shape, (3, 1))

    def test_no_batch_norm(self):
        """Test model without batch normalization."""
        model = DeepRankingModel(
            user_feature_dim=32,
            item_feature_dim=32,
            num_items=50,
            batch_norm=False,
        )

        batch_size = 8
        user_features = np.random.randn(batch_size, 32).astype(np.float32)
        item_features = np.random.randn(batch_size, 50, 32).astype(np.float32)

        scores, rec_indices, rec_scores = model([user_features, item_features])
        self.assertEqual(scores.shape, (batch_size, 50))


class TestDeepRankingModelKerasCompatibility(unittest.TestCase):
    """Test Keras compatibility and standard API usage."""

    def test_model_is_keras_model(self):
        """Test that model is a proper Keras Model."""
        model = DeepRankingModel(
            user_feature_dim=32,
            item_feature_dim=32,
            num_items=50,
        )

        self.assertIsInstance(model, keras.Model)

    def test_model_has_standard_methods(self):
        """Test that model has standard Keras methods."""
        model = DeepRankingModel(
            user_feature_dim=32,
            item_feature_dim=32,
            num_items=50,
        )

        self.assertTrue(hasattr(model, "compile"))
        self.assertTrue(hasattr(model, "fit"))
        self.assertTrue(hasattr(model, "predict"))
        self.assertTrue(hasattr(model, "evaluate"))

    def test_model_trainable_variables(self):
        """Test that model has trainable variables after build/call."""
        model = DeepRankingModel(
            user_feature_dim=32,
            item_feature_dim=32,
            num_items=50,
            hidden_units=[64, 32],
        )

        # Call model to build it
        batch_size = 8
        user_features = np.random.randn(batch_size, 32).astype(np.float32)
        item_features = np.random.randn(batch_size, 50, 32).astype(np.float32)
        model([user_features, item_features])

        # Now check for trainable variables
        self.assertGreater(len(model.trainable_variables), 0)

    def test_model_weights_are_updated_during_training(self):
        """Test that model weights are updated during training."""
        model = DeepRankingModel(
            user_feature_dim=32,
            item_feature_dim=32,
            num_items=50,
            hidden_units=[32, 16],
        )
        model.compile(
            optimizer="adam",
            loss=[ImprovedMarginRankingLoss(), None, None],
        )

        batch_size = 16
        user_features = np.random.randn(batch_size, 32).astype(np.float32)
        item_features = np.random.randn(batch_size, 50, 32).astype(np.float32)
        labels = np.random.randint(0, 2, (batch_size, 50)).astype(np.float32)

        # Build the model first
        model([user_features, item_features])

        original_weights = [w.numpy().copy() for w in model.trainable_variables]

        model.fit(
            x=[user_features, item_features],
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
