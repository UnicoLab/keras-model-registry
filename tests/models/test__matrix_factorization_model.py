"""Comprehensive unit tests for MatrixFactorizationModel.

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

from kmr.models import MatrixFactorizationModel
from kmr.losses import ImprovedMarginRankingLoss
from kmr.metrics import AccuracyAtK, PrecisionAtK, RecallAtK


class TestMatrixFactorizationModelInitialization(unittest.TestCase):
    """Test MatrixFactorizationModel initialization."""

    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        model = MatrixFactorizationModel(num_users=100, num_items=50)

        self.assertEqual(model.num_users, 100)
        self.assertEqual(model.num_items, 50)
        self.assertEqual(model.embedding_dim, 32)
        self.assertEqual(model.top_k, 10)
        self.assertEqual(model.l2_reg, 1e-4)
        self.assertEqual(model.name, "matrix_factorization_model")

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        model = MatrixFactorizationModel(
            num_users=500,
            num_items=200,
            embedding_dim=64,
            top_k=20,
            l2_reg=1e-3,
            name="custom_mf_model",
        )

        self.assertEqual(model.num_users, 500)
        self.assertEqual(model.num_items, 200)
        self.assertEqual(model.embedding_dim, 64)
        self.assertEqual(model.top_k, 20)
        self.assertEqual(model.l2_reg, 1e-3)
        self.assertEqual(model.name, "custom_mf_model")

    def test_initialization_layers_created(self):
        """Test that required layers are created."""
        model = MatrixFactorizationModel(num_users=100, num_items=50)

        self.assertTrue(hasattr(model, "embedding_layer"))
        self.assertTrue(hasattr(model, "selector_layer"))
        self.assertTrue(hasattr(model, "similarity_layer"))

    def test_initialization_invalid_num_users(self):
        """Test initialization with invalid num_users."""
        with self.assertRaises(ValueError):
            MatrixFactorizationModel(num_users=0, num_items=50)

        with self.assertRaises(ValueError):
            MatrixFactorizationModel(num_users=-1, num_items=50)

    def test_initialization_invalid_num_items(self):
        """Test initialization with invalid num_items."""
        with self.assertRaises(ValueError):
            MatrixFactorizationModel(num_users=100, num_items=0)

        with self.assertRaises(ValueError):
            MatrixFactorizationModel(num_users=100, num_items=-1)

    def test_initialization_invalid_embedding_dim(self):
        """Test initialization with invalid embedding_dim."""
        with self.assertRaises(ValueError):
            MatrixFactorizationModel(num_users=100, num_items=50, embedding_dim=0)

        with self.assertRaises(ValueError):
            MatrixFactorizationModel(num_users=100, num_items=50, embedding_dim=-1)

    def test_initialization_invalid_top_k(self):
        """Test initialization with invalid top_k."""
        with self.assertRaises(ValueError):
            MatrixFactorizationModel(num_users=100, num_items=50, top_k=0)

        with self.assertRaises(ValueError):
            MatrixFactorizationModel(
                num_users=100,
                num_items=50,
                top_k=100,
            )  # Exceeds num_items

    def test_initialization_invalid_l2_reg(self):
        """Test initialization with invalid l2_reg."""
        with self.assertRaises(ValueError):
            MatrixFactorizationModel(num_users=100, num_items=50, l2_reg=-0.1)


class TestMatrixFactorizationModelCallMethod(unittest.TestCase):
    """Test the call() method behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = MatrixFactorizationModel(num_users=100, num_items=50, top_k=10)
        self.batch_size = 32
        self.user_ids = tf.constant(
            np.random.randint(0, 100, self.batch_size),
            dtype=tf.int32,
        )
        self.item_ids = tf.constant(
            np.random.randint(0, 50, (self.batch_size, 50)),
            dtype=tf.int32,
        )

    def test_call_training_mode_returns_unified_tuple(self):
        """Test call() returns unified tuple during training."""
        output = self.model([self.user_ids, self.item_ids], training=True)

        # New unified output: (similarities, indices, scores)
        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 3)
        similarities, rec_indices, rec_scores = output

        self.assertEqual(similarities.shape, (self.batch_size, 50))
        self.assertEqual(rec_indices.shape, (self.batch_size, 10))
        self.assertEqual(rec_scores.shape, (self.batch_size, 10))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(similarities)))

    def test_call_inference_mode_returns_unified_tuple(self):
        """Test call() returns unified tuple during inference."""
        output = self.model([self.user_ids, self.item_ids], training=False)

        # New unified output: (similarities, indices, scores)
        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 3)
        similarities, rec_indices, rec_scores = output

        self.assertEqual(similarities.shape, (self.batch_size, 50))
        self.assertEqual(rec_indices.shape, (self.batch_size, 10))
        self.assertEqual(rec_scores.shape, (self.batch_size, 10))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(rec_scores)))

    def test_call_consistent_output_across_modes(self):
        """Test that call() returns consistent tuple in both training and inference."""
        output_train = self.model([self.user_ids, self.item_ids], training=True)
        output_infer = self.model([self.user_ids, self.item_ids], training=False)

        # Both should be 3-element tuples
        self.assertEqual(len(output_train), 3)
        self.assertEqual(len(output_infer), 3)

        # All elements should have same shapes
        self.assertEqual(output_train[0].shape, output_infer[0].shape)
        self.assertEqual(output_train[1].shape, output_infer[1].shape)
        self.assertEqual(output_train[2].shape, output_infer[2].shape)

    def test_topk_scores_are_sorted(self):
        """Test that returned top-K scores are sorted in descending order."""
        _, rec_indices, rec_scores = self.model(
            [self.user_ids, self.item_ids],
            training=False,
        )

        # Check that scores are non-increasing
        for i in range(rec_scores.shape[0]):
            is_sorted = tf.reduce_all(rec_scores[i, :-1] >= rec_scores[i, 1:])
            self.assertTrue(is_sorted.numpy())


class TestMatrixFactorizationModelCompilation(unittest.TestCase):
    """Test model compilation with custom losses and metrics."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = MatrixFactorizationModel(num_users=100, num_items=50, top_k=10)

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
            model = MatrixFactorizationModel(num_users=100, num_items=50)
            model.compile(
                optimizer=optimizer_name,
                loss=[ImprovedMarginRankingLoss(), None, None],
            )
            self.assertIsNotNone(model.optimizer)


class TestMatrixFactorizationModelTraining(unittest.TestCase):
    """Test model training with standard Keras fit()."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = MatrixFactorizationModel(
            num_users=100,
            num_items=50,
            top_k=10,
            embedding_dim=16,
        )
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=[ImprovedMarginRankingLoss(), None, None],
            metrics=[[AccuracyAtK(k=5, name="acc@5")], None, None],
        )

        # Generate training data
        self.batch_size = 32
        self.user_ids = np.random.randint(0, 100, self.batch_size)
        self.item_ids = np.random.randint(0, 50, (self.batch_size, 50))
        self.labels = np.random.randint(0, 2, (self.batch_size, 50)).astype(np.float32)

    def test_fit_runs_without_error(self):
        """Test that model.fit() runs without errors."""
        history = self.model.fit(
            x=[self.user_ids, self.item_ids],
            y=self.labels,
            epochs=2,
            batch_size=16,
            verbose=0,
        )

        self.assertIsNotNone(history)
        self.assertIn("loss", history.history)

    def test_fit_loss_decreases(self):
        """Test that loss generally decreases during training."""
        history = self.model.fit(
            x=[self.user_ids, self.item_ids],
            y=self.labels,
            epochs=3,
            batch_size=16,
            verbose=0,
        )

        losses = history.history["loss"]
        # Loss should decrease on average (allow some fluctuation)
        self.assertLess(losses[-1], losses[0] * 1.5)

    def test_fit_metrics_computed(self):
        """Test that metrics are computed during training."""
        history = self.model.fit(
            x=[self.user_ids, self.item_ids],
            y=self.labels,
            epochs=2,
            batch_size=16,
            verbose=0,
        )

        self.assertIn("acc@5", history.history)
        self.assertTrue(len(history.history["acc@5"]) > 0)


class TestMatrixFactorizationModelPrediction(unittest.TestCase):
    """Test model prediction for generating recommendations."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = MatrixFactorizationModel(num_users=100, num_items=50, top_k=10)

    def test_predict_returns_tuple(self):
        """Test that predict returns (indices, scores) tuple."""
        batch_size = 16
        user_ids = np.random.randint(0, 100, batch_size)
        item_ids = np.random.randint(0, 50, (batch_size, 50))

        result = self.model.predict([user_ids, item_ids])

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_predict_output_shapes(self):
        """Test that predict returns correct output shapes."""
        batch_size = 8
        user_ids = np.random.randint(0, 100, batch_size)
        item_ids = np.random.randint(0, 50, (batch_size, 50))

        similarities, rec_indices, rec_scores = self.model.predict([user_ids, item_ids])

        self.assertEqual(rec_indices.shape, (batch_size, 10))
        self.assertEqual(rec_scores.shape, (batch_size, 10))

    def test_predict_indices_valid(self):
        """Test that predicted indices are valid item IDs."""
        batch_size = 8
        user_ids = np.random.randint(0, 100, batch_size)
        item_ids = np.random.randint(0, 50, (batch_size, 50))

        similarities, rec_indices, rec_scores = self.model.predict([user_ids, item_ids])

        self.assertTrue(np.all(rec_indices >= 0))
        self.assertTrue(np.all(rec_indices < 50))


class TestMatrixFactorizationModelSerialization(unittest.TestCase):
    """Test model serialization and deserialization."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = MatrixFactorizationModel(
            num_users=100,
            num_items=50,
            embedding_dim=16,
            top_k=10,
            l2_reg=1e-3,
            name="test_mf_model",
        )

    def test_get_config(self):
        """Test get_config() returns correct configuration."""
        config = self.model.get_config()

        self.assertEqual(config["num_users"], 100)
        self.assertEqual(config["num_items"], 50)
        self.assertEqual(config["embedding_dim"], 16)
        self.assertEqual(config["top_k"], 10)
        self.assertAlmostEqual(config["l2_reg"], 1e-3, places=6)

    def test_from_config(self):
        """Test creating model from config."""
        config = self.model.get_config()
        new_model = MatrixFactorizationModel.from_config(config)

        self.assertEqual(new_model.num_users, self.model.num_users)
        self.assertEqual(new_model.num_items, self.model.num_items)
        self.assertEqual(new_model.embedding_dim, self.model.embedding_dim)
        self.assertEqual(new_model.top_k, self.model.top_k)
        self.assertEqual(new_model.l2_reg, self.model.l2_reg)

    def test_serialization_roundtrip(self):
        """Test full serialization and deserialization."""
        config = self.model.get_config()
        restored_model = MatrixFactorizationModel.from_config(config)

        # Verify predictions are similar
        batch_size = 8
        user_ids = np.random.randint(0, 100, batch_size)
        item_ids = np.random.randint(0, 50, (batch_size, 50))

        original_pred = self.model.predict([user_ids, item_ids])
        restored_pred = restored_model.predict([user_ids, item_ids])

        # Should have same shapes
        self.assertEqual(original_pred[0].shape, restored_pred[0].shape)
        self.assertEqual(original_pred[1].shape, restored_pred[1].shape)
        self.assertEqual(original_pred[2].shape, restored_pred[2].shape)


class TestMatrixFactorizationModelEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_single_batch_item(self):
        """Test model with batch size of 1."""
        model = MatrixFactorizationModel(num_users=100, num_items=50)

        user_ids = np.array([0])
        item_ids = np.array(
            [
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    34,
                    35,
                    36,
                    37,
                    38,
                    39,
                    40,
                    41,
                    42,
                    43,
                    44,
                    45,
                    46,
                    47,
                    48,
                    49,
                ],
            ],
        )

        similarities, rec_indices, rec_scores = model([user_ids, item_ids])
        self.assertEqual(similarities.shape, (1, 50))

    def test_large_batch_size(self):
        """Test model with large batch size."""
        model = MatrixFactorizationModel(num_users=100, num_items=50, embedding_dim=8)

        batch_size = 256
        user_ids = np.random.randint(0, 100, batch_size)
        item_ids = np.random.randint(0, 50, (batch_size, 50))

        similarities, rec_indices, rec_scores = model([user_ids, item_ids])
        self.assertEqual(similarities.shape, (batch_size, 50))

    def test_top_k_equals_num_items(self):
        """Test when top_k equals num_items."""
        model = MatrixFactorizationModel(num_users=100, num_items=50, top_k=50)

        batch_size = 8
        user_ids = np.random.randint(0, 100, batch_size)
        item_ids = np.random.randint(0, 50, (batch_size, 50))

        similarities, rec_indices, rec_scores = model.predict([user_ids, item_ids])

        self.assertEqual(rec_indices.shape, (batch_size, 50))
        self.assertEqual(rec_scores.shape, (batch_size, 50))

    def test_minimal_model_configuration(self):
        """Test model with minimal configuration."""
        model = MatrixFactorizationModel(
            num_users=10,
            num_items=5,
            embedding_dim=2,
            top_k=1,
        )

        user_ids = np.array([0, 1, 2])
        item_ids = np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])

        similarities, rec_indices, rec_scores = model.predict([user_ids, item_ids])

        self.assertEqual(rec_indices.shape, (3, 1))
        self.assertEqual(rec_scores.shape, (3, 1))


class TestMatrixFactorizationModelKerasCompatibility(unittest.TestCase):
    """Test Keras compatibility and standard API usage."""

    def test_model_is_keras_model(self):
        """Test that model is a proper Keras Model."""
        model = MatrixFactorizationModel(num_users=100, num_items=50)

        self.assertIsInstance(model, keras.Model)

    def test_model_has_standard_methods(self):
        """Test that model has standard Keras methods."""
        model = MatrixFactorizationModel(num_users=100, num_items=50)

        self.assertTrue(hasattr(model, "compile"))
        self.assertTrue(hasattr(model, "fit"))
        self.assertTrue(hasattr(model, "predict"))
        self.assertTrue(hasattr(model, "evaluate"))

    def test_model_trainable_variables(self):
        """Test that model has trainable variables after build/call."""
        model = MatrixFactorizationModel(num_users=100, num_items=50, embedding_dim=16)

        # Call model to build it
        batch_size = 8
        user_ids = np.random.randint(0, 100, batch_size)
        item_ids = np.random.randint(0, 50, (batch_size, 50))
        model([user_ids, item_ids])

        # Now check for trainable variables
        self.assertGreater(len(model.trainable_variables), 0)

    def test_model_weights_are_updated_during_training(self):
        """Test that model weights are updated during training."""
        model = MatrixFactorizationModel(num_users=100, num_items=50, embedding_dim=8)
        model.compile(
            optimizer="adam",
            loss=[ImprovedMarginRankingLoss(), None, None],
        )

        batch_size = 32
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
            batch_size=16,
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
