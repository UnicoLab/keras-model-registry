"""Comprehensive Keras 3 compatibility tests for TwoTowerModel."""

import unittest
from unittest.mock import MagicMock

import numpy as np
import tensorflow as tf
import keras
from keras import ops

from kmr.models.TwoTowerModel import TwoTowerModel
from kmr.losses.improved_margin_ranking_loss import ImprovedMarginRankingLoss
from kmr.metrics.accuracy_at_k import AccuracyAtK


class TestTwoTowerModelKerasCompatibility(unittest.TestCase):
    """Test Keras 3 compatibility of TwoTowerModel."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.user_feature_dim = 16
        self.item_feature_dim = 10
        self.num_users = 8
        self.num_items = 50
        self.top_k = 10
        self.batch_size = 4

        self.model = TwoTowerModel(
            user_feature_dim=self.user_feature_dim,
            item_feature_dim=self.item_feature_dim,
            num_items=self.num_items,
            hidden_units=[32, 16],
            output_dim=8,
            top_k=self.top_k,
        )

        # Create sample data
        self.user_features = tf.random.normal(
            (self.batch_size, self.user_feature_dim),
        )
        self.item_features = tf.random.normal(
            (self.batch_size, self.num_items, self.item_feature_dim),
        )

    def test_call_returns_dictionary(self) -> None:
        """Test that call() returns dictionary with all required keys."""
        output = self.model(
            [self.user_features, self.item_features],
            training=True,
        )

        # Should be dictionary with all required keys
        self.assertIsInstance(output, dict)
        self.assertIn("similarities", output)
        self.assertIn("rec_indices", output)
        self.assertIn("rec_scores", output)

        # Check shapes
        self.assertEqual(
            output["similarities"].shape,
            (self.batch_size, self.num_items),
        )
        self.assertEqual(output["rec_indices"].shape, (self.batch_size, self.top_k))
        self.assertEqual(output["rec_scores"].shape, (self.batch_size, self.top_k))

    def test_call_dictionary_consistent_across_modes(self) -> None:
        """Test that call() returns dictionary consistently for both training and inference."""
        output_train = self.model(
            [self.user_features, self.item_features],
            training=True,
        )
        output_infer = self.model(
            [self.user_features, self.item_features],
            training=False,
        )

        # Both should be dictionaries
        self.assertIsInstance(output_train, dict)
        self.assertIsInstance(output_infer, dict)

        # Same keys
        self.assertEqual(set(output_train.keys()), set(output_infer.keys()))

    def test_predict_returns_dictionary(self) -> None:
        """Test that predict() returns dictionary output."""
        # Use numpy arrays for predict
        user_feat_np = np.random.randn(
            self.batch_size,
            self.user_feature_dim,
        ).astype(np.float32)
        item_feat_np = np.random.randn(
            self.batch_size,
            self.num_items,
            self.item_feature_dim,
        ).astype(np.float32)

        # predict() uses training=False
        output = self.model.predict(
            [user_feat_np, item_feat_np],
            verbose=0,
        )

        # Should be dictionary
        self.assertIsInstance(output, dict)
        self.assertIn("similarities", output)
        self.assertIn("rec_indices", output)
        self.assertIn("rec_scores", output)

        # Shapes should be correct
        self.assertEqual(
            output["similarities"].shape,
            (self.batch_size, self.num_items),
        )
        self.assertEqual(output["rec_indices"].shape, (self.batch_size, self.top_k))
        self.assertEqual(output["rec_scores"].shape, (self.batch_size, self.top_k))

    def test_loss_computation_on_dictionary_output(self) -> None:
        """Test that loss can be computed on dictionary output."""
        loss_fn = ImprovedMarginRankingLoss(margin=1.0)

        # Create dummy labels
        y_true = tf.constant(
            np.random.randint(0, 2, (self.batch_size, self.num_items)),
            dtype=tf.float32,
        )

        # Get training output (dictionary)
        y_pred = self.model(
            [self.user_features, self.item_features],
            training=True,
        )

        # Loss should compute without errors on dictionary
        loss_value = loss_fn(y_true, y_pred)
        self.assertIsInstance(loss_value, keras.KerasTensor)
        self.assertGreater(loss_value.numpy(), 0)

    def test_compile_with_standard_keras_loss(self) -> None:
        """Test that model can be compiled with standard Keras setup."""
        model = TwoTowerModel(
            user_feature_dim=self.user_feature_dim,
            item_feature_dim=self.item_feature_dim,
            num_items=self.num_items,
            hidden_units=[32, 16],
            output_dim=8,
            top_k=self.top_k,
        )

        # Should compile without errors
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=ImprovedMarginRankingLoss(margin=1.0),
            metrics=[AccuracyAtK(k=5)],
        )

        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)

    def test_fit_with_standard_keras_training(self) -> None:
        """Test that model can be trained with standard Keras fit()."""
        model = TwoTowerModel(
            user_feature_dim=self.user_feature_dim,
            item_feature_dim=self.item_feature_dim,
            num_items=self.num_items,
            hidden_units=[32, 16],
            output_dim=8,
            top_k=self.top_k,
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss=ImprovedMarginRankingLoss(margin=1.0),
        )

        # Create small training dataset
        user_feat = np.random.randn(16, self.user_feature_dim).astype(np.float32)
        item_feat = np.random.randn(16, self.num_items, self.item_feature_dim).astype(
            np.float32,
        )
        labels = np.random.randint(0, 2, (16, self.num_items)).astype(np.float32)

        # Should train without errors
        history = model.fit(
            x=[user_feat, item_feat],
            y=labels,
            epochs=1,
            batch_size=4,
            verbose=0,
        )

        self.assertIn("loss", history.history)
        self.assertGreater(len(history.history["loss"]), 0)

    def test_evaluate_with_standard_keras(self) -> None:
        """Test that model can be evaluated with standard Keras evaluate()."""
        model = TwoTowerModel(
            user_feature_dim=self.user_feature_dim,
            item_feature_dim=self.item_feature_dim,
            num_items=self.num_items,
            hidden_units=[32, 16],
            output_dim=8,
            top_k=self.top_k,
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss=ImprovedMarginRankingLoss(margin=1.0),
        )

        # Create test data
        user_feat = np.random.randn(8, self.user_feature_dim).astype(np.float32)
        item_feat = np.random.randn(8, self.num_items, self.item_feature_dim).astype(
            np.float32,
        )
        labels = np.random.randint(0, 2, (8, self.num_items)).astype(np.float32)

        # Should evaluate without errors
        loss_value = model.evaluate(
            x=[user_feat, item_feat],
            y=labels,
            verbose=0,
        )

        self.assertIsInstance(loss_value, float)
        self.assertGreater(loss_value, 0)

    def test_training_mode_none_returns_dictionary(self) -> None:
        """Test that training=None returns dictionary output."""
        output = self.model(
            [self.user_features, self.item_features],
            training=None,
        )

        # Should return dictionary
        self.assertIsInstance(output, dict)
        self.assertIn("similarities", output)
        self.assertIn("rec_indices", output)
        self.assertIn("rec_scores", output)

    def test_output_consistency_dictionary(self) -> None:
        """Test that dictionary outputs are consistent and valid."""
        output = self.model(
            [self.user_features, self.item_features],
            training=False,
        )

        similarities = output["similarities"]
        indices = output["rec_indices"]
        scores = output["rec_scores"]

        # Similarities should be in reasonable range
        sim_np = similarities.numpy()
        self.assertLessEqual(np.abs(sim_np).max(), 10.0)

        # Indices should be valid
        indices_np = indices.numpy()
        self.assertTrue(np.all(indices_np >= 0))
        self.assertTrue(np.all(indices_np < self.num_items))

        # Scores should match selected similarities
        for b in range(self.batch_size):
            for k in range(self.top_k):
                idx = indices_np[b, k]
                score = scores.numpy()[b, k]
                sim = sim_np[b, idx]
                # Allow small floating point differences
                self.assertAlmostEqual(score, sim, places=5)

    def test_serialization_preserves_behavior(self) -> None:
        """Test that model can be serialized and deserialized."""
        # Get config
        config = self.model.get_config()
        self.assertIn("user_feature_dim", config)
        self.assertIn("item_feature_dim", config)
        self.assertIn("num_items", config)

        # Recreate from config
        reconstructed = TwoTowerModel.from_config(config)
        self.assertEqual(reconstructed.user_feature_dim, self.user_feature_dim)
        self.assertEqual(reconstructed.item_feature_dim, self.item_feature_dim)
        self.assertEqual(reconstructed.num_items, self.num_items)


class TestTwoTowerModelKerasWorkflow(unittest.TestCase):
    """Test complete Keras workflows."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.model = TwoTowerModel(
            user_feature_dim=8,
            item_feature_dim=6,
            num_items=30,
            hidden_units=[16],
            output_dim=4,
            top_k=5,
        )

    def test_full_training_workflow(self) -> None:
        """Test complete training workflow from compile to evaluate."""
        # 1. Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss=ImprovedMarginRankingLoss(margin=1.0),
            metrics=[AccuracyAtK(k=5)],
        )

        # 2. Create data
        user_feat = np.random.randn(32, 8).astype(np.float32)
        item_feat = np.random.randn(32, 30, 6).astype(np.float32)
        labels = np.random.randint(0, 2, (32, 30)).astype(np.float32)

        # 3. Train
        history = self.model.fit(
            x=[user_feat, item_feat],
            y=labels,
            epochs=2,
            batch_size=8,
            verbose=0,
        )

        # 4. Evaluate
        eval_loss = self.model.evaluate(
            x=[user_feat, item_feat],
            y=labels,
            verbose=0,
        )

        # 5. Predict
        predictions = self.model.predict([user_feat, item_feat], verbose=0)

        # Verify
        self.assertIsNotNone(history)
        self.assertGreater(eval_loss, 0)
        self.assertIsInstance(predictions, tuple)
        self.assertEqual(len(predictions), 3)


if __name__ == "__main__":
    unittest.main()
