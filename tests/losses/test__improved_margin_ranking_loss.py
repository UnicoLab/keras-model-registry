"""Unit tests for ImprovedMarginRankingLoss."""
import unittest

import keras
import numpy as np
import tensorflow as tf
from loguru import logger

from kmr.losses import ImprovedMarginRankingLoss, MaxMinMarginLoss, AverageMarginLoss


class TestImprovedMarginRankingLoss(unittest.TestCase):
    """Test cases for ImprovedMarginRankingLoss."""

    def setUp(self) -> None:
        """Set up test case."""
        self.loss = ImprovedMarginRankingLoss(
            margin=1.0,
            max_min_weight=0.7,
            avg_weight=0.3,
        )

    def test_loss_initialization(self) -> None:
        """Test loss initialization."""
        logger.info("🧪 Testing ImprovedMarginRankingLoss initialization")
        self.assertIsInstance(self.loss, ImprovedMarginRankingLoss)
        self.assertEqual(self.loss.margin, 1.0)
        self.assertEqual(self.loss.max_min_weight, 0.7)
        self.assertEqual(self.loss.avg_weight, 0.3)
        self.assertEqual(self.loss.name, "improved_margin_ranking_loss")

    def test_loss_initialization_with_custom_params(self) -> None:
        """Test loss initialization with custom parameters."""
        logger.info("🧪 Testing ImprovedMarginRankingLoss with custom params")
        custom_loss = ImprovedMarginRankingLoss(
            margin=2.0,
            max_min_weight=0.6,
            avg_weight=0.4,
            name="custom_combined_loss",
        )
        self.assertEqual(custom_loss.margin, 2.0)
        self.assertEqual(custom_loss.max_min_weight, 0.6)
        self.assertEqual(custom_loss.avg_weight, 0.4)
        self.assertEqual(custom_loss.name, "custom_combined_loss")

    def test_loss_combined_computation(self) -> None:
        """Test that combined loss correctly weights both components."""
        logger.info("🧪 Testing ImprovedMarginRankingLoss combined computation")
        y_true = tf.constant([[1.0, 0.0, 1.0, 0.0, 0.0]], dtype=tf.float32)
        y_pred = tf.constant([[0.9, 0.1, 0.8, 0.2, 0.0]], dtype=tf.float32)

        # Compute combined loss
        combined_loss_value = self.loss(y_true, y_pred)
        combined_numpy = combined_loss_value.numpy()

        # Compute component losses separately
        max_min_loss = MaxMinMarginLoss(margin=1.0)
        avg_loss = AverageMarginLoss(margin=1.0)
        max_min_value = max_min_loss(y_true, y_pred).numpy()
        avg_value = avg_loss(y_true, y_pred).numpy()

        # Expected: 0.7 * max_min + 0.3 * avg
        expected = 0.7 * max_min_value + 0.3 * avg_value

        self.assertAlmostEqual(combined_numpy, expected, places=4)
        logger.info(f"   Combined: {combined_numpy}, Expected: {expected}")

    def test_loss_weight_balance(self) -> None:
        """Test that weights balance the contribution of components."""
        logger.info("🧪 Testing ImprovedMarginRankingLoss weight balance")
        y_true = tf.constant([[1.0, 0.0, 1.0, 0.0, 0.0]], dtype=tf.float32)
        y_pred = tf.constant([[0.9, 0.1, 0.8, 0.2, 0.0]], dtype=tf.float32)

        # Test with different weight distributions
        loss_equal = ImprovedMarginRankingLoss(
            margin=1.0,
            max_min_weight=0.5,
            avg_weight=0.5,
        )
        loss_max_min_heavy = ImprovedMarginRankingLoss(
            margin=1.0,
            max_min_weight=0.9,
            avg_weight=0.1,
        )
        loss_avg_heavy = ImprovedMarginRankingLoss(
            margin=1.0,
            max_min_weight=0.1,
            avg_weight=0.9,
        )

        equal_value = loss_equal(y_true, y_pred).numpy()
        max_min_heavy_value = loss_max_min_heavy(y_true, y_pred).numpy()
        avg_heavy_value = loss_avg_heavy(y_true, y_pred).numpy()

        # They should all be different
        self.assertNotAlmostEqual(equal_value, max_min_heavy_value, places=4)
        self.assertNotAlmostEqual(equal_value, avg_heavy_value, places=4)
        logger.info(
            f"   Equal: {equal_value}, MaxMin Heavy: {max_min_heavy_value}, Avg Heavy: {avg_heavy_value}",
        )

    def test_loss_batch(self) -> None:
        """Test loss with batch of multiple users."""
        logger.info("🧪 Testing ImprovedMarginRankingLoss with batch")
        y_true = tf.constant(
            [[1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0]],
            dtype=tf.float32,
        )
        y_pred = tf.constant(
            [[0.9, 0.1, 0.8, 0.2, 0.0], [0.2, 0.8, 0.1, 0.7, 0.0]],
            dtype=tf.float32,
        )

        loss_value = self.loss(y_true, y_pred)
        loss_numpy = loss_value.numpy()

        # Should be a reasonable positive value
        self.assertGreaterEqual(loss_numpy, 0.0)
        self.assertLess(loss_numpy, 10.0)  # Shouldn't be too large
        logger.info(f"   Batch loss value: {loss_numpy}")

    def test_loss_gradient_flow(self) -> None:
        """Test that loss supports gradient flow."""
        logger.info("🧪 Testing ImprovedMarginRankingLoss gradient flow")
        y_true = tf.constant([[1.0, 0.0, 1.0, 0.0, 0.0]], dtype=tf.float32)
        y_pred = tf.Variable([[0.5, 0.5, 0.5, 0.5, 0.5]], dtype=tf.float32)

        with tf.GradientTape() as tape:
            loss_value = self.loss(y_true, y_pred)

        gradients = tape.gradient(loss_value, y_pred)

        # Gradients should exist and not be None
        self.assertIsNotNone(gradients)
        # At least some gradients should be non-zero
        self.assertTrue(tf.reduce_any(tf.abs(gradients) > 0.0))
        logger.info(f"   Gradient shape: {gradients.shape}")

    def test_loss_serialization(self) -> None:
        """Test loss serialization and deserialization."""
        logger.info("🧪 Testing ImprovedMarginRankingLoss serialization")
        config = self.loss.get_config()

        self.assertIn("margin", config)
        self.assertIn("max_min_weight", config)
        self.assertIn("avg_weight", config)
        self.assertEqual(config["margin"], 1.0)
        self.assertEqual(config["max_min_weight"], 0.7)
        self.assertEqual(config["avg_weight"], 0.3)

        # Recreate from config
        loss_from_config = ImprovedMarginRankingLoss.from_config(config)
        self.assertEqual(loss_from_config.margin, self.loss.margin)
        self.assertEqual(loss_from_config.max_min_weight, self.loss.max_min_weight)
        self.assertEqual(loss_from_config.avg_weight, self.loss.avg_weight)
        logger.info(f"   Config: {config}")

    def test_loss_weight_normalization(self) -> None:
        """Test that loss works with non-normalized weights."""
        logger.info("🧪 Testing ImprovedMarginRankingLoss with non-normalized weights")
        # Weights sum to 1.5 instead of 1.0
        custom_loss = ImprovedMarginRankingLoss(
            margin=1.0,
            max_min_weight=1.0,
            avg_weight=0.5,
        )

        y_true = tf.constant([[1.0, 0.0, 1.0, 0.0, 0.0]], dtype=tf.float32)
        y_pred = tf.constant([[0.9, 0.1, 0.8, 0.2, 0.0]], dtype=tf.float32)

        loss_value = custom_loss(y_true, y_pred)
        loss_numpy = loss_value.numpy()

        # Should work without error
        self.assertIsNotNone(loss_numpy)
        self.assertGreaterEqual(loss_numpy, 0.0)
        logger.info(f"   Non-normalized weight loss: {loss_numpy}")

    def test_loss_with_keras_model(self) -> None:
        """Test that loss can be used with a Keras model."""
        logger.info("🧪 Testing ImprovedMarginRankingLoss with Keras model")
        # Create simple model
        model = keras.Sequential(
            [
                keras.layers.Dense(32, activation="relu", input_shape=(10,)),
                keras.layers.Dense(5),  # Output 5 scores
            ],
        )

        model.compile(optimizer="adam", loss=self.loss)

        # Create dummy data
        x = np.random.randn(32, 10).astype(np.float32)
        y_true = np.random.randint(0, 2, (32, 5)).astype(np.float32)

        # Should be able to fit
        history = model.fit(x, y_true, epochs=1, verbose=0)

        self.assertIn("loss", history.history)
        logger.info(f"   Training loss: {history.history['loss'][0]}")

    def test_loss_with_tuple_input(self) -> None:
        """Test that loss handles tuple input from unified model output."""
        logger.info("🧪 Testing ImprovedMarginRankingLoss with tuple input")
        y_true = tf.constant([[1.0, 0.0, 1.0, 0.0, 0.0]], dtype=tf.float32)
        similarities = tf.constant([[0.9, 0.1, 0.8, 0.2, 0.0]], dtype=tf.float32)
        indices = tf.constant([[0, 2]], dtype=tf.int32)
        scores = tf.constant([[0.9, 0.8]], dtype=tf.float32)

        # Create tuple output format (similarities, indices, scores)
        y_pred_tuple = (similarities, indices, scores)

        # Loss should extract similarities and compute correctly
        loss_value_tuple = self.loss(y_true, y_pred_tuple)
        loss_value_direct = self.loss(y_true, similarities)

        # Both should be equivalent
        self.assertAlmostEqual(
            loss_value_tuple.numpy(),
            loss_value_direct.numpy(),
            places=5,
        )
        logger.info(
            f"   Tuple loss: {loss_value_tuple.numpy()}, Direct loss: {loss_value_direct.numpy()}",
        )

    def test_loss_backward_compatibility(self) -> None:
        """Test that loss maintains backward compatibility with raw similarities."""
        logger.info("🧪 Testing ImprovedMarginRankingLoss backward compatibility")
        y_true = tf.constant(
            [[1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0]],
            dtype=tf.float32,
        )
        y_pred_raw = tf.constant(
            [[0.9, 0.1, 0.8, 0.2, 0.0], [0.2, 0.8, 0.1, 0.7, 0.0]],
            dtype=tf.float32,
        )

        # Should work with raw similarities
        loss_value = self.loss(y_true, y_pred_raw)
        self.assertIsNotNone(loss_value)
        self.assertGreaterEqual(loss_value.numpy(), 0.0)
        logger.info(f"   Backward compatible loss: {loss_value.numpy()}")


if __name__ == "__main__":
    unittest.main()
