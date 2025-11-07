"""Unit tests for MaxMinMarginLoss."""
import unittest

import keras
import numpy as np
import tensorflow as tf
from loguru import logger

from kmr.losses import MaxMinMarginLoss


class TestMaxMinMarginLoss(unittest.TestCase):
    """Test cases for MaxMinMarginLoss."""

    def setUp(self) -> None:
        """Set up test case."""
        self.loss = MaxMinMarginLoss(margin=1.0)

    def test_loss_initialization(self) -> None:
        """Test loss initialization."""
        logger.info("🧪 Testing MaxMinMarginLoss initialization")
        self.assertIsInstance(self.loss, MaxMinMarginLoss)
        self.assertEqual(self.loss.margin, 1.0)
        self.assertEqual(self.loss.name, "max_min_margin_loss")

    def test_loss_initialization_with_custom_params(self) -> None:
        """Test loss initialization with custom parameters."""
        logger.info("🧪 Testing MaxMinMarginLoss initialization with custom params")
        custom_loss = MaxMinMarginLoss(margin=2.0, name="custom_loss")
        self.assertEqual(custom_loss.margin, 2.0)
        self.assertEqual(custom_loss.name, "custom_loss")

    def test_loss_clear_separation(self) -> None:
        """Test loss when positive and negative scores are clearly separated."""
        logger.info("🧪 Testing MaxMinMarginLoss with clear separation")
        # Positive items: 0.9, 0.8 (max = 0.9)
        # Negative items: 0.1, 0.2 (min = 0.1)
        # margin - (0.9 - 0.1) = 1.0 - 0.8 = 0.2
        y_true = tf.constant([[1.0, 0.0, 1.0, 0.0, 0.0]], dtype=tf.float32)
        y_pred = tf.constant([[0.9, 0.1, 0.8, 0.2, 0.0]], dtype=tf.float32)

        loss_value = self.loss(y_true, y_pred)
        loss_numpy = loss_value.numpy()

        # Expected: max(0, 1.0 - 0.8) = 0.2
        self.assertAlmostEqual(loss_numpy, 0.2, places=4)
        logger.info(f"   Loss value: {loss_numpy}")

    def test_loss_overlapping_scores(self) -> None:
        """Test loss when positive and negative scores overlap."""
        logger.info("🧪 Testing MaxMinMarginLoss with overlapping scores")
        # Positive items: 0.5, 0.4 (max = 0.5)
        # Negative items: 0.6, 0.7 (min = 0.6)
        # margin - (0.5 - 0.6) = 1.0 - (-0.1) = 1.1
        y_true = tf.constant([[1.0, 0.0, 1.0, 0.0, 0.0]], dtype=tf.float32)
        y_pred = tf.constant([[0.5, 0.6, 0.4, 0.7, 0.3]], dtype=tf.float32)

        loss_value = self.loss(y_true, y_pred)
        loss_numpy = loss_value.numpy()

        # Expected: max(0, 1.0 - (-0.1)) = 1.1
        self.assertAlmostEqual(loss_numpy, 1.1, places=4)
        logger.info(f"   Loss value: {loss_numpy}")

    def test_loss_batch(self) -> None:
        """Test loss with batch of multiple users."""
        logger.info("🧪 Testing MaxMinMarginLoss with batch")
        # User 1: max_pos=0.9, min_neg=0.1, loss=max(0, 1.0-(0.9-0.1))=0.2
        # User 2: max_pos=0.8, min_neg=0.2, loss=max(0, 1.0-(0.8-0.2))=0.4
        # Mean loss = (0.2 + 0.4) / 2 = 0.3
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

        # Expected: mean([0.2, 0.4]) = 0.3
        self.assertAlmostEqual(loss_numpy, 0.3, places=4)
        logger.info(f"   Batch loss value: {loss_numpy}")

    def test_loss_all_positive(self) -> None:
        """Test loss edge case with all items positive."""
        logger.info("🧪 Testing MaxMinMarginLoss with all positive items")
        y_true = tf.constant([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=tf.float32)
        y_pred = tf.constant([[0.9, 0.8, 0.7, 0.6, 0.5]], dtype=tf.float32)

        # With all positive, min_negative becomes inf, loss becomes 0
        loss_value = self.loss(y_true, y_pred)
        loss_numpy = loss_value.numpy()

        # Should be 0 (max positive always >= min negative when all positive)
        self.assertAlmostEqual(loss_numpy, 0.0, places=4)
        logger.info(f"   Loss value (all positive): {loss_numpy}")

    def test_loss_all_negative(self) -> None:
        """Test loss edge case with all items negative."""
        logger.info("🧪 Testing MaxMinMarginLoss with all negative items")
        y_true = tf.constant([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=tf.float32)
        y_pred = tf.constant([[0.9, 0.8, 0.7, 0.6, 0.5]], dtype=tf.float32)

        # With all negative, max_positive becomes -inf, loss is large
        loss_value = self.loss(y_true, y_pred)
        loss_numpy = loss_value.numpy()

        # Should be large (margin - (-inf - max_neg) = large positive)
        self.assertGreater(loss_numpy, 100.0)
        logger.info(f"   Loss value (all negative): {loss_numpy}")

    def test_loss_gradient_flow(self) -> None:
        """Test that loss supports gradient flow."""
        logger.info("🧪 Testing MaxMinMarginLoss gradient flow")
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
        logger.info("🧪 Testing MaxMinMarginLoss serialization")
        config = self.loss.get_config()

        self.assertIn("margin", config)
        self.assertEqual(config["margin"], 1.0)

        # Recreate from config
        loss_from_config = MaxMinMarginLoss.from_config(config)
        self.assertEqual(loss_from_config.margin, self.loss.margin)
        logger.info(f"   Config: {config}")

    def test_loss_custom_margin(self) -> None:
        """Test loss with custom margin value."""
        logger.info("🧪 Testing MaxMinMarginLoss with custom margin")
        custom_loss = MaxMinMarginLoss(margin=2.0)

        y_true = tf.constant([[1.0, 0.0, 1.0, 0.0, 0.0]], dtype=tf.float32)
        y_pred = tf.constant([[0.9, 0.1, 0.8, 0.2, 0.0]], dtype=tf.float32)

        loss_value = custom_loss(y_true, y_pred)
        loss_numpy = loss_value.numpy()

        # With margin=2.0: max(0, 2.0 - 0.8) = 1.2
        self.assertAlmostEqual(loss_numpy, 1.2, places=4)
        logger.info(f"   Loss value (margin=2.0): {loss_numpy}")


if __name__ == "__main__":
    unittest.main()
