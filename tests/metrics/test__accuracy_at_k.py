"""Unit tests for AccuracyAtK metric."""
import unittest

import keras
import numpy as np
import tensorflow as tf
from loguru import logger

from kmr.metrics import AccuracyAtK


class TestAccuracyAtK(unittest.TestCase):
    """Test cases for AccuracyAtK metric."""

    def setUp(self) -> None:
        """Set up test case."""
        self.metric = AccuracyAtK(k=5)

    def test_metric_initialization(self) -> None:
        """Test metric initialization."""
        logger.info("🧪 Testing AccuracyAtK initialization")
        self.assertIsInstance(self.metric, AccuracyAtK)
        self.assertEqual(self.metric.name, "accuracy_at_k")
        self.assertEqual(self.metric.k, 5)

    def test_metric_initialization_with_custom_name(self) -> None:
        """Test metric initialization with custom name."""
        logger.info("🧪 Testing AccuracyAtK initialization with custom name")
        custom_metric = AccuracyAtK(k=10, name="custom_acc@10")
        self.assertEqual(custom_metric.name, "custom_acc@10")
        self.assertEqual(custom_metric.k, 10)

    def test_metric_update_state_basic(self) -> None:
        """Test metric update state with basic case."""
        logger.info("🧪 Testing AccuracyAtK update_state - basic case")

        # y_true: items 0 and 2 are positive
        # y_pred: top-5 are [0, 1, 3, 4, 5] - item 0 is in top-5
        y_true = tf.constant([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred = tf.constant([[0, 1, 3, 4, 5]], dtype=tf.int32)

        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()

        # Should be 1.0 (item 0 is in top-5)
        self.assertAlmostEqual(result.numpy(), 1.0, places=4)

    def test_metric_update_state_no_hit(self) -> None:
        """Test metric when no positive item is in top-K."""
        logger.info("🧪 Testing AccuracyAtK update_state - no hit")

        # y_true: items 0 and 2 are positive
        # y_pred: top-5 are [1, 3, 4, 5, 6] - no positive items
        y_true = tf.constant([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred = tf.constant([[1, 3, 4, 5, 6]], dtype=tf.int32)

        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()

        # Should be 0.0 (no positive items in top-5)
        self.assertAlmostEqual(result.numpy(), 0.0, places=4)

    def test_metric_update_state_multiple_batches(self) -> None:
        """Test metric update state with multiple batches."""
        logger.info("🧪 Testing AccuracyAtK update_state - multiple batches")

        # Batch 1: has hit
        y_true_1 = tf.constant([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred_1 = tf.constant([[0, 1, 3, 4, 5]], dtype=tf.int32)

        # Batch 2: no hit
        y_true_2 = tf.constant([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred_2 = tf.constant([[1, 3, 4, 5, 6]], dtype=tf.int32)

        self.metric.update_state(y_true_1, y_pred_1)
        self.metric.update_state(y_true_2, y_pred_2)

        result = self.metric.result()
        # Average: (1.0 + 0.0) / 2 = 0.5
        self.assertAlmostEqual(result.numpy(), 0.5, places=4)

    def test_metric_update_state_multiple_users(self) -> None:
        """Test metric with multiple users in batch."""
        logger.info("🧪 Testing AccuracyAtK update_state - multiple users")

        # User 1: has hit (item 0 in top-5)
        # User 2: no hit
        y_true = tf.constant(
            [
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # User 1: items 0, 2 positive
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # User 2: items 0, 2 positive
            ],
            dtype=tf.float32,
        )
        y_pred = tf.constant(
            [
                [0, 1, 3, 4, 5],  # User 1: item 0 is in top-5
                [1, 3, 4, 5, 6],  # User 2: no positive items
            ],
            dtype=tf.int32,
        )

        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()

        # Average: (1.0 + 0.0) / 2 = 0.5
        self.assertAlmostEqual(result.numpy(), 0.5, places=4)

    def test_metric_reset_state(self) -> None:
        """Test metric reset state."""
        logger.info("🧪 Testing AccuracyAtK reset_state")

        y_true = tf.constant([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred = tf.constant([[0, 1, 3, 4, 5]], dtype=tf.int32)

        self.metric.update_state(y_true, y_pred)
        self.metric.result()

        # Reset state
        self.metric.reset_state()
        result2 = self.metric.result()

        # After reset, result should be 0
        self.assertAlmostEqual(result2.numpy(), 0.0, places=4)

    def test_metric_serialization(self) -> None:
        """Test metric serialization."""
        logger.info("🧪 Testing AccuracyAtK serialization")

        config = self.metric.get_config()
        self.assertIsInstance(config, dict)
        self.assertIn("name", config)
        self.assertIn("k", config)
        self.assertEqual(config["k"], 5)

        # Test from_config
        new_metric = AccuracyAtK.from_config(config)
        self.assertIsInstance(new_metric, AccuracyAtK)
        self.assertEqual(new_metric.name, self.metric.name)
        self.assertEqual(new_metric.k, self.metric.k)

    def test_metric_with_different_k_values(self) -> None:
        """Test metric with different K values."""
        logger.info("🧪 Testing AccuracyAtK with different K values")

        # Test with k=3
        metric_k3 = AccuracyAtK(k=3)
        y_true = tf.constant([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred = tf.constant([[0, 1, 2]], dtype=tf.int32)  # top-3: [0, 1, 2]

        metric_k3.update_state(y_true, y_pred)
        result_k3 = metric_k3.result()
        # Item 0 is in top-3, so should be 1.0
        self.assertAlmostEqual(result_k3.numpy(), 1.0, places=4)

        # Test with k=10
        metric_k10 = AccuracyAtK(k=10)
        y_pred_k10 = tf.constant([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=tf.int32)

        metric_k10.update_state(y_true, y_pred_k10)
        result_k10 = metric_k10.result()
        # Items 0 and 2 are both in top-10, so should be 1.0
        self.assertAlmostEqual(result_k10.numpy(), 1.0, places=4)

    def test_metric_with_all_positive_items_in_top_k(self) -> None:
        """Test metric when all positive items are in top-K."""
        logger.info("🧪 Testing AccuracyAtK - all positives in top-K")

        # y_true: items 0, 1, 2 are positive
        # y_pred: top-5 are [0, 1, 2, 3, 4] - all positives are in top-5
        y_true = tf.constant([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int32)

        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()

        # Should be 1.0 (at least one positive in top-5)
        self.assertAlmostEqual(result.numpy(), 1.0, places=4)

    def test_metric_with_no_positive_items(self) -> None:
        """Test metric when user has no positive items."""
        logger.info("🧪 Testing AccuracyAtK - no positive items")

        # y_true: no positive items
        y_true = tf.constant([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int32)

        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()

        # Should be 0.0 (no positive items to find)
        self.assertAlmostEqual(result.numpy(), 0.0, places=4)

    def test_metric_result_type(self) -> None:
        """Test that metric result is a tensor."""
        logger.info("🧪 Testing AccuracyAtK result type")

        y_true = tf.constant([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred = tf.constant([[0, 1, 3, 4, 5]], dtype=tf.int32)

        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()

        # Result should be a tensor (can be converted to numpy)
        self.assertTrue(hasattr(result, "numpy"))
        self.assertIsInstance(result.numpy(), (float, np.floating))


if __name__ == "__main__":
    unittest.main()
