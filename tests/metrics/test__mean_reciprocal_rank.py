"""Unit tests for MeanReciprocalRank metric."""
import unittest

import numpy as np
import tensorflow as tf
from loguru import logger

from kmr.metrics import MeanReciprocalRank


class TestMeanReciprocalRank(unittest.TestCase):
    """Test cases for MeanReciprocalRank metric."""

    def setUp(self) -> None:
        """Set up test case."""
        self.metric = MeanReciprocalRank()

    def test_metric_initialization(self) -> None:
        """Test metric initialization."""
        logger.info("🧪 Testing MeanReciprocalRank initialization")
        self.assertIsInstance(self.metric, MeanReciprocalRank)
        self.assertEqual(self.metric.name, "mean_reciprocal_rank")

    def test_metric_initialization_with_custom_name(self) -> None:
        """Test metric initialization with custom name."""
        logger.info("🧪 Testing MeanReciprocalRank initialization with custom name")
        custom_metric = MeanReciprocalRank(name="custom_mrr")
        self.assertEqual(custom_metric.name, "custom_mrr")

    def test_metric_update_state_basic(self) -> None:
        """Test metric update state with basic case."""
        logger.info("🧪 Testing MeanReciprocalRank update_state - basic case")

        # y_true: items 0 and 2 are positive
        # y_pred: top-5 are [1, 0, 3, 4, 5] - item 0 is at position 2 (1-indexed)
        y_true = tf.constant([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred = tf.constant([[1, 0, 3, 4, 5]], dtype=tf.int32)

        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()

        # MRR = 1/2 = 0.5 (first positive at rank 2)
        self.assertAlmostEqual(result.numpy(), 0.5, places=4)

    def test_metric_update_state_first_position(self) -> None:
        """Test metric when first positive is at position 1."""
        logger.info("🧪 Testing MeanReciprocalRank - first position")

        # y_true: items 0 and 2 are positive
        # y_pred: top-5 are [0, 1, 3, 4, 5] - item 0 is at position 1
        y_true = tf.constant([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred = tf.constant([[0, 1, 3, 4, 5]], dtype=tf.int32)

        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()

        # MRR = 1/1 = 1.0
        self.assertAlmostEqual(result.numpy(), 1.0, places=4)

    def test_metric_update_state_no_hit(self) -> None:
        """Test metric when no positive item is found."""
        logger.info("🧪 Testing MeanReciprocalRank - no hit")

        # y_true: items 0 and 2 are positive
        # y_pred: top-5 are [1, 3, 4, 5, 6] - no positive items
        y_true = tf.constant([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred = tf.constant([[1, 3, 4, 5, 6]], dtype=tf.int32)

        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()

        # MRR = 0.0 (no positive found)
        self.assertAlmostEqual(result.numpy(), 0.0, places=4)

    def test_metric_update_state_multiple_batches(self) -> None:
        """Test metric update state with multiple batches."""
        logger.info("🧪 Testing MeanReciprocalRank update_state - multiple batches")

        # Batch 1: MRR = 1.0 (first positive at rank 1)
        y_true_1 = tf.constant([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred_1 = tf.constant([[0, 1, 3, 4, 5]], dtype=tf.int32)

        # Batch 2: MRR = 0.5 (first positive at rank 2)
        y_true_2 = tf.constant([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred_2 = tf.constant([[1, 0, 3, 4, 5]], dtype=tf.int32)

        self.metric.update_state(y_true_1, y_pred_1)
        self.metric.update_state(y_true_2, y_pred_2)

        result = self.metric.result()
        # Average: (1.0 + 0.5) / 2 = 0.75
        self.assertAlmostEqual(result.numpy(), 0.75, places=4)

    def test_metric_update_state_multiple_users(self) -> None:
        """Test metric with multiple users in batch."""
        logger.info("🧪 Testing MeanReciprocalRank update_state - multiple users")

        # User 1: MRR = 1.0 (first positive at rank 1)
        # User 2: MRR = 0.3333 (first positive at rank 3)
        y_true = tf.constant(
            [
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # User 1: items 0, 2 positive
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # User 2: items 0, 2 positive
            ],
            dtype=tf.float32,
        )
        y_pred = tf.constant(
            [
                [0, 1, 3, 4, 5],  # User 1: item 0 at rank 1
                [1, 3, 0, 4, 5],  # User 2: item 0 at rank 3
            ],
            dtype=tf.int32,
        )

        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()

        # Average: (1.0 + 1/3) / 2 = (1.0 + 0.3333) / 2 = 0.6667
        self.assertAlmostEqual(result.numpy(), (1.0 + 1.0 / 3.0) / 2.0, places=4)

    def test_metric_reset_state(self) -> None:
        """Test metric reset state."""
        logger.info("🧪 Testing MeanReciprocalRank reset_state")

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
        logger.info("🧪 Testing MeanReciprocalRank serialization")

        config = self.metric.get_config()
        self.assertIsInstance(config, dict)
        self.assertIn("name", config)

        # Test from_config
        new_metric = MeanReciprocalRank.from_config(config)
        self.assertIsInstance(new_metric, MeanReciprocalRank)
        self.assertEqual(new_metric.name, self.metric.name)

    def test_metric_with_no_positive_items(self) -> None:
        """Test metric when user has no positive items."""
        logger.info("🧪 Testing MeanReciprocalRank - no positive items")

        # y_true: no positive items
        y_true = tf.constant([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int32)

        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()

        # Should be 0.0 (no positive items to find)
        self.assertAlmostEqual(result.numpy(), 0.0, places=4)

    def test_metric_result_type(self) -> None:
        """Test that metric result is a tensor."""
        logger.info("🧪 Testing MeanReciprocalRank result type")

        y_true = tf.constant([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred = tf.constant([[0, 1, 3, 4, 5]], dtype=tf.int32)

        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()

        # Result should be a tensor (can be converted to numpy)
        self.assertTrue(hasattr(result, "numpy"))
        self.assertIsInstance(result.numpy(), (float, np.floating))


if __name__ == "__main__":
    unittest.main()
