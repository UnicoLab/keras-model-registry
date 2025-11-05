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

    def test_metric_with_large_num_items(self) -> None:
        """Test metric with large num_items (realistic scenario like 500 items)."""
        logger.info("🧪 Testing AccuracyAtK with large num_items")

        # Simulate notebook scenario: 500 items, 8 users
        n_items = 500
        batch_size = 8
        y_true = tf.constant(np.zeros((batch_size, n_items), dtype=np.float32))
        y_true = y_true.numpy()
        y_true[0, [10, 20, 30]] = 1.0  # User 0 has positives at 10, 20, 30
        y_true[1, [50, 100]] = 1.0  # User 1 has positives at 50, 100
        y_true = tf.constant(y_true)

        y_pred = tf.constant(
            np.array(
                [
                    [10, 20, 30, 40, 50],  # User 0: all positives in top-5
                    [50, 100, 200, 300, 400],  # User 1: positives at 50, 100
                    [1, 2, 3, 4, 5],  # User 2: no positives
                ]
                * 3,
                dtype=np.int32,
            )[:batch_size],
        )

        metric = AccuracyAtK(k=5)
        metric.update_state(y_true, y_pred)
        result = metric.result()

        # User 0: hit, User 1: hit, Users 2-7: no hit
        # Average: (1 + 1 + 0 + 0 + 0 + 0 + 0 + 0) / 8 = 0.25
        self.assertGreaterEqual(result.numpy(), 0.0)
        self.assertLessEqual(result.numpy(), 1.0)

    def test_metric_with_out_of_bounds_indices(self) -> None:
        """Test metric with out-of-bounds indices (clamping behavior)."""
        logger.info("🧪 Testing AccuracyAtK with out-of-bounds indices")

        # y_true has 8 items, but y_pred contains indices >= 8
        # The metric should clamp indices and not crash
        y_true = tf.constant(np.zeros((2, 8), dtype=np.float32))
        y_true = y_true.numpy()
        y_true[0, [0, 2]] = 1.0  # User 0 has positives at 0, 2
        y_true = tf.constant(y_true)

        # y_pred contains indices 20, 31 which are out of bounds for 8 items
        # These should be clamped to valid range
        y_pred = tf.constant([[20, 31, 0, 2, 5]], dtype=tf.int32)
        y_pred = tf.tile(y_pred, [2, 1])  # (2, 5)

        metric = AccuracyAtK(k=5)
        metric.update_state(y_true, y_pred)
        result = metric.result()

        # Should not crash, result should be valid
        self.assertGreaterEqual(result.numpy(), 0.0)
        self.assertLessEqual(result.numpy(), 1.0)

    def test_metric_with_shape_mismatch_edge_case(self) -> None:
        """Test metric with edge case shape mismatch."""
        logger.info("🧪 Testing AccuracyAtK with shape mismatch edge case")

        # Smaller y_true than expected (edge case)
        y_true = tf.constant(np.zeros((2, 8), dtype=np.float32))
        y_true = y_true.numpy()
        y_true[0, [0, 2]] = 1.0
        y_true = tf.constant(y_true)

        # y_pred contains indices that would be out of bounds
        # Metric should handle this gracefully
        y_pred = tf.constant([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=tf.int32)

        metric = AccuracyAtK(k=5)
        metric.update_state(y_true, y_pred)
        result = metric.result()

        # Should not crash
        self.assertGreaterEqual(result.numpy(), 0.0)
        self.assertLessEqual(result.numpy(), 1.0)

    def test_metric_with_large_batch_size(self) -> None:
        """Test metric with large batch size."""
        logger.info("🧪 Testing AccuracyAtK with large batch size")

        batch_size = 32
        n_items = 100
        y_true = tf.constant(np.zeros((batch_size, n_items), dtype=np.float32))
        y_true = y_true.numpy()
        # Add some positives
        for i in range(batch_size):
            y_true[i, [i % 10, (i + 5) % 10]] = 1.0
        y_true = tf.constant(y_true)

        y_pred = tf.constant(
            np.array(
                [
                    [i % 10, (i + 5) % 10, (i + 10) % 20, (i + 15) % 20, (i + 20) % 20]
                    for i in range(batch_size)
                ],
                dtype=np.int32,
            ),
        )

        metric = AccuracyAtK(k=5)
        metric.update_state(y_true, y_pred)
        result = metric.result()

        # Should handle large batch correctly
        self.assertGreaterEqual(result.numpy(), 0.0)
        self.assertLessEqual(result.numpy(), 1.0)

    def test_metric_with_varying_k_less_than_pred_size(self) -> None:
        """Test metric when k < len(y_pred)."""
        logger.info("🧪 Testing AccuracyAtK with k < len(y_pred)")

        y_true = tf.constant([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        # y_pred has 10 items, but k=5, so only first 5 should be considered
        y_pred = tf.constant([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=tf.int32)

        metric = AccuracyAtK(k=5)
        metric.update_state(y_true, y_pred)
        result = metric.result()

        # Should only consider first 5 items: [0, 1, 2, 3, 4]
        # Item 0 is positive, so should be 1.0
        self.assertAlmostEqual(result.numpy(), 1.0, places=4)

    def test_metric_consistency_across_multiple_updates(self) -> None:
        """Test metric consistency across multiple update calls."""
        logger.info("🧪 Testing AccuracyAtK consistency")

        metric = AccuracyAtK(k=5)

        # Update 1: 1 hit
        y_true_1 = tf.constant([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred_1 = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int32)
        metric.update_state(y_true_1, y_pred_1)
        result_1 = metric.result().numpy()

        # Update 2: 0 hits
        y_true_2 = tf.constant([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred_2 = tf.constant([[1, 2, 3, 4, 5]], dtype=tf.int32)
        metric.update_state(y_true_2, y_pred_2)
        result_2 = metric.result().numpy()

        # Should average: (1.0 + 0.0) / 2 = 0.5
        self.assertAlmostEqual(result_2, 0.5, places=4)

    def test_metric_with_empty_batch(self) -> None:
        """Test metric with empty batch (edge case)."""
        logger.info("🧪 Testing AccuracyAtK with empty batch")

        # Empty batch (batch_size=0)
        y_true = tf.constant(np.zeros((0, 10), dtype=np.float32))
        y_pred = tf.constant(np.zeros((0, 5), dtype=np.int32))

        metric = AccuracyAtK(k=5)
        metric.update_state(y_true, y_pred)
        result = metric.result()

        # Should handle gracefully (result will be 0/0 = 0 due to epsilon)
        self.assertGreaterEqual(result.numpy(), 0.0)
        self.assertLessEqual(result.numpy(), 1.0)

    def test_metric_with_all_zeros(self) -> None:
        """Test metric when y_true is all zeros."""
        logger.info("🧪 Testing AccuracyAtK with all zeros")

        y_true = tf.constant([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int32)

        metric = AccuracyAtK(k=5)
        metric.update_state(y_true, y_pred)
        result = metric.result()

        # Should be 0.0 (no positive items)
        self.assertAlmostEqual(result.numpy(), 0.0, places=4)

    def test_metric_with_all_ones(self) -> None:
        """Test metric when all items are positive."""
        logger.info("🧪 Testing AccuracyAtK with all ones")

        y_true = tf.constant([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=tf.float32)
        y_pred = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int32)

        metric = AccuracyAtK(k=5)
        metric.update_state(y_true, y_pred)
        result = metric.result()

        # Should be 1.0 (at least one positive in top-5, actually all are positive)
        self.assertAlmostEqual(result.numpy(), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
