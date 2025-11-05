"""Unit tests for NDCGAtK metric."""
import unittest

import numpy as np
import tensorflow as tf
from loguru import logger

from kmr.metrics import NDCGAtK


class TestNDCGAtK(unittest.TestCase):
    """Test cases for NDCGAtK metric."""

    def setUp(self) -> None:
        """Set up test case."""
        self.metric = NDCGAtK(k=5)

    def test_metric_initialization(self) -> None:
        """Test metric initialization."""
        logger.info("🧪 Testing NDCGAtK initialization")
        self.assertIsInstance(self.metric, NDCGAtK)
        self.assertEqual(self.metric.name, "ndcg_at_k")
        self.assertEqual(self.metric.k, 5)

    def test_metric_initialization_with_custom_name(self) -> None:
        """Test metric initialization with custom name."""
        logger.info("🧪 Testing NDCGAtK initialization with custom name")
        custom_metric = NDCGAtK(k=10, name="custom_ndcg@10")
        self.assertEqual(custom_metric.name, "custom_ndcg@10")
        self.assertEqual(custom_metric.k, 10)

    def test_metric_update_state_basic(self) -> None:
        """Test metric update state with basic case."""
        logger.info("🧪 Testing NDCGAtK update_state - basic case")

        # y_true: items 0 and 2 are positive
        # y_pred: top-5 are [0, 1, 3, 2, 4] - items 0 and 2 are in top-5
        y_true = tf.constant([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred = tf.constant([[0, 1, 3, 2, 4]], dtype=tf.int32)

        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()

        # NDCG should be > 0 (positive items found)
        self.assertGreater(result.numpy(), 0.0)
        self.assertLessEqual(result.numpy(), 1.0)

    def test_metric_update_state_ideal_ranking(self) -> None:
        """Test metric with ideal ranking (all positives at top)."""
        logger.info("🧪 Testing NDCGAtK - ideal ranking")

        # y_true: items 0, 1, 2 are positive
        # y_pred: top-5 are [0, 1, 2, 3, 4] - all positives at top (ideal)
        y_true = tf.constant([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int32)

        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()

        # Ideal ranking should give NDCG close to 1.0
        self.assertGreater(result.numpy(), 0.9)
        self.assertLessEqual(result.numpy(), 1.0)

    def test_metric_update_state_no_relevant(self) -> None:
        """Test metric when no positive items are in top-K."""
        logger.info("🧪 Testing NDCGAtK - no relevant")

        # y_true: items 0 and 2 are positive
        # y_pred: top-5 are [1, 3, 4, 5, 6] - no positive items
        y_true = tf.constant([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred = tf.constant([[1, 3, 4, 5, 6]], dtype=tf.int32)

        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()

        # NDCG = 0.0 (no relevant items found)
        self.assertAlmostEqual(result.numpy(), 0.0, places=4)

    def test_metric_update_state_multiple_batches(self) -> None:
        """Test metric update state with multiple batches."""
        logger.info("🧪 Testing NDCGAtK update_state - multiple batches")

        # Batch 1: has relevant items
        y_true_1 = tf.constant([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred_1 = tf.constant([[0, 1, 3, 2, 4]], dtype=tf.int32)

        # Batch 2: no relevant items
        y_true_2 = tf.constant([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred_2 = tf.constant([[1, 3, 4, 5, 6]], dtype=tf.int32)

        self.metric.update_state(y_true_1, y_pred_1)
        self.metric.update_state(y_true_2, y_pred_2)

        result = self.metric.result()
        # Should be average of two batches
        self.assertGreaterEqual(result.numpy(), 0.0)
        self.assertLessEqual(result.numpy(), 1.0)

    def test_metric_update_state_multiple_users(self) -> None:
        """Test metric with multiple users in batch."""
        logger.info("🧪 Testing NDCGAtK update_state - multiple users")

        # User 1: has relevant items
        # User 2: has relevant items
        y_true = tf.constant(
            [
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # User 1: items 0, 2 positive
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # User 2: items 0, 2 positive
            ],
            dtype=tf.float32,
        )
        y_pred = tf.constant(
            [
                [0, 1, 3, 2, 4],  # User 1: items 0, 2 in top-5
                [0, 1, 3, 2, 4],  # User 2: items 0, 2 in top-5
            ],
            dtype=tf.int32,
        )

        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()

        # Should be positive and <= 1.0
        self.assertGreater(result.numpy(), 0.0)
        self.assertLessEqual(result.numpy(), 1.0)

    def test_metric_reset_state(self) -> None:
        """Test metric reset state."""
        logger.info("🧪 Testing NDCGAtK reset_state")

        y_true = tf.constant([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred = tf.constant([[0, 1, 3, 2, 4]], dtype=tf.int32)

        self.metric.update_state(y_true, y_pred)
        self.metric.result()

        # Reset state
        self.metric.reset_state()
        result2 = self.metric.result()

        # After reset, result should be 0
        self.assertAlmostEqual(result2.numpy(), 0.0, places=4)

    def test_metric_serialization(self) -> None:
        """Test metric serialization."""
        logger.info("🧪 Testing NDCGAtK serialization")

        config = self.metric.get_config()
        self.assertIsInstance(config, dict)
        self.assertIn("name", config)
        self.assertIn("k", config)
        self.assertEqual(config["k"], 5)

        # Test from_config
        new_metric = NDCGAtK.from_config(config)
        self.assertIsInstance(new_metric, NDCGAtK)
        self.assertEqual(new_metric.name, self.metric.name)
        self.assertEqual(new_metric.k, self.metric.k)

    def test_metric_with_different_k_values(self) -> None:
        """Test metric with different K values."""
        logger.info("🧪 Testing NDCGAtK with different K values")

        y_true = tf.constant([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)

        # Test with k=3
        metric_k3 = NDCGAtK(k=3)
        y_pred_k3 = tf.constant([[0, 1, 2]], dtype=tf.int32)

        metric_k3.update_state(y_true, y_pred_k3)
        result_k3 = metric_k3.result()
        self.assertGreaterEqual(result_k3.numpy(), 0.0)
        self.assertLessEqual(result_k3.numpy(), 1.0)

        # Test with k=10
        metric_k10 = NDCGAtK(k=10)
        y_pred_k10 = tf.constant([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=tf.int32)

        metric_k10.update_state(y_true, y_pred_k10)
        result_k10 = metric_k10.result()
        self.assertGreaterEqual(result_k10.numpy(), 0.0)
        self.assertLessEqual(result_k10.numpy(), 1.0)

    def test_metric_with_no_positive_items(self) -> None:
        """Test metric when user has no positive items."""
        logger.info("🧪 Testing NDCGAtK - no positive items")

        # y_true: no positive items
        y_true = tf.constant([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int32)

        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()

        # Should be 0.0 (no positive items to find)
        self.assertAlmostEqual(result.numpy(), 0.0, places=4)

    def test_metric_result_type(self) -> None:
        """Test that metric result is a tensor."""
        logger.info("🧪 Testing NDCGAtK result type")

        y_true = tf.constant([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred = tf.constant([[0, 1, 3, 2, 4]], dtype=tf.int32)

        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()

        # Result should be a tensor (can be converted to numpy)
        self.assertTrue(hasattr(result, "numpy"))
        self.assertIsInstance(result.numpy(), (float, np.floating))

    def test_metric_with_large_num_items(self) -> None:
        """Test metric with large num_items (realistic scenario)."""
        logger.info("🧪 Testing NDCGAtK with large num_items")

        n_items = 500
        batch_size = 8
        y_true = tf.constant(np.zeros((batch_size, n_items), dtype=np.float32))
        y_true = y_true.numpy()
        y_true[0, [10, 20, 30]] = 1.0
        y_true[1, [50, 100]] = 1.0
        y_true = tf.constant(y_true)

        y_pred = tf.constant(
            np.array(
                [
                    [10, 20, 30, 40, 50],
                    [50, 100, 200, 300, 400],
                    [1, 2, 3, 4, 5],
                ]
                * 3,
                dtype=np.int32,
            )[:batch_size],
        )

        metric = NDCGAtK(k=5)
        metric.update_state(y_true, y_pred)
        result = metric.result()

        self.assertGreaterEqual(result.numpy(), 0.0)
        self.assertLessEqual(result.numpy(), 1.0)

    def test_metric_with_out_of_bounds_indices(self) -> None:
        """Test metric with out-of-bounds indices (clamping behavior)."""
        logger.info("🧪 Testing NDCGAtK with out-of-bounds indices")

        y_true = tf.constant(np.zeros((2, 8), dtype=np.float32))
        y_true = y_true.numpy()
        y_true[0, [0, 2]] = 1.0
        y_true = tf.constant(y_true)

        y_pred = tf.constant([[20, 31, 0, 2, 5]], dtype=tf.int32)
        y_pred = tf.tile(y_pred, [2, 1])

        metric = NDCGAtK(k=5)
        metric.update_state(y_true, y_pred)
        result = metric.result()

        self.assertGreaterEqual(result.numpy(), 0.0)
        self.assertLessEqual(result.numpy(), 1.0)

    def test_metric_with_large_batch_size(self) -> None:
        """Test metric with large batch size."""
        logger.info("🧪 Testing NDCGAtK with large batch size")

        batch_size = 32
        n_items = 100
        y_true = tf.constant(np.zeros((batch_size, n_items), dtype=np.float32))
        y_true = y_true.numpy()
        for i in range(batch_size):
            pos1 = (i * 2) % n_items
            pos2 = (i * 2 + 1) % n_items
            y_true[i, [pos1, pos2]] = 1.0
        y_true = tf.constant(y_true)

        y_pred = tf.constant(
            np.array(
                [
                    [
                        (i * 2) % n_items,
                        (i * 2 + 1) % n_items,
                        (i * 2 + 10) % n_items,
                        (i * 2 + 20) % n_items,
                        (i * 2 + 30) % n_items,
                    ]
                    for i in range(batch_size)
                ],
                dtype=np.int32,
            ),
        )

        metric = NDCGAtK(k=5)
        metric.update_state(y_true, y_pred)
        result = metric.result()

        self.assertGreaterEqual(result.numpy(), 0.0)
        self.assertLessEqual(result.numpy(), 1.0)


if __name__ == "__main__":
    unittest.main()
