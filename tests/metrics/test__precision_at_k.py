"""Unit tests for PrecisionAtK metric."""
import unittest

import keras
import numpy as np
import tensorflow as tf
from loguru import logger

from kmr.metrics import PrecisionAtK


class TestPrecisionAtK(unittest.TestCase):
    """Test cases for PrecisionAtK metric."""

    def setUp(self) -> None:
        """Set up test case."""
        self.metric = PrecisionAtK(k=5)

    def test_metric_initialization(self) -> None:
        """Test metric initialization."""
        logger.info("🧪 Testing PrecisionAtK initialization")
        self.assertIsInstance(self.metric, PrecisionAtK)
        self.assertEqual(self.metric.name, "precision_at_k")
        self.assertEqual(self.metric.k, 5)

    def test_metric_initialization_with_custom_name(self) -> None:
        """Test metric initialization with custom name."""
        logger.info("🧪 Testing PrecisionAtK initialization with custom name")
        custom_metric = PrecisionAtK(k=10, name="custom_prec@10")
        self.assertEqual(custom_metric.name, "custom_prec@10")
        self.assertEqual(custom_metric.k, 10)

    def test_metric_update_state_basic(self) -> None:
        """Test metric update state with basic case."""
        logger.info("🧪 Testing PrecisionAtK update_state - basic case")

        # y_true: items 0 and 2 are positive
        # y_pred: top-5 are [0, 1, 3, 2, 4] - items 0 and 2 are positive
        y_true = tf.constant([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred = tf.constant([[0, 1, 3, 2, 4]], dtype=tf.int32)

        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()

        # Precision@5 = 2 positive / 5 total = 0.4
        self.assertAlmostEqual(result.numpy(), 0.4, places=4)

    def test_metric_update_state_no_relevant(self) -> None:
        """Test metric when no positive items are in top-K."""
        logger.info("🧪 Testing PrecisionAtK update_state - no relevant")

        # y_true: items 0 and 2 are positive
        # y_pred: top-5 are [1, 3, 4, 5, 6] - no positive items
        y_true = tf.constant([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred = tf.constant([[1, 3, 4, 5, 6]], dtype=tf.int32)

        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()

        # Precision@5 = 0 / 5 = 0.0
        self.assertAlmostEqual(result.numpy(), 0.0, places=4)

    def test_metric_update_state_all_relevant(self) -> None:
        """Test metric when all items in top-K are positive."""
        logger.info("🧪 Testing PrecisionAtK update_state - all relevant")

        # y_true: items 0, 1, 2, 3, 4 are positive
        # y_pred: top-5 are [0, 1, 2, 3, 4] - all positive
        y_true = tf.constant([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int32)

        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()

        # Precision@5 = 5 / 5 = 1.0
        self.assertAlmostEqual(result.numpy(), 1.0, places=4)

    def test_metric_update_state_multiple_batches(self) -> None:
        """Test metric update state with multiple batches."""
        logger.info("🧪 Testing PrecisionAtK update_state - multiple batches")

        # Batch 1: precision = 0.4 (2/5)
        y_true_1 = tf.constant([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred_1 = tf.constant([[0, 1, 3, 2, 4]], dtype=tf.int32)

        # Batch 2: precision = 0.0 (0/5)
        y_true_2 = tf.constant([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred_2 = tf.constant([[1, 3, 4, 5, 6]], dtype=tf.int32)

        self.metric.update_state(y_true_1, y_pred_1)
        self.metric.update_state(y_true_2, y_pred_2)

        result = self.metric.result()
        # Average: (0.4 + 0.0) / 2 = 0.2
        self.assertAlmostEqual(result.numpy(), 0.2, places=4)

    def test_metric_update_state_multiple_users(self) -> None:
        """Test metric with multiple users in batch."""
        logger.info("🧪 Testing PrecisionAtK update_state - multiple users")

        # User 1: precision = 0.4 (2/5)
        # User 2: precision = 0.2 (1/5)
        y_true = tf.constant(
            [
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # User 1: items 0, 2 positive
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # User 2: item 0 positive
            ],
            dtype=tf.float32,
        )
        y_pred = tf.constant(
            [
                [0, 1, 3, 2, 4],  # User 1: items 0, 2 in top-5
                [0, 1, 2, 3, 4],  # User 2: item 0 in top-5
            ],
            dtype=tf.int32,
        )

        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()

        # Average: (0.4 + 0.2) / 2 = 0.3
        self.assertAlmostEqual(result.numpy(), 0.3, places=4)

    def test_metric_reset_state(self) -> None:
        """Test metric reset state."""
        logger.info("🧪 Testing PrecisionAtK reset_state")

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
        logger.info("🧪 Testing PrecisionAtK serialization")

        config = self.metric.get_config()
        self.assertIsInstance(config, dict)
        self.assertIn("name", config)
        self.assertIn("k", config)
        self.assertEqual(config["k"], 5)

        # Test from_config
        new_metric = PrecisionAtK.from_config(config)
        self.assertIsInstance(new_metric, PrecisionAtK)
        self.assertEqual(new_metric.name, self.metric.name)
        self.assertEqual(new_metric.k, self.metric.k)

    def test_metric_with_different_k_values(self) -> None:
        """Test metric with different K values."""
        logger.info("🧪 Testing PrecisionAtK with different K values")

        # Test with k=3
        metric_k3 = PrecisionAtK(k=3)
        y_true = tf.constant([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred = tf.constant(
            [[0, 1, 2]],
            dtype=tf.int32,
        )  # top-3: [0, 1, 2], items 0 and 2 are positive

        metric_k3.update_state(y_true, y_pred)
        result_k3 = metric_k3.result()
        # Precision@3 = 2 / 3 = 0.6667
        self.assertAlmostEqual(result_k3.numpy(), 2.0 / 3.0, places=4)

    def test_metric_result_type(self) -> None:
        """Test that metric result is a tensor."""
        logger.info("🧪 Testing PrecisionAtK result type")

        y_true = tf.constant([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred = tf.constant([[0, 1, 3, 2, 4]], dtype=tf.int32)

        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()

        # Result should be a tensor (can be converted to numpy)
        self.assertTrue(hasattr(result, "numpy"))
        self.assertIsInstance(result.numpy(), (float, np.floating))

    def test_metric_with_large_num_items(self) -> None:
        """Test metric with large num_items (realistic scenario)."""
        logger.info("🧪 Testing PrecisionAtK with large num_items")

        n_items = 500
        batch_size = 8
        y_true = tf.constant(np.zeros((batch_size, n_items), dtype=np.float32))
        y_true = y_true.numpy()
        y_true[0, [10, 20, 30]] = 1.0  # User 0 has 3 positives
        y_true[1, [50, 100]] = 1.0  # User 1 has 2 positives
        y_true = tf.constant(y_true)

        y_pred = tf.constant(
            np.array(
                [
                    [10, 20, 30, 40, 50],  # User 0: 3/5 = 0.6
                    [50, 100, 200, 300, 400],  # User 1: 2/5 = 0.4
                    [1, 2, 3, 4, 5],  # User 2: 0/5 = 0.0
                ]
                * 3,
                dtype=np.int32,
            )[:batch_size],
        )

        metric = PrecisionAtK(k=5)
        metric.update_state(y_true, y_pred)
        result = metric.result()

        # Should be valid precision value
        self.assertGreaterEqual(result.numpy(), 0.0)
        self.assertLessEqual(result.numpy(), 1.0)

    def test_metric_with_out_of_bounds_indices(self) -> None:
        """Test metric with out-of-bounds indices (clamping behavior)."""
        logger.info("🧪 Testing PrecisionAtK with out-of-bounds indices")

        y_true = tf.constant(np.zeros((2, 8), dtype=np.float32))
        y_true = y_true.numpy()
        y_true[0, [0, 2]] = 1.0
        y_true = tf.constant(y_true)

        y_pred = tf.constant([[20, 31, 0, 2, 5]], dtype=tf.int32)
        y_pred = tf.tile(y_pred, [2, 1])

        metric = PrecisionAtK(k=5)
        metric.update_state(y_true, y_pred)
        result = metric.result()

        self.assertGreaterEqual(result.numpy(), 0.0)
        self.assertLessEqual(result.numpy(), 1.0)

    def test_metric_with_large_batch_size(self) -> None:
        """Test metric with large batch size."""
        logger.info("🧪 Testing PrecisionAtK with large batch size")

        batch_size = 32
        n_items = 100
        y_true = tf.constant(np.zeros((batch_size, n_items), dtype=np.float32))
        y_true = y_true.numpy()
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

        metric = PrecisionAtK(k=5)
        metric.update_state(y_true, y_pred)
        result = metric.result()

        self.assertGreaterEqual(result.numpy(), 0.0)
        self.assertLessEqual(result.numpy(), 1.0)

    def test_metric_with_perfect_precision(self) -> None:
        """Test metric when all top-K items are positive."""
        logger.info("🧪 Testing PrecisionAtK with perfect precision")

        y_true = tf.constant([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int32)

        metric = PrecisionAtK(k=5)
        metric.update_state(y_true, y_pred)
        result = metric.result()

        # Precision@5 = 5/5 = 1.0
        self.assertAlmostEqual(result.numpy(), 1.0, places=4)

    def test_metric_with_zero_precision(self) -> None:
        """Test metric when no top-K items are positive."""
        logger.info("🧪 Testing PrecisionAtK with zero precision")

        y_true = tf.constant([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred = tf.constant([[1, 3, 4, 5, 6]], dtype=tf.int32)

        metric = PrecisionAtK(k=5)
        metric.update_state(y_true, y_pred)
        result = metric.result()

        # Precision@5 = 0/5 = 0.0
        self.assertAlmostEqual(result.numpy(), 0.0, places=4)

    def test_metric_consistency_across_multiple_updates(self) -> None:
        """Test metric consistency across multiple update calls."""
        logger.info("🧪 Testing PrecisionAtK consistency")

        metric = PrecisionAtK(k=5)

        # Update 1: precision = 0.4 (2/5)
        y_true_1 = tf.constant([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred_1 = tf.constant([[0, 1, 3, 2, 4]], dtype=tf.int32)
        metric.update_state(y_true_1, y_pred_1)

        # Update 2: precision = 0.0 (0/5)
        y_true_2 = tf.constant([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
        y_pred_2 = tf.constant([[1, 3, 4, 5, 6]], dtype=tf.int32)
        metric.update_state(y_true_2, y_pred_2)
        result = metric.result()

        # Should average: (0.4 + 0.0) / 2 = 0.2
        self.assertAlmostEqual(result.numpy(), 0.2, places=4)


if __name__ == "__main__":
    unittest.main()
