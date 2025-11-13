"""Unit tests for Median metric."""
import unittest

import keras
import tensorflow as tf
from loguru import logger

from kerasfactory.metrics import Median


class TestMedian(unittest.TestCase):
    """Test cases for Median metric."""

    def setUp(self) -> None:
        """Set up test case."""
        self.metric = Median()

    def test_metric_initialization(self) -> None:
        """Test metric initialization."""
        logger.info("🧪 Testing Median initialization")
        self.assertIsInstance(self.metric, Median)
        self.assertEqual(self.metric.name, "median")

    def test_metric_initialization_with_custom_name(self) -> None:
        """Test metric initialization with custom name."""
        logger.info("🧪 Testing Median initialization with custom name")
        custom_metric = Median(name="custom_median")
        self.assertEqual(custom_metric.name, "custom_median")

    def test_metric_update_state(self) -> None:
        """Test metric update state."""
        logger.info("🧪 Testing Median update_state")

        # Create test data
        y_pred = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0], dtype=tf.float32)

        # Update metric
        self.metric.update_state(y_pred)

        # Check result
        result = self.metric.result()
        self.assertIsInstance(result, keras.Variable)
        self.assertGreater(result.numpy(), 0)

    def test_metric_update_state_multiple_times(self) -> None:
        """Test metric update state multiple times."""
        logger.info("🧪 Testing Median update_state multiple times")

        # Create test data
        y_pred1 = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        y_pred2 = tf.constant([4.0, 5.0, 6.0], dtype=tf.float32)

        # Update metric multiple times
        self.metric.update_state(y_pred1)
        self.metric.update_state(y_pred2)

        # Check result
        result = self.metric.result()
        self.assertIsInstance(result, keras.Variable)
        self.assertGreater(result.numpy(), 0)

    def test_metric_reset_state(self) -> None:
        """Test metric reset state."""
        logger.info("🧪 Testing Median reset_state")

        # Create test data
        y_pred = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0], dtype=tf.float32)

        # Update metric
        self.metric.update_state(y_pred)
        self.metric.result()

        # Reset state
        self.metric.reset_state()
        result2 = self.metric.result()

        # After reset, result should be 0
        self.assertEqual(result2.numpy(), 0.0)

    def test_metric_serialization(self) -> None:
        """Test metric serialization."""
        logger.info("🧪 Testing Median serialization")

        config = self.metric.get_config()
        self.assertIsInstance(config, dict)
        self.assertIn("name", config)

        # Test from_config
        new_metric = Median.from_config(config)
        self.assertIsInstance(new_metric, Median)
        self.assertEqual(new_metric.name, self.metric.name)

    def test_metric_with_different_data_shapes(self) -> None:
        """Test metric with different data shapes."""
        logger.info("🧪 Testing Median with different data shapes")

        # Test with 1D data
        y_pred_1d = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0], dtype=tf.float32)
        self.metric.update_state(y_pred_1d)
        result_1d = self.metric.result()
        self.assertGreater(result_1d.numpy(), 0)

        # Reset and test with 2D data
        self.metric.reset_state()
        y_pred_2d = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
        self.metric.update_state(y_pred_2d)
        result_2d = self.metric.result()
        self.assertGreater(result_2d.numpy(), 0)

    def test_metric_with_known_median(self) -> None:
        """Test metric with known median value."""
        logger.info("🧪 Testing Median with known median value")

        # Create data with known median (3.0)
        y_pred = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0], dtype=tf.float32)

        # Update metric
        self.metric.update_state(y_pred)

        # Check result (should be close to 3.0)
        result = self.metric.result()
        self.assertAlmostEqual(result.numpy(), 3.0, places=1)

    def test_metric_with_even_number_of_elements(self) -> None:
        """Test metric with even number of elements."""
        logger.info("🧪 Testing Median with even number of elements")

        # Create data with even number of elements (median should be average of middle two)
        y_pred = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)

        # Update metric
        self.metric.update_state(y_pred)

        # Check result (should be 2.5, average of 2.0 and 3.0)
        result = self.metric.result()
        self.assertAlmostEqual(result.numpy(), 2.5, places=1)

    def test_metric_with_single_element(self) -> None:
        """Test metric with single element."""
        logger.info("🧪 Testing Median with single element")

        # Create data with single element
        y_pred = tf.constant([5.0], dtype=tf.float32)

        # Update metric
        self.metric.update_state(y_pred)

        # Check result (should be the element itself)
        result = self.metric.result()
        self.assertEqual(result.numpy(), 5.0)


if __name__ == "__main__":
    unittest.main()
