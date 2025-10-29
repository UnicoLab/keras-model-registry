"""Unit tests for StandardDeviation metric."""
import unittest

import keras
import numpy as np
import tensorflow as tf
from loguru import logger

from kmr.metrics import StandardDeviation


class TestStandardDeviation(unittest.TestCase):
    """Test cases for StandardDeviation metric."""

    def setUp(self) -> None:
        """Set up test case."""
        self.metric = StandardDeviation()

    def test_metric_initialization(self) -> None:
        """Test metric initialization."""
        logger.info("🧪 Testing StandardDeviation initialization")
        self.assertIsInstance(self.metric, StandardDeviation)
        self.assertEqual(self.metric.name, "standard_deviation")

    def test_metric_initialization_with_custom_name(self) -> None:
        """Test metric initialization with custom name."""
        logger.info("🧪 Testing StandardDeviation initialization with custom name")
        custom_metric = StandardDeviation(name="custom_std")
        self.assertEqual(custom_metric.name, "custom_std")

    def test_metric_update_state(self) -> None:
        """Test metric update state."""
        logger.info("🧪 Testing StandardDeviation update_state")
        
        # Create test data
        y_pred = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=tf.float32)
        
        # Update metric
        self.metric.update_state(y_pred)
        
        # Check result
        result = self.metric.result()
        self.assertIsInstance(result, keras.Variable)
        self.assertGreater(result.numpy(), 0)

    def test_metric_update_state_multiple_times(self) -> None:
        """Test metric update state multiple times."""
        logger.info("🧪 Testing StandardDeviation update_state multiple times")
        
        # Create test data
        y_pred1 = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
        y_pred2 = tf.constant([[4.0, 5.0, 6.0]], dtype=tf.float32)
        
        # Update metric multiple times
        self.metric.update_state(y_pred1)
        self.metric.update_state(y_pred2)
        
        # Check result
        result = self.metric.result()
        self.assertIsInstance(result, keras.Variable)
        self.assertGreater(result.numpy(), 0)

    def test_metric_reset_state(self) -> None:
        """Test metric reset state."""
        logger.info("🧪 Testing StandardDeviation reset_state")
        
        # Create test data
        y_pred = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
        
        # Update metric
        self.metric.update_state(y_pred)
        result1 = self.metric.result()
        
        # Reset state
        self.metric.reset_state()
        result2 = self.metric.result()
        
        # After reset, result should be 0
        self.assertEqual(result2.numpy(), 0.0)

    def test_metric_serialization(self) -> None:
        """Test metric serialization."""
        logger.info("🧪 Testing StandardDeviation serialization")
        
        config = self.metric.get_config()
        self.assertIsInstance(config, dict)
        self.assertIn("name", config)
        
        # Test from_config
        new_metric = StandardDeviation.from_config(config)
        self.assertIsInstance(new_metric, StandardDeviation)
        self.assertEqual(new_metric.name, self.metric.name)

    def test_metric_with_different_data_shapes(self) -> None:
        """Test metric with different data shapes."""
        logger.info("🧪 Testing StandardDeviation with different data shapes")
        
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

    def test_metric_with_zero_variance_data(self) -> None:
        """Test metric with zero variance data."""
        logger.info("🧪 Testing StandardDeviation with zero variance data")
        
        # Create data with zero variance
        y_pred = tf.constant([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=tf.float32)
        
        # Update metric
        self.metric.update_state(y_pred)
        
        # Check result (should be 0 for zero variance)
        result = self.metric.result()
        self.assertEqual(result.numpy(), 0.0)


if __name__ == "__main__":
    unittest.main()
