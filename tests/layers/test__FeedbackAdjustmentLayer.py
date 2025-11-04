"""Tests for FeedbackAdjustmentLayer."""

import unittest
import numpy as np
import tensorflow as tf  # Used for testing only
import keras
from kmr.layers import FeedbackAdjustmentLayer


class TestFeedbackAdjustmentLayer(unittest.TestCase):
    """Test suite for FeedbackAdjustmentLayer."""

    def test_output_shape(self) -> None:
        """Test output shape."""
        layer = FeedbackAdjustmentLayer()
        predictions = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        feedback = tf.constant([[0.5, 1.0], [1.0, 0.5]])
        output = layer([predictions, feedback])
        self.assertEqual(output.shape, (2, 2))

    def test_feedback_multiplication(self) -> None:
        """Test that feedback is multiplied correctly."""
        layer = FeedbackAdjustmentLayer()
        predictions = tf.constant([[2.0, 4.0]])
        feedback = tf.constant([[0.5, 0.5]])
        output = layer([predictions, feedback]).numpy()
        expected = np.array([[1.0, 2.0]])
        np.testing.assert_array_almost_equal(output, expected)

    def test_zero_feedback(self) -> None:
        """Test with zero feedback."""
        layer = FeedbackAdjustmentLayer()
        predictions = tf.constant([[1.0, 2.0]])
        feedback = tf.constant([[0.0, 0.0]])
        output = layer([predictions, feedback]).numpy()
        expected = np.array([[0.0, 0.0]])
        np.testing.assert_array_almost_equal(output, expected)

    def test_unit_feedback(self) -> None:
        """Test with unit feedback (no change)."""
        layer = FeedbackAdjustmentLayer()
        predictions = tf.constant([[1.0, 2.0]])
        feedback = tf.constant([[1.0, 1.0]])
        output = layer([predictions, feedback]).numpy()
        np.testing.assert_array_almost_equal(output, predictions.numpy())

    def test_fractional_feedback(self) -> None:
        """Test with fractional feedback values."""
        layer = FeedbackAdjustmentLayer()
        predictions = tf.constant([[2.0, 4.0, 6.0]])
        feedback = tf.constant([[0.25, 0.5, 0.75]])
        output = layer([predictions, feedback]).numpy()
        expected = np.array([[0.5, 2.0, 4.5]])
        np.testing.assert_array_almost_equal(output, expected)

    def test_large_feedback_values(self) -> None:
        """Test with large feedback multipliers."""
        layer = FeedbackAdjustmentLayer()
        predictions = tf.constant([[1.0, 2.0]])
        feedback = tf.constant([[10.0, 100.0]])
        output = layer([predictions, feedback]).numpy()
        expected = np.array([[10.0, 200.0]])
        np.testing.assert_array_almost_equal(output, expected)

    def test_mixed_feedback_values(self) -> None:
        """Test with mixed positive, zero, and fractional feedback."""
        layer = FeedbackAdjustmentLayer()
        predictions = tf.constant([[1.0, 2.0, 3.0, 4.0]])
        feedback = tf.constant([[2.0, 0.5, 0.0, 1.0]])
        output = layer([predictions, feedback]).numpy()
        expected = np.array([[2.0, 1.0, 0.0, 4.0]])
        np.testing.assert_array_almost_equal(output, expected)

    def test_negative_predictions(self) -> None:
        """Test with negative prediction values."""
        layer = FeedbackAdjustmentLayer()
        predictions = tf.constant([[-1.0, -2.0]])
        feedback = tf.constant([[0.5, 2.0]])
        output = layer([predictions, feedback]).numpy()
        expected = np.array([[-0.5, -4.0]])
        np.testing.assert_array_almost_equal(output, expected)

    def test_negative_feedback(self) -> None:
        """Test with negative feedback values."""
        layer = FeedbackAdjustmentLayer()
        predictions = tf.constant([[1.0, 2.0]])
        feedback = tf.constant([[-1.0, -0.5]])
        output = layer([predictions, feedback]).numpy()
        expected = np.array([[-1.0, -1.0]])
        np.testing.assert_array_almost_equal(output, expected)

    def test_batch_independence(self) -> None:
        """Test that batch samples are independent."""
        layer = FeedbackAdjustmentLayer()
        predictions = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        feedback = tf.constant([[0.5, 1.0], [1.0, 2.0], [0.0, 0.5]])
        output = layer([predictions, feedback]).numpy()

        expected = np.array([[0.5, 2.0], [3.0, 8.0], [0.0, 3.0]])
        np.testing.assert_array_almost_equal(output, expected)

    def test_large_batch_size(self) -> None:
        """Test with large batch sizes."""
        layer = FeedbackAdjustmentLayer()
        predictions = keras.random.normal((256, 100))
        feedback = keras.random.uniform((256, 100))
        output = layer([predictions, feedback])
        self.assertEqual(output.shape, (256, 100))

    def test_single_feature(self) -> None:
        """Test with single feature dimension."""
        layer = FeedbackAdjustmentLayer()
        predictions = tf.constant([[5.0], [10.0], [15.0]])
        feedback = tf.constant([[2.0], [0.5], [0.0]])
        output = layer([predictions, feedback]).numpy()
        expected = np.array([[10.0], [5.0], [0.0]])
        np.testing.assert_array_almost_equal(output, expected)

    def test_many_features(self) -> None:
        """Test with many feature dimensions."""
        layer = FeedbackAdjustmentLayer()
        predictions = keras.random.normal((8, 1024))
        feedback = keras.random.uniform((8, 1024))
        output = layer([predictions, feedback])
        self.assertEqual(output.shape, (8, 1024))

    def test_output_shape_preserved(self) -> None:
        """Test that output shape is preserved."""
        layer = FeedbackAdjustmentLayer()
        for shape in [(10, 5), (32, 100), (1, 1000), (256, 1)]:
            predictions = keras.random.normal(shape)
            feedback = keras.random.uniform(shape)
            output = layer([predictions, feedback])
            self.assertEqual(output.shape, shape)

    def test_commutative_with_scaling(self) -> None:
        """Test that multiplication order doesn't affect result."""
        layer = FeedbackAdjustmentLayer()
        predictions = tf.constant([[2.0, 4.0]])
        feedback = tf.constant([[3.0, 0.5]])

        output = layer([predictions, feedback]).numpy()
        # Verify manual multiplication
        expected = predictions.numpy() * feedback.numpy()
        np.testing.assert_array_almost_equal(output, expected)

    def test_serialization(self) -> None:
        """Test serialization."""
        layer = FeedbackAdjustmentLayer()
        config = layer.get_config()
        new_layer = FeedbackAdjustmentLayer.from_config(config)
        self.assertIsNotNone(new_layer)

    def test_model_save_load(self) -> None:
        """Test model save and load."""
        import tempfile

        layer = FeedbackAdjustmentLayer()
        pred_input = keras.Input(shape=(10,))
        feedback_input = keras.Input(shape=(10,))
        output = layer([pred_input, feedback_input])
        model = keras.Model([pred_input, feedback_input], output)

        predictions = keras.random.normal((8, 10))
        feedback = keras.random.uniform((8, 10))
        pred1 = model.predict([predictions, feedback], verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(f"{tmpdir}/model.keras")
            loaded = keras.models.load_model(f"{tmpdir}/model.keras")
            pred2 = loaded.predict([predictions, feedback], verbose=0)
            np.testing.assert_array_almost_equal(pred1, pred2)


if __name__ == "__main__":
    unittest.main()
