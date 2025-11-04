"""Tests for ThresholdBasedMasking layer."""

import unittest
import numpy as np
import keras
from kmr.layers import ThresholdBasedMasking


class TestThresholdBasedMasking(unittest.TestCase):
    """Test suite for ThresholdBasedMasking."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.layer = ThresholdBasedMasking(threshold=0.5)

    def test_initialization_default(self) -> None:
        """Test layer initialization with default threshold."""
        layer = ThresholdBasedMasking()
        self.assertEqual(layer.threshold, 0.0)

    def test_initialization_custom_threshold(self) -> None:
        """Test layer initialization with custom threshold."""
        layer = ThresholdBasedMasking(threshold=0.5)
        self.assertEqual(layer.threshold, 0.5)

    def test_initialization_negative_threshold(self) -> None:
        """Test layer initialization with negative threshold."""
        layer = ThresholdBasedMasking(threshold=-1.0)
        self.assertEqual(layer.threshold, -1.0)

    def test_initialization_with_name(self) -> None:
        """Test layer initialization with custom name."""
        layer = ThresholdBasedMasking(threshold=0.5, name="masking")
        self.assertEqual(layer.name, "masking")

    def test_invalid_threshold_type(self) -> None:
        """Test that invalid threshold type raises error."""
        with self.assertRaises(ValueError):
            ThresholdBasedMasking(threshold="0.5")

    def test_masking_values_above_threshold(self) -> None:
        """Test that values above threshold are preserved."""
        layer = ThresholdBasedMasking(threshold=0.0)
        x = keras.constant([[1.0, 2.0], [3.0, 4.0]])
        y = layer(x)
        np.testing.assert_array_equal(y.numpy(), x.numpy())

    def test_masking_values_below_threshold(self) -> None:
        """Test that values below threshold are zeroed."""
        layer = ThresholdBasedMasking(threshold=1.5)
        x = keras.constant([[1.0, 2.0], [0.5, 4.0]])
        y = layer(x)
        expected = np.array([[0.0, 2.0], [0.0, 4.0]])
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_masking_exact_threshold(self) -> None:
        """Test behavior at exact threshold value."""
        layer = ThresholdBasedMasking(threshold=1.0)
        x = keras.constant([[0.9, 1.0], [1.1, 2.0]])
        y = layer(x)
        # Values >= threshold are kept
        expected = np.array([[0.0, 1.0], [1.1, 2.0]])
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_output_shape_preserved(self) -> None:
        """Test that output shape matches input shape."""
        x = keras.random.normal((32, 10))
        y = self.layer(x)
        self.assertEqual(y.shape, x.shape)

    def test_output_dtype_preserved(self) -> None:
        """Test that output dtype matches input dtype."""
        x_float32 = keras.random.normal((20, 10), dtype="float32")
        y_float32 = self.layer(x_float32)
        self.assertEqual(y_float32.dtype, x_float32.dtype)

        x_float64 = keras.random.normal((20, 10), dtype="float64")
        y_float64 = self.layer(x_float64)
        self.assertEqual(y_float64.dtype, x_float64.dtype)

    def test_negative_values_masked(self) -> None:
        """Test masking with negative values and positive threshold."""
        layer = ThresholdBasedMasking(threshold=0.0)
        x = keras.constant([[-1.0, 0.5], [-0.5, 1.0]])
        y = layer(x)
        expected = np.array([[0.0, 0.5], [0.0, 1.0]])
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_all_values_masked(self) -> None:
        """Test when all values are below threshold."""
        layer = ThresholdBasedMasking(threshold=10.0)
        x = keras.constant([[1.0, 2.0], [3.0, 4.0]])
        y = layer(x)
        expected = np.zeros_like(x.numpy())
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_no_values_masked(self) -> None:
        """Test when no values are below threshold."""
        layer = ThresholdBasedMasking(threshold=-10.0)
        x = keras.constant([[1.0, 2.0], [3.0, 4.0]])
        y = layer(x)
        np.testing.assert_array_equal(y.numpy(), x.numpy())

    def test_2d_input(self) -> None:
        """Test with 2D input."""
        layer = ThresholdBasedMasking(threshold=0.0)
        x = keras.random.normal((32, 10))
        y = layer(x)
        self.assertEqual(y.shape, (32, 10))

    def test_3d_input(self) -> None:
        """Test with 3D input."""
        layer = ThresholdBasedMasking(threshold=0.0)
        x = keras.random.normal((32, 10, 5))
        y = layer(x)
        self.assertEqual(y.shape, (32, 10, 5))

    def test_serialization_get_config(self) -> None:
        """Test layer serialization via get_config."""
        layer = ThresholdBasedMasking(threshold=0.7)
        config = layer.get_config()
        self.assertEqual(config["threshold"], 0.7)

    def test_deserialization_from_config(self) -> None:
        """Test layer deserialization via from_config."""
        layer = ThresholdBasedMasking(threshold=0.7)
        config = layer.get_config()
        new_layer = ThresholdBasedMasking.from_config(config)
        self.assertEqual(new_layer.threshold, 0.7)

    def test_model_save_load(self) -> None:
        """Test that model with layer can be saved and loaded."""
        import tempfile

        layer = ThresholdBasedMasking(threshold=0.5)
        inputs = keras.Input(shape=(10,))
        outputs = layer(inputs)
        model = keras.Model(inputs, outputs)

        x = keras.random.normal((32, 10))
        pred1 = model.predict(x, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = f"{tmpdir}/model.keras"
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)
            pred2 = loaded_model.predict(x, verbose=0)
            np.testing.assert_array_equal(pred1, pred2)


if __name__ == "__main__":
    unittest.main()
