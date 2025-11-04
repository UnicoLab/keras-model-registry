"""Tests for TensorDimensionExpander layer."""

import unittest
import numpy as np
import keras
from kmr.layers import TensorDimensionExpander


class TestTensorDimensionExpander(unittest.TestCase):
    """Test suite for TensorDimensionExpander."""

    def test_initialization_default(self) -> None:
        """Test layer initialization with default parameters."""
        layer = TensorDimensionExpander()
        self.assertEqual(layer.axis, 1)

    def test_initialization_custom_axis(self) -> None:
        """Test layer initialization with custom axis."""
        for axis in [0, 1, 2, -1]:
            layer = TensorDimensionExpander(axis=axis)
            self.assertEqual(layer.axis, axis)

    def test_initialization_with_name(self) -> None:
        """Test layer initialization with custom name."""
        layer = TensorDimensionExpander(axis=1, name="expand_dims")
        self.assertEqual(layer.name, "expand_dims")

    def test_invalid_axis_type(self) -> None:
        """Test that invalid axis type raises error."""
        with self.assertRaises(ValueError):
            TensorDimensionExpander(axis="1")

    def test_expand_axis_1(self) -> None:
        """Test expanding dimension at axis 1."""
        layer = TensorDimensionExpander(axis=1)
        x = keras.random.normal((32, 10))
        y = layer(x)
        self.assertEqual(y.shape, (32, 1, 10))

    def test_expand_axis_0(self) -> None:
        """Test expanding dimension at axis 0."""
        layer = TensorDimensionExpander(axis=0)
        x = keras.random.normal((32, 10))
        y = layer(x)
        self.assertEqual(y.shape, (1, 32, 10))

    def test_expand_axis_negative(self) -> None:
        """Test expanding dimension at negative axis."""
        layer = TensorDimensionExpander(axis=-1)
        x = keras.random.normal((32, 10))
        y = layer(x)
        self.assertEqual(y.shape, (32, 10, 1))

    def test_expand_3d_input(self) -> None:
        """Test expanding dimensions on 3D input."""
        layer = TensorDimensionExpander(axis=2)
        x = keras.random.normal((32, 10, 5))
        y = layer(x)
        self.assertEqual(y.shape, (32, 10, 1, 5))

    def test_output_dtype_preserved(self) -> None:
        """Test that output dtype matches input dtype."""
        layer = TensorDimensionExpander(axis=1)
        x_float32 = keras.random.normal((20, 10), dtype="float32")
        y_float32 = layer(x_float32)
        self.assertEqual(y_float32.dtype, x_float32.dtype)

        x_float64 = keras.random.normal((20, 10), dtype="float64")
        y_float64 = layer(x_float64)
        self.assertEqual(y_float64.dtype, x_float64.dtype)

    def test_output_values_preserved(self) -> None:
        """Test that output values are preserved."""
        layer = TensorDimensionExpander(axis=1)
        x = keras.constant([[1.0, 2.0], [3.0, 4.0]])
        y = layer(x)
        expected = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_serialization_get_config(self) -> None:
        """Test layer serialization via get_config."""
        layer = TensorDimensionExpander(axis=2)
        config = layer.get_config()
        self.assertEqual(config["axis"], 2)

    def test_deserialization_from_config(self) -> None:
        """Test layer deserialization via from_config."""
        layer = TensorDimensionExpander(axis=2)
        config = layer.get_config()
        new_layer = TensorDimensionExpander.from_config(config)
        self.assertEqual(new_layer.axis, 2)

    def test_model_save_load(self) -> None:
        """Test that model with layer can be saved and loaded."""
        import tempfile

        layer = TensorDimensionExpander(axis=1)
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
            np.testing.assert_array_almost_equal(pred1, pred2)

    def test_multiple_expansions(self) -> None:
        """Test stacking multiple expanders."""
        layer1 = TensorDimensionExpander(axis=1)
        layer2 = TensorDimensionExpander(axis=2)

        x = keras.random.normal((32, 10))
        y = layer1(x)  # (32, 1, 10)
        z = layer2(y)  # (32, 1, 1, 10)

        self.assertEqual(z.shape, (32, 1, 1, 10))


if __name__ == "__main__":
    unittest.main()
