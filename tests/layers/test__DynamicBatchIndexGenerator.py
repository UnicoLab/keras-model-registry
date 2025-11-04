"""Tests for DynamicBatchIndexGenerator layer."""

import unittest
import numpy as np
import keras
from kmr.layers import DynamicBatchIndexGenerator


class TestDynamicBatchIndexGenerator(unittest.TestCase):
    """Test suite for DynamicBatchIndexGenerator."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.layer = DynamicBatchIndexGenerator()

    def test_initialization(self) -> None:
        """Test layer initialization with default parameters."""
        layer = DynamicBatchIndexGenerator()
        self.assertIsNotNone(layer)
        self.assertIsInstance(layer, DynamicBatchIndexGenerator)

    def test_initialization_with_name(self) -> None:
        """Test layer initialization with custom name."""
        layer = DynamicBatchIndexGenerator(name="test_batch_index")
        self.assertEqual(layer.name, "test_batch_index")

    def test_output_shape_batch_32(self) -> None:
        """Test output shape with batch size 32."""
        x = keras.random.normal((32, 10))
        indices = self.layer(x)
        self.assertEqual(indices.shape, (32,))

    def test_output_shape_batch_64(self) -> None:
        """Test output shape with batch size 64."""
        x = keras.random.normal((64, 20))
        indices = self.layer(x)
        self.assertEqual(indices.shape, (64,))

    def test_output_values(self) -> None:
        """Test that output contains correct sequential indices."""
        x = keras.random.normal((10, 5))
        indices = self.layer(x)
        expected = np.arange(10, dtype=x.dtype)
        np.testing.assert_array_equal(indices.numpy(), expected)

    def test_output_dtype_float32(self) -> None:
        """Test output dtype matches input dtype (float32)."""
        x = keras.random.normal((20, 10), dtype="float32")
        indices = self.layer(x)
        self.assertEqual(indices.dtype, x.dtype)

    def test_output_dtype_float64(self) -> None:
        """Test output dtype matches input dtype (float64)."""
        x = keras.random.normal((20, 10), dtype="float64")
        indices = self.layer(x)
        self.assertEqual(indices.dtype, x.dtype)

    def test_deterministic_output(self) -> None:
        """Test that output is deterministic."""
        x = keras.random.normal((15, 8))
        indices1 = self.layer(x).numpy()
        indices2 = self.layer(x).numpy()
        np.testing.assert_array_equal(indices1, indices2)

    def test_serialization_get_config(self) -> None:
        """Test layer serialization via get_config."""
        config = self.layer.get_config()
        self.assertIsInstance(config, dict)
        self.assertIn("name", config)

    def test_deserialization_from_config(self) -> None:
        """Test layer deserialization via from_config."""
        config = self.layer.get_config()
        new_layer = DynamicBatchIndexGenerator.from_config(config)
        self.assertIsInstance(new_layer, DynamicBatchIndexGenerator)

    def test_model_save_load(self) -> None:
        """Test that model with layer can be saved and loaded."""
        import tempfile

        # Create model
        inputs = keras.Input(shape=(10,))
        indices = self.layer(inputs)
        model = keras.Model(inputs, indices)

        # Create sample data
        x = keras.random.normal((32, 10))
        pred1 = model.predict(x, verbose=0)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = f"{tmpdir}/model.keras"
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

            # Verify predictions are identical
            pred2 = loaded_model.predict(x, verbose=0)
            np.testing.assert_array_equal(pred1, pred2)

    def test_different_input_ranks(self) -> None:
        """Test layer with inputs of different ranks."""
        # 2D input
        x2d = keras.random.normal((16, 10))
        indices2d = self.layer(x2d)
        self.assertEqual(indices2d.shape, (16,))

        # 3D input
        x3d = keras.random.normal((16, 10, 5))
        indices3d = self.layer(x3d)
        self.assertEqual(indices3d.shape, (16,))

        # 4D input
        x4d = keras.random.normal((16, 10, 5, 3))
        indices4d = self.layer(x4d)
        self.assertEqual(indices4d.shape, (16,))

    def test_batch_size_one(self) -> None:
        """Test with batch size of 1."""
        x = keras.random.normal((1, 10))
        indices = self.layer(x)
        self.assertEqual(indices.shape, (1,))
        self.assertEqual(indices.numpy()[0], 0)

    def test_large_batch_size(self) -> None:
        """Test with large batch size."""
        x = keras.random.normal((1000, 10))
        indices = self.layer(x)
        self.assertEqual(indices.shape, (1000,))
        expected = np.arange(1000, dtype=x.dtype)
        np.testing.assert_array_equal(indices.numpy(), expected)


if __name__ == "__main__":
    unittest.main()
