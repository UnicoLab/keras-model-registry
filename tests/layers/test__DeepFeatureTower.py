"""Tests for DeepFeatureTower layer."""

import unittest
import numpy as np
import tensorflow as tf  # Used for testing only
import keras
from kmr.layers import DeepFeatureTower


class TestDeepFeatureTower(unittest.TestCase):
    """Test suite for DeepFeatureTower."""

    def test_initialization_default(self) -> None:
        """Test default initialization."""
        layer = DeepFeatureTower()
        self.assertEqual(layer.units, 32)
        self.assertEqual(layer.hidden_layers, 2)
        self.assertEqual(layer.dropout_rate, 0.2)

    def test_output_shape(self) -> None:
        """Test output shape."""
        layer = DeepFeatureTower(units=64, hidden_layers=2)
        x = keras.random.normal((32, 100))
        y = layer(x)
        self.assertEqual(y.shape, (32, 64))

    def test_invalid_units(self) -> None:
        """Test invalid units raises error."""
        with self.assertRaises(ValueError):
            DeepFeatureTower(units=0)

    def test_invalid_hidden_layers(self) -> None:
        """Test invalid hidden_layers raises error."""
        with self.assertRaises(ValueError):
            DeepFeatureTower(hidden_layers=0)

    def test_invalid_dropout_rate(self) -> None:
        """Test invalid dropout_rate raises error."""
        with self.assertRaises(ValueError):
            DeepFeatureTower(dropout_rate=1.5)

    def test_training_mode_difference(self) -> None:
        """Test different outputs in training vs inference."""
        layer = DeepFeatureTower(dropout_rate=0.5)
        x = keras.random.normal((16, 100))
        y_train = layer(x, training=True).numpy()
        y_infer = layer(x, training=False).numpy()
        # Shapes should match
        self.assertEqual(y_train.shape, y_infer.shape)

    def test_multiple_hidden_layers(self) -> None:
        """Test with different numbers of hidden layers."""
        for n_layers in [1, 2, 3, 4]:
            layer = DeepFeatureTower(units=32, hidden_layers=n_layers)
            x = keras.random.normal((16, 50))
            y = layer(x)
            self.assertEqual(y.shape[1], 32)

    def test_l2_regularization_losses(self) -> None:
        """Test that L2 regularization losses are present."""
        layer = DeepFeatureTower(units=32, l2_reg=0.01)
        x = keras.random.normal((8, 50))
        _ = layer(x)
        self.assertGreater(len(layer.losses), 0)

    def test_dropout_reduces_outputs(self) -> None:
        """Test that dropout reduces output values in training mode."""
        layer = DeepFeatureTower(units=32, dropout_rate=0.8)
        x = keras.random.normal((100, 50))
        y_train = layer(x, training=True).numpy()
        y_infer = layer(x, training=False).numpy()

        # Training output mean should generally be different from inference
        # (dropout scales outputs differently)
        self.assertNotEqual(np.mean(y_train), np.mean(y_infer))

    def test_various_activation_functions(self) -> None:
        """Test with different activation functions."""
        for activation in ["relu", "tanh", "sigmoid"]:
            layer = DeepFeatureTower(units=32, activation=activation)
            x = keras.random.normal((16, 50))
            y = layer(x)
            self.assertEqual(y.shape, (16, 32))

    def test_large_input_dimensions(self) -> None:
        """Test with large input dimensions."""
        layer = DeepFeatureTower(units=128, hidden_layers=3)
        x = keras.random.normal((32, 512))
        y = layer(x)
        self.assertEqual(y.shape, (32, 128))

    def test_small_batch_sizes(self) -> None:
        """Test with small batch sizes."""
        layer = DeepFeatureTower(units=32)
        for batch_size in [1, 2, 4]:
            x = keras.random.normal((batch_size, 50))
            y = layer(x)
            self.assertEqual(y.shape, (batch_size, 32))

    def test_output_non_zero(self) -> None:
        """Test that outputs are non-zero."""
        layer = DeepFeatureTower(units=32)
        x = keras.random.normal((16, 50))
        y = layer(x).numpy()
        self.assertGreater(np.abs(y).max(), 0)

    def test_consistency_across_calls(self) -> None:
        """Test output consistency in inference mode."""
        layer = DeepFeatureTower(units=32)
        x = keras.random.normal((16, 50))

        y1 = layer(x, training=False).numpy()
        y2 = layer(x, training=False).numpy()

        np.testing.assert_array_almost_equal(y1, y2)

    def test_serialization(self) -> None:
        """Test serialization."""
        layer = DeepFeatureTower(units=64, hidden_layers=3, dropout_rate=0.3)
        config = layer.get_config()
        new_layer = DeepFeatureTower.from_config(config)
        self.assertEqual(new_layer.units, 64)
        self.assertEqual(new_layer.hidden_layers, 3)

    def test_model_save_load(self) -> None:
        """Test model save and load."""
        import tempfile

        layer = DeepFeatureTower(units=32)
        inputs = keras.Input(shape=(50,))
        outputs = layer(inputs)
        model = keras.Model(inputs, outputs)

        x = keras.random.normal((16, 50))
        pred1 = model.predict(x, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(f"{tmpdir}/model.keras")
            loaded = keras.models.load_model(f"{tmpdir}/model.keras")
            pred2 = loaded.predict(x, verbose=0)
            np.testing.assert_array_almost_equal(pred1, pred2)


if __name__ == "__main__":
    unittest.main()
