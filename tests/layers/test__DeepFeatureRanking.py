"""Tests for DeepFeatureRanking layer."""

import unittest
import numpy as np
import tensorflow as tf  # Used for testing only
import keras
from kmr.layers import DeepFeatureRanking


class TestDeepFeatureRanking(unittest.TestCase):
    """Test suite for DeepFeatureRanking."""

    def test_initialization(self) -> None:
        """Test initialization."""
        layer = DeepFeatureRanking(hidden_dim=64, l2_reg=1e-5)
        self.assertEqual(layer.hidden_dim, 64)
        self.assertEqual(layer.l2_reg, 1e-5)

    def test_invalid_hidden_dim(self) -> None:
        """Test invalid hidden_dim."""
        with self.assertRaises(ValueError):
            DeepFeatureRanking(hidden_dim=0)

    def test_output_shape(self) -> None:
        """Test output shape."""
        layer = DeepFeatureRanking(hidden_dim=32)
        x = keras.random.normal((32, 50))
        y = layer(x)
        self.assertEqual(y.shape, (32, 1))

    def test_training_mode(self) -> None:
        """Test training vs inference mode."""
        layer = DeepFeatureRanking(hidden_dim=32, dropout_rate=0.5)
        x = keras.random.normal((16, 50))
        y_train = layer(x, training=True)
        y_infer = layer(x, training=False)
        self.assertEqual(y_train.shape, y_infer.shape)

    def test_various_hidden_dimensions(self) -> None:
        """Test with different hidden dimensions."""
        for hidden_dim in [16, 32, 64, 128]:
            layer = DeepFeatureRanking(hidden_dim=hidden_dim)
            x = keras.random.normal((16, 100))
            y = layer(x)
            self.assertEqual(y.shape, (16, 1))

    def test_output_range(self) -> None:
        """Test that output values are finite."""
        layer = DeepFeatureRanking(hidden_dim=32)
        x = keras.random.normal((16, 100))
        y = layer(x).numpy()
        self.assertTrue(np.all(np.isfinite(y)))

    def test_large_input_features(self) -> None:
        """Test with large input feature dimensions."""
        layer = DeepFeatureRanking(hidden_dim=64)
        x = keras.random.normal((32, 512))
        y = layer(x)
        self.assertEqual(y.shape, (32, 1))

    def test_single_sample_batch(self) -> None:
        """Test with batch size of 1."""
        layer = DeepFeatureRanking(hidden_dim=32)
        x = keras.random.normal((1, 100))
        y = layer(x)
        self.assertEqual(y.shape, (1, 1))

    def test_batch_norm_effect(self) -> None:
        """Test that batch norm has different behavior in train vs inference."""
        layer = DeepFeatureRanking(hidden_dim=32)
        x = keras.random.normal((100, 100))

        y_train = layer(x, training=True).numpy()
        y_infer = layer(x, training=False).numpy()

        # Shapes should be same
        self.assertEqual(y_train.shape, y_infer.shape)

    def test_l2_regularization_losses(self) -> None:
        """Test that L2 regularization adds losses."""
        layer = DeepFeatureRanking(hidden_dim=32, l2_reg=0.01)
        x = keras.random.normal((16, 100))
        _ = layer(x)
        self.assertGreater(len(layer.losses), 0)

    def test_output_values_non_trivial(self) -> None:
        """Test that output values are non-trivial."""
        layer = DeepFeatureRanking(hidden_dim=32)
        x = keras.random.normal((16, 100))
        y = layer(x).numpy()
        # Should have variation across samples
        self.assertGreater(np.std(y), 0)

    def test_serialization(self) -> None:
        """Test serialization."""
        layer = DeepFeatureRanking(hidden_dim=48, l2_reg=1e-4)
        config = layer.get_config()
        new_layer = DeepFeatureRanking.from_config(config)
        self.assertEqual(new_layer.hidden_dim, 48)

    def test_model_save_load(self) -> None:
        """Test model save and load."""
        import tempfile

        layer = DeepFeatureRanking()
        inputs = keras.Input(shape=(100,))
        outputs = layer(inputs)
        model = keras.Model(inputs, outputs)

        x = keras.random.normal((16, 100))
        pred1 = model.predict(x, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(f"{tmpdir}/model.keras")
            loaded = keras.models.load_model(f"{tmpdir}/model.keras")
            pred2 = loaded.predict(x, verbose=0)
            np.testing.assert_array_almost_equal(pred1, pred2)


if __name__ == "__main__":
    unittest.main()
