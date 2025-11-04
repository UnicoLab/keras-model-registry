"""Unit tests for ReversibleInstanceNormMultivariate layer.

Note: TensorFlow is used in tests for validation purposes only.
The actual layer implementation uses only Keras 3 operations.
"""

import unittest
import tensorflow as tf
import keras
from kmr.layers.ReversibleInstanceNormMultivariate import (
    ReversibleInstanceNormMultivariate,
)


class TestReversibleInstanceNormMultivariate(unittest.TestCase):
    """Test cases for ReversibleInstanceNormMultivariate layer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.num_features = 8
        self.batch_size = 16
        self.time_steps = 100
        self.layer = ReversibleInstanceNormMultivariate(
            num_features=self.num_features,
            eps=1e-5,
            affine=False,
        )
        self.layer_affine = ReversibleInstanceNormMultivariate(
            num_features=self.num_features,
            eps=1e-5,
            affine=True,
        )

    def test_initialization(self) -> None:
        """Test layer initialization."""
        layer = ReversibleInstanceNormMultivariate(
            num_features=8,
            eps=1e-5,
            affine=False,
        )
        self.assertEqual(layer.num_features, 8)
        self.assertEqual(layer.eps, 1e-5)
        self.assertEqual(layer.affine, False)

    def test_initialization_affine(self) -> None:
        """Test layer initialization with affine transform."""
        layer = ReversibleInstanceNormMultivariate(
            num_features=8,
            eps=1e-5,
            affine=True,
        )
        self.assertEqual(layer.affine, True)

    def test_invalid_parameters(self) -> None:
        """Test layer initialization with invalid parameters."""
        with self.assertRaises(ValueError):
            ReversibleInstanceNormMultivariate(num_features=0)

        with self.assertRaises(ValueError):
            ReversibleInstanceNormMultivariate(num_features=-1)

    def test_normalization_output_shape(self) -> None:
        """Test output shape for normalization."""
        x = tf.random.normal((self.batch_size, self.time_steps, self.num_features))
        outputs = self.layer(x, mode="norm")

        expected_shape = (self.batch_size, self.time_steps, self.num_features)
        self.assertEqual(tuple(outputs.shape), expected_shape)

    def test_denormalization_output_shape(self) -> None:
        """Test output shape for denormalization."""
        x = tf.random.normal((self.batch_size, self.time_steps, self.num_features))
        x_norm = self.layer(x, mode="norm")
        x_denorm = self.layer(x_norm, mode="denorm")

        expected_shape = (self.batch_size, self.time_steps, self.num_features)
        self.assertEqual(tuple(x_denorm.shape), expected_shape)

    def test_exact_reconstruction(self) -> None:
        """Test exact reconstruction after normalization and denormalization."""
        x = tf.random.normal((self.batch_size, self.time_steps, self.num_features))
        x_norm = self.layer(x, mode="norm")
        x_reconstructed = self.layer(x_norm, mode="denorm")

        # Reconstruction error should be very small
        error = float(tf.reduce_mean(tf.abs(x - x_reconstructed)))
        self.assertLess(error, 1e-4)

    def test_normalized_statistics(self) -> None:
        """Test that normalized data has expected statistics."""
        x = tf.random.normal((self.batch_size, self.time_steps, self.num_features))
        x_norm = self.layer(x, mode="norm")

        # Mean should be close to 0
        mean = float(tf.reduce_mean(x_norm))
        self.assertAlmostEqual(mean, 0.0, places=1)

        # Std should be close to 1
        std = float(tf.math.reduce_std(x_norm))
        self.assertGreater(std, 0.1)  # Should be non-trivial

    def test_affine_transform(self) -> None:
        """Test affine transform functionality."""
        # Test that affine layer has trainable parameters
        self.assertFalse(self.layer.affine)
        self.assertTrue(self.layer_affine.affine)

        # Build the layers
        x = tf.random.normal((self.batch_size, self.time_steps, self.num_features))
        self.layer(x, mode="norm")
        self.layer_affine(x, mode="norm")

        # Check that affine layer has gamma and beta parameters
        affine_params = self.layer_affine.trainable_weights
        self.assertGreater(
            len(affine_params),
            0,
            "Affine layer should have trainable parameters",
        )

    def test_consistency_across_calls(self) -> None:
        """Test that normalization is consistent across multiple calls."""
        x = tf.random.normal((self.batch_size, self.time_steps, self.num_features))

        outputs1 = self.layer(x, mode="norm")
        outputs2 = self.layer(x, mode="norm")

        # Outputs should be identical
        difference = float(tf.reduce_mean(tf.abs(outputs1 - outputs2)))
        self.assertAlmostEqual(difference, 0.0, places=5)

    def test_different_batch_sizes(self) -> None:
        """Test layer with different batch sizes."""
        for batch_size in [1, 8, 16, 32]:
            x = tf.random.normal((batch_size, self.time_steps, self.num_features))
            outputs = self.layer(x, mode="norm")

            expected_shape = (batch_size, self.time_steps, self.num_features)
            self.assertEqual(tuple(outputs.shape), expected_shape)

    def test_serialization(self) -> None:
        """Test layer serialization."""
        config = self.layer.get_config()
        self.assertEqual(config["num_features"], self.num_features)
        self.assertEqual(config["eps"], 1e-5)
        self.assertEqual(config["affine"], False)

        # Recreate from config
        new_layer = ReversibleInstanceNormMultivariate.from_config(config)
        self.assertEqual(new_layer.num_features, self.num_features)
        self.assertEqual(new_layer.affine, False)

    def test_model_integration(self) -> None:
        """Test layer integration in a model."""
        inputs = keras.Input(shape=(self.time_steps, self.num_features))
        outputs = self.layer(inputs, mode="norm")
        model = keras.Model(inputs=inputs, outputs=outputs)

        x = tf.random.normal((self.batch_size, self.time_steps, self.num_features))
        predictions = model.predict(x, verbose=0)

        expected_shape = (self.batch_size, self.time_steps, self.num_features)
        self.assertEqual(tuple(predictions.shape), expected_shape)

    def test_batched_statistics(self) -> None:
        """Test that statistics are computed across batch."""
        # Create data with different scales in batch
        batch1 = tf.random.normal((8, self.time_steps, self.num_features)) * 1.0
        batch2 = tf.random.normal((8, self.time_steps, self.num_features)) * 10.0
        x = tf.concat([batch1, batch2], axis=0)

        x_norm = self.layer(x, mode="norm")

        # All samples should have similar statistics after normalization
        mean_all = float(tf.reduce_mean(x_norm))
        self.assertAlmostEqual(mean_all, 0.0, places=1)

    def test_eps_stability(self) -> None:
        """Test epsilon for numerical stability."""
        # Create near-zero variance data
        x = tf.ones((self.batch_size, self.time_steps, self.num_features)) * 1e-6

        # Should not raise errors with small eps
        layer_small_eps = ReversibleInstanceNormMultivariate(
            num_features=self.num_features,
            eps=1e-8,
        )
        outputs = layer_small_eps(x, mode="norm")

        # Check for NaN or Inf
        self.assertFalse(tf.reduce_any(tf.math.is_nan(outputs)))
        self.assertFalse(tf.reduce_any(tf.math.is_inf(outputs)))

    def test_reconstruction_with_affine(self) -> None:
        """Test exact reconstruction with affine transform."""
        x = tf.random.normal((self.batch_size, self.time_steps, self.num_features))
        x_norm = self.layer_affine(x, mode="norm")
        x_reconstructed = self.layer_affine(x_norm, mode="denorm")

        # Reconstruction error should be small even with affine
        error = float(tf.reduce_mean(tf.abs(x - x_reconstructed)))
        self.assertLess(error, 1e-4)


if __name__ == "__main__":
    unittest.main()
