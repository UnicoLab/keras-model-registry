"""Unit tests for ReversibleInstanceNorm layer.

Note: TensorFlow is used in tests for validation purposes only.
The actual layer implementation uses only Keras 3 operations.
"""

import unittest
import tensorflow as tf  # Used for testing only
from keras import layers, Model
from kerasfactory.layers.ReversibleInstanceNorm import ReversibleInstanceNorm


class TestReversibleInstanceNorm(unittest.TestCase):
    """Test cases for ReversibleInstanceNorm layer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.batch_size = 16
        self.time_steps = 100
        self.n_features = 8
        self.layer = ReversibleInstanceNorm(num_features=self.n_features)
        self.inputs = tf.random.normal(
            (self.batch_size, self.time_steps, self.n_features),
        )

    def test_initialization(self) -> None:
        """Test layer initialization."""
        layer = ReversibleInstanceNorm(num_features=8)
        self.assertEqual(layer.num_features, 8)
        self.assertEqual(layer.eps, 1e-5)
        self.assertFalse(layer.affine)

    def test_invalid_initialization(self) -> None:
        """Test layer initialization with invalid parameters."""
        with self.assertRaises(ValueError):
            ReversibleInstanceNorm(num_features=0)

        with self.assertRaises(ValueError):
            ReversibleInstanceNorm(num_features=-1)

        with self.assertRaises(ValueError):
            ReversibleInstanceNorm(num_features=8, eps=-0.1)

    def test_normalization(self) -> None:
        """Test normalization functionality."""
        normalized = self.layer(self.inputs, mode="norm")

        # Normalized tensor should have mean close to 0 and std close to 1
        batch_mean = tf.reduce_mean(normalized, axis=1, keepdims=True)
        batch_std = tf.math.reduce_std(normalized, axis=1, keepdims=True)

        self.assertLess(float(tf.reduce_mean(tf.abs(batch_mean))), 0.1)
        self.assertGreater(float(tf.reduce_min(batch_std)), 0.8)
        self.assertLess(float(tf.reduce_max(batch_std)), 1.2)

    def test_denormalization(self) -> None:
        """Test denormalization reverses normalization."""
        normalized = self.layer(self.inputs, mode="norm")
        denormalized = self.layer(normalized, mode="denorm")

        tf.debugging.assert_near(denormalized, self.inputs, atol=1e-5)

    def test_affine_transformation(self) -> None:
        """Test affine transformation parameters."""
        layer = ReversibleInstanceNorm(num_features=self.n_features, affine=True)
        layer.build((None, self.time_steps, self.n_features))

        # Check that affine parameters were created
        self.assertIsNotNone(layer.affine_weight)
        self.assertIsNotNone(layer.affine_bias)

        # Affine weight should be initialized to ones
        expected_weight = tf.ones((self.n_features,))
        tf.debugging.assert_equal(layer.affine_weight, expected_weight)

    def test_subtract_last_mode(self) -> None:
        """Test normalization by last value instead of mean."""
        layer = ReversibleInstanceNorm(num_features=self.n_features, subtract_last=True)
        normalized = layer(self.inputs, mode="norm")

        # First value after normalization should relate to difference from last
        self.assertIsNotNone(normalized)

    def test_non_normalization_mode(self) -> None:
        """Test that non_norm flag skips normalization."""
        layer = ReversibleInstanceNorm(num_features=self.n_features, non_norm=True)
        output = layer(self.inputs, mode="norm")

        # Output should be identical to input when non_norm=True
        tf.debugging.assert_equal(output, self.inputs)

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        config = self.layer.get_config()
        self.assertIn("num_features", config)
        self.assertIn("eps", config)
        self.assertIn("affine", config)

        new_layer = ReversibleInstanceNorm.from_config(config)
        self.assertEqual(new_layer.num_features, self.layer.num_features)
        self.assertEqual(new_layer.eps, self.layer.eps)
        self.assertEqual(new_layer.affine, self.layer.affine)

    def test_model_integration(self) -> None:
        """Test layer integration in a model."""
        inputs = layers.Input(shape=(self.time_steps, self.n_features))
        normalized = self.layer(inputs, mode="norm")
        outputs = layers.Dense(1)(normalized)
        model = Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer="adam", loss="mse")

        x_data = tf.random.normal((10, self.time_steps, self.n_features))
        y_data = tf.random.normal((10, 1))

        history = model.fit(x_data, y_data, epochs=1, verbose=0)
        self.assertTrue(history.history["loss"][0] > 0)

    def test_different_batch_sizes(self) -> None:
        """Test with different batch sizes."""
        for batch_size in [1, 4, 16, 32]:
            inputs = tf.random.normal((batch_size, self.time_steps, self.n_features))
            normalized = self.layer(inputs, mode="norm")
            self.assertEqual(normalized.shape[0], batch_size)

    def test_reversibility_property(self) -> None:
        """Test reversibility property thoroughly."""
        for _ in range(5):
            random_input = tf.random.normal((4, 50, 8))
            normalized = self.layer(random_input, mode="norm")
            recovered = self.layer(normalized, mode="denorm")

            tf.debugging.assert_near(recovered, random_input, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
