"""Unit tests for FeatureCutout layer."""
import unittest
import tensorflow as tf
import keras
from keras import layers, Model, ops

from kmr.layers.FeatureCutout import FeatureCutout


class TestFeatureCutout(unittest.TestCase):
    """Test cases for FeatureCutout layer."""

    def setUp(self) -> None:
        """Set up test cases."""
        self.feature_dim = 10
        self.batch_size = 32
        self.cutout_prob = 0.2
        self.noise_value = -1.0
        self.seed = 42
        
        self.layer = FeatureCutout(
            cutout_prob=self.cutout_prob,
            noise_value=self.noise_value,
            seed=self.seed
        )
        
        # Build the layer
        self.layer.build((self.batch_size, self.feature_dim))

    def test_initialization(self) -> None:
        """Test layer initialization."""
        self.assertEqual(self.layer.cutout_prob, self.cutout_prob)
        self.assertEqual(self.layer.noise_value, self.noise_value)
        self.assertEqual(self.layer.seed, self.seed)
        
        # Test invalid cutout_prob
        with self.assertRaises(ValueError):
            FeatureCutout(cutout_prob=1.5)
        with self.assertRaises(ValueError):
            FeatureCutout(cutout_prob=-0.1)

    def test_build_validation(self) -> None:
        """Test input validation during build."""
        layer = FeatureCutout()
        
        # Test invalid rank
        with self.assertRaises(ValueError):
            layer.build((self.batch_size, self.feature_dim, 1))

    def test_output_shape(self) -> None:
        """Test output shape of the layer."""
        inputs = tf.random.normal((self.batch_size, self.feature_dim))
        outputs = self.layer(inputs)
        
        # Check output shape (should match input shape)
        self.assertEqual(outputs.shape, inputs.shape)

    def test_training_mode(self) -> None:
        """Test if the layer behaves correctly in training mode."""
        inputs = tf.ones((self.batch_size, self.feature_dim))
        
        # Test training mode
        outputs_train = self.layer(inputs, training=True)
        
        # Some features should be masked (not all ones)
        self.assertLess(float(ops.mean(outputs_train)), 1.0)
        
        # Test inference mode
        outputs_test = self.layer(inputs, training=False)
        
        # All features should be kept (all ones)
        tf.debugging.assert_equal(outputs_test, inputs)

    def test_noise_value(self) -> None:
        """Test if noise value is correctly applied."""
        inputs = tf.ones((self.batch_size, self.feature_dim))
        outputs = self.layer(inputs, training=True)
        
        # Check if masked values are equal to noise_value
        mask = ops.cast(outputs != self.noise_value, tf.float32)
        masked_ratio = 1.0 - float(ops.mean(mask))
        
        # Ratio of masked values should be close to cutout_prob
        self.assertAlmostEqual(
            masked_ratio,
            self.cutout_prob,
            delta=0.1
        )

    def test_reproducibility(self) -> None:
        """Test if results are reproducible with same seed."""
        inputs = tf.ones((self.batch_size, self.feature_dim))
        
        # Create two layers with same seed
        layer1 = FeatureCutout(
            cutout_prob=self.cutout_prob,
            seed=self.seed
        )
        layer2 = FeatureCutout(
            cutout_prob=self.cutout_prob,
            seed=self.seed
        )
        
        # Outputs should be identical
        outputs1 = layer1(inputs, training=True)
        outputs2 = layer2(inputs, training=True)
        tf.debugging.assert_equal(outputs1, outputs2)

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        # Create a simple model with our layer
        inputs = layers.Input(shape=(self.feature_dim,))
        outputs = self.layer(inputs)
        model = Model(inputs=inputs, outputs=outputs)

        # Save and reload the model
        model_config = model.get_config()
        reloaded_model = Model.from_config(model_config)
        
        # Compare configurations
        self.assertEqual(
            model.get_config(),
            reloaded_model.get_config()
        )
        
        # Test with same input (in inference mode to be deterministic)
        test_input = tf.ones((1, self.feature_dim))
        original_output = model(test_input, training=False)
        reloaded_output = reloaded_model(test_input, training=False)
        tf.debugging.assert_equal(original_output, reloaded_output)


if __name__ == '__main__':
    unittest.main()