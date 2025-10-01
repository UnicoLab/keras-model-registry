"""Unit tests for StochasticDepth layer."""
import unittest
import tensorflow as tf
from keras import layers, Model

from kmr.layers.StochasticDepth import StochasticDepth


class TestStochasticDepth(unittest.TestCase):
    """Test cases for StochasticDepth layer."""

    def setUp(self) -> None:
        """Set up test cases."""
        self.batch_size = 32
        self.height = 64
        self.width = 64
        self.channels = 128
        self.survival_prob = 0.8
        self.seed = 42

        self.layer = StochasticDepth(survival_prob=self.survival_prob, seed=self.seed)

    def test_initialization(self) -> None:
        """Test layer initialization."""
        self.assertEqual(self.layer.survival_prob, self.survival_prob)
        self.assertEqual(self.layer.seed, self.seed)

        # Test invalid survival_prob
        with self.assertRaises(ValueError):
            StochasticDepth(survival_prob=1.5)
        with self.assertRaises(ValueError):
            StochasticDepth(survival_prob=-0.1)

    def test_input_validation(self) -> None:
        """Test input validation."""
        inputs = tf.random.normal((self.batch_size, self.channels))

        # Test single input
        with self.assertRaises(ValueError):
            self.layer(inputs)

        # Test wrong number of inputs
        with self.assertRaises(ValueError):
            self.layer([inputs, inputs, inputs])

    def test_output_shape(self) -> None:
        """Test output shape of the layer."""
        # Test 2D inputs
        inputs_2d = tf.random.normal((self.batch_size, self.channels))
        residual_2d = tf.random.normal((self.batch_size, self.channels))
        outputs_2d = self.layer([inputs_2d, residual_2d])
        self.assertEqual(outputs_2d.shape, inputs_2d.shape)

        # Test 4D inputs
        inputs_4d = tf.random.normal(
            (self.batch_size, self.height, self.width, self.channels),
        )
        residual_4d = tf.random.normal(
            (self.batch_size, self.height, self.width, self.channels),
        )
        outputs_4d = self.layer([inputs_4d, residual_4d])
        self.assertEqual(outputs_4d.shape, inputs_4d.shape)

    def test_training_mode(self) -> None:
        """Test if the layer behaves correctly in training mode."""
        inputs = tf.ones((self.batch_size, self.channels))
        residual = tf.ones((self.batch_size, self.channels))

        # Test training mode
        outputs_train = self.layer([inputs, residual], training=True)

        # Some residual paths should be dropped
        self.assertLess(float(tf.reduce_mean(outputs_train)), 2.0)

        # Test inference mode
        outputs_test = self.layer([inputs, residual], training=False)

        # All paths should be kept but scaled
        expected = inputs + self.survival_prob * residual
        tf.debugging.assert_near(outputs_test, expected)

    def test_reproducibility(self) -> None:
        """Test if results are reproducible with same seed."""
        inputs = tf.ones((self.batch_size, self.channels))
        residual = tf.ones((self.batch_size, self.channels))

        # Create two layers with same seed
        layer1 = StochasticDepth(survival_prob=self.survival_prob, seed=self.seed)
        layer2 = StochasticDepth(survival_prob=self.survival_prob, seed=self.seed)

        # Outputs should be identical
        outputs1 = layer1([inputs, residual], training=True)
        outputs2 = layer2([inputs, residual], training=True)
        tf.debugging.assert_equal(outputs1, outputs2)

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        # Create a simple model with our layer
        inputs = layers.Input(shape=(self.channels,))
        residual = tf.ones((1, self.channels))  # Use constant residual
        outputs = self.layer([inputs, residual])
        model = Model(inputs=inputs, outputs=outputs)

        # Save and reload the model
        model_config = model.get_config()
        reloaded_model = Model.from_config(model_config)

        # Compare configurations
        self.assertEqual(model.get_config(), reloaded_model.get_config())

        # Test with same input (in inference mode to be deterministic)
        test_input = tf.ones((1, self.channels))
        original_output = model(test_input, training=False)
        reloaded_output = reloaded_model(test_input, training=False)
        tf.debugging.assert_near(original_output, reloaded_output)


if __name__ == "__main__":
    unittest.main()
