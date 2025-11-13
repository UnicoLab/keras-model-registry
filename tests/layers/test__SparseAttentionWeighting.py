"""Unit tests for SparseAttentionWeighting layer."""
import unittest
import tensorflow as tf
from keras import layers, Model, ops

from kerasfactory.layers.SparseAttentionWeighting import SparseAttentionWeighting


class TestSparseAttentionWeighting(unittest.TestCase):
    """Test cases for SparseAttentionWeighting layer."""

    def setUp(self) -> None:
        """Set up test cases."""
        self.num_modules = 3
        self.feature_dim = 10
        self.batch_size = 32
        self.temperature = 0.5
        self.layer = SparseAttentionWeighting(
            num_modules=self.num_modules,
            temperature=self.temperature,
        )

    def test_initialization(self) -> None:
        """Test layer initialization."""
        self.assertEqual(self.layer.num_modules, self.num_modules)
        self.assertEqual(self.layer.temperature, self.temperature)

        # Check if attention weights are initialized
        self.assertIsNotNone(self.layer.attention_weights)
        self.assertEqual(self.layer.attention_weights.shape, (self.num_modules,))

    def test_output_shape(self) -> None:
        """Test output shape of the layer."""
        # Create dummy module outputs
        module_outputs = [
            tf.random.normal((self.batch_size, self.feature_dim))
            for _ in range(self.num_modules)
        ]

        # Get layer output
        output = self.layer(module_outputs)

        # Check output shape (should be [batch_size, feature_dim])
        self.assertEqual(output.shape, (self.batch_size, self.feature_dim))

    def test_attention_weights(self) -> None:
        """Test if attention weights sum to 1 after softmax."""
        # Create dummy module outputs to trigger weight creation
        module_outputs = [
            tf.random.normal((1, self.feature_dim)) for _ in range(self.num_modules)
        ]

        # Forward pass to ensure weights are built
        self.layer(module_outputs)

        # Get attention probabilities (need to recreate the computation from the layer)
        attention_probs = ops.softmax(self.layer.attention_weights / self.temperature)

        # Check if probabilities sum to 1
        self.assertAlmostEqual(float(ops.sum(attention_probs)), 1.0, places=6)

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        # Create inputs and modules
        inputs = layers.Input(shape=(self.feature_dim,))
        module_outputs = [
            layers.Dense(self.feature_dim)(inputs) for _ in range(self.num_modules)
        ]

        # Create model with our layer
        outputs = self.layer(module_outputs)
        model = Model(inputs=inputs, outputs=outputs)

        # Train for one step to ensure weights are built
        x = tf.random.normal((self.batch_size, self.feature_dim))
        y = tf.random.normal((self.batch_size, self.feature_dim))
        model.compile(optimizer="adam", loss="mse")
        model.fit(x, y, epochs=1, verbose=0)

        # Save and reload the model
        model_config = model.get_config()
        reloaded_model = Model.from_config(model_config)
        reloaded_model.set_weights(model.get_weights())

        # Test with same input
        test_input = tf.random.normal((1, self.feature_dim))
        original_output = model(test_input)
        reloaded_output = reloaded_model(test_input)

        # Check if outputs are equal
        tf.debugging.assert_equal(original_output, reloaded_output)

    def test_temperature_scaling(self) -> None:
        """Test if temperature affects attention distribution."""
        # Create two layers with different temperatures
        layer_hot = SparseAttentionWeighting(
            num_modules=self.num_modules,
            temperature=0.1,
        )
        layer_cold = SparseAttentionWeighting(
            num_modules=self.num_modules,
            temperature=10.0,
        )

        # Create dummy module outputs
        module_outputs = [
            tf.random.normal((1, self.feature_dim)) for _ in range(self.num_modules)
        ]

        # Set different weights to test temperature effect
        initial_weights = tf.constant([1.0, 0.5, 0.1])
        layer_hot.attention_weights.assign(initial_weights)
        layer_cold.attention_weights.assign(initial_weights)

        # Get outputs from both layers
        _ = layer_hot(module_outputs)
        _ = layer_cold(module_outputs)

        # Get attention probabilities
        probs_hot = ops.softmax(layer_hot.attention_weights / 0.1)
        probs_cold = ops.softmax(layer_cold.attention_weights / 10.0)

        # Lower temperature should give more extreme probabilities
        self.assertGreater(float(ops.max(probs_hot)), float(ops.max(probs_cold)))


if __name__ == "__main__":
    unittest.main()
