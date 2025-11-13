"""Unit tests for RowAttention layer."""
import unittest
import tensorflow as tf
from keras import layers, Model, ops

from kerasfactory.layers.RowAttention import RowAttention


class TestRowAttention(unittest.TestCase):
    """Test cases for RowAttention layer."""

    def setUp(self) -> None:
        """Set up test cases."""
        self.feature_dim = 10
        self.batch_size = 32
        self.layer = RowAttention(feature_dim=self.feature_dim)

    def test_initialization(self) -> None:
        """Test layer initialization."""
        self.assertEqual(self.layer.feature_dim, self.feature_dim)
        self.assertEqual(self.layer.hidden_dim, max(self.feature_dim // 2, 1))

        # Check if attention network is initialized correctly
        self.assertIsNotNone(self.layer.attention_net)
        self.assertEqual(len(self.layer.attention_net.layers), 3)  # Dense, BN, Dense

        # Check last layer output dimension and activation
        last_layer = self.layer.attention_net.layers[-1]
        self.assertEqual(last_layer.units, 1)
        self.assertEqual(last_layer.activation.__name__, "sigmoid")

    def test_build_validation(self) -> None:
        """Test input validation during build."""
        # Test invalid rank
        with self.assertRaises(ValueError):
            self.layer.build((self.batch_size, self.feature_dim, 1))

        # Test invalid feature dimension
        with self.assertRaises(ValueError):
            self.layer.build((self.batch_size, self.feature_dim + 1))

    def test_output_shape(self) -> None:
        """Test output shape of the layer."""
        inputs = tf.random.normal((self.batch_size, self.feature_dim))
        outputs = self.layer(inputs)

        # Check output shape (should match input shape)
        self.assertEqual(outputs.shape, inputs.shape)

    def test_attention_weights(self) -> None:
        """Test if attention weights are properly normalized."""
        # Create input data
        inputs = tf.random.normal((self.batch_size, self.feature_dim))

        # Get attention weights from the network
        attention_weights = self.layer.attention_net(inputs)

        # Check shape
        self.assertEqual(attention_weights.shape, (self.batch_size, 1))

        # Check if weights are non-negative and bounded
        tf.debugging.assert_greater_equal(
            attention_weights,
            tf.zeros_like(attention_weights),
        )
        tf.debugging.assert_less_equal(
            attention_weights,
            tf.ones_like(attention_weights),
        )

    def test_sample_importance(self) -> None:
        """Test if the layer can identify important samples."""
        # Create input with one dominant sample
        inputs = tf.random.normal((self.batch_size, self.feature_dim))
        dominant_sample = 5.0 * tf.ones((1, self.feature_dim))
        inputs = tf.concat([dominant_sample, inputs[1:]], axis=0)

        # Get output
        outputs = self.layer(inputs)

        # The first sample should have higher importance
        self.assertGreater(float(ops.mean(outputs[0])), float(ops.mean(outputs[1:])))

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        # Create a simple model with our layer
        inputs = layers.Input(shape=(self.feature_dim,))
        outputs = self.layer(inputs)
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

    def test_training_mode(self) -> None:
        """Test if the layer behaves correctly in training mode."""
        # Create input data with specific pattern
        inputs = tf.ones((1, self.feature_dim))
        inputs = inputs * tf.reshape(
            tf.range(self.feature_dim, dtype=tf.float32),
            (1, -1),
        )

        # Get attention weights in both modes
        with tf.GradientTape() as tape:
            outputs = self.layer(inputs, training=True)
            loss = ops.mean(outputs)

        # Check if gradients can be computed (layer is trainable)
        grads = tape.gradient(loss, self.layer.trainable_weights)
        self.assertTrue(all(g is not None for g in grads))

        # Check if attention weights are properly normalized
        attention = self.layer.attention_net(inputs, training=False)
        tf.debugging.assert_greater_equal(attention, tf.zeros_like(attention))
        tf.debugging.assert_less_equal(attention, tf.ones_like(attention))


if __name__ == "__main__":
    unittest.main()
