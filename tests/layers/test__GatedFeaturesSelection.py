"""Unit tests for GatedFeatureSelection layer."""
import unittest
import tensorflow as tf
import keras
from keras import layers, Model

from kerasfactory.layers.GatedFeaturesSelection import GatedFeatureSelection


class TestGatedFeatureSelection(unittest.TestCase):
    """Test cases for GatedFeatureSelection layer."""

    def setUp(self) -> None:
        """Set up test cases."""
        self.input_dim = 10
        self.batch_size = 32
        self.layer = GatedFeatureSelection(input_dim=self.input_dim)

    def test_initialization(self) -> None:
        """Test layer initialization."""
        self.assertEqual(self.layer.input_dim, self.input_dim)
        self.assertEqual(self.layer.reduction_ratio, 4)  # default value

    def test_output_shape(self) -> None:
        """Test output shape of the layer."""
        inputs = tf.random.normal((self.batch_size, self.input_dim))
        outputs = self.layer(inputs)

        self.assertEqual(outputs.shape, (self.batch_size, self.input_dim))

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        # Create a simple model with our layer
        inputs = layers.Input(shape=(self.input_dim,))
        outputs = self.layer(inputs)
        model = Model(inputs=inputs, outputs=outputs)

        # Train the model for one step to ensure weights are built
        x = tf.random.normal((self.batch_size, self.input_dim))
        y = tf.random.normal((self.batch_size, self.input_dim))
        model.compile(optimizer="adam", loss="mse")
        model.fit(x, y, epochs=1, verbose=0)

        # Save and reload the model
        model_config = model.get_config()
        reloaded_model = Model.from_config(model_config)

        # Set the weights from the original model
        reloaded_model.set_weights(model.get_weights())

        # Test with same input
        test_input = tf.random.normal((1, self.input_dim))
        original_output = model(test_input)
        reloaded_output = reloaded_model(test_input)

        # Check if outputs are exactly equal since we copied weights
        tf.debugging.assert_equal(original_output, reloaded_output)

    def test_residual_connection(self) -> None:
        """Test if residual connection is working properly."""
        inputs = tf.ones((1, self.input_dim))
        outputs = self.layer(inputs)

        # Output should be different from input due to gating and residual
        tf.debugging.assert_none_equal(inputs, outputs)

        # But should maintain the same shape
        self.assertEqual(inputs.shape, outputs.shape)

    def test_model_integration(self) -> None:
        """Test layer integration in a model."""
        # Create a simple model
        model = keras.Sequential(
            [layers.Input(shape=(self.input_dim,)), self.layer, layers.Dense(1)],
        )

        # Ensure model can be compiled and trained
        model.compile(optimizer="adam", loss="mse")

        # Generate dummy data
        x = tf.random.normal((100, self.input_dim))
        y = tf.random.normal((100, 1))

        # Train for one epoch
        history = model.fit(x, y, epochs=1, verbose=0)
        self.assertTrue(history.history["loss"][0] > 0)


if __name__ == "__main__":
    unittest.main()
