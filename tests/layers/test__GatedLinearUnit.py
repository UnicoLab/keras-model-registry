"""Unit tests for GatedLinearUnit layer.

Note: TensorFlow is used in tests for validation purposes only.
The actual layer implementation uses only Keras 3 operations.
"""

import unittest
import tensorflow as tf  # Used for testing only
from keras import layers, Model, Sequential
from kerasfactory.layers.GatedLinearUnit import GatedLinearUnit


class TestGatedLinearUnit(unittest.TestCase):
    """Test cases for GatedLinearUnit layer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.batch_size = 32
        self.input_dim = 16
        self.units = 8
        self.layer = GatedLinearUnit(units=self.units)
        # Using TensorFlow for test data generation only
        self.inputs = tf.random.normal((self.batch_size, self.input_dim))
        tf.random.set_seed(42)  # For reproducibility

    def test_initialization(self) -> None:
        """Test layer initialization with various parameters."""
        # Test default initialization
        layer = GatedLinearUnit(units=self.units)
        self.assertEqual(layer.units, self.units)

        # Test with different units
        units = 16
        layer = GatedLinearUnit(units=units)
        self.assertEqual(layer.units, units)

    def test_invalid_initialization(self) -> None:
        """Test layer initialization with invalid parameters."""
        # Test with negative units
        with self.assertRaises(ValueError):
            GatedLinearUnit(units=-1)

        # Test with zero units
        with self.assertRaises(ValueError):
            GatedLinearUnit(units=0)

        # Test with non-integer units
        with self.assertRaises(ValueError):
            GatedLinearUnit(units=3.5)

    def test_build(self) -> None:
        """Test layer building."""
        # Build the layer
        self.layer.build((None, self.input_dim))

        # Check if linear and sigmoid layers are created
        self.assertIsNotNone(self.layer.linear)
        self.assertIsNotNone(self.layer.sigmoid)

        # Check if linear layer has the correct units
        self.assertEqual(self.layer.linear.units, self.units)

        # Check if sigmoid layer has the correct units and activation
        self.assertEqual(self.layer.sigmoid.units, self.units)
        self.assertEqual(self.layer.sigmoid.activation.__name__, "sigmoid")

    def test_output_shape(self) -> None:
        """Test output shape of the layer."""
        outputs = self.layer(self.inputs)

        # Check output shape
        self.assertEqual(outputs.shape, (self.batch_size, self.units))

        # Test with different input shapes
        test_shapes = [(16, 32), (64, 8), (128, 64)]
        for batch, features in test_shapes:
            inputs = tf.random.normal((batch, features))
            layer = GatedLinearUnit(units=self.units)
            outputs = layer(inputs)
            self.assertEqual(outputs.shape, (batch, self.units))

    def test_call(self) -> None:
        """Test forward pass of the layer."""
        # Call the layer
        outputs = self.layer(self.inputs)

        # Check that outputs are not None
        self.assertIsNotNone(outputs)

        # Check that outputs are not all zeros
        self.assertFalse(tf.reduce_all(tf.equal(outputs, 0)))

    def test_training_mode(self) -> None:
        """Test layer behavior in training and inference modes."""
        # For this layer, training mode doesn't affect the output
        # But we test it for completeness
        output_train = self.layer(self.inputs, training=True)
        output_infer = self.layer(self.inputs, training=False)

        # Outputs should be the same regardless of training mode
        self.assertTrue(tf.reduce_all(tf.equal(output_train, output_infer)))

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        # Create a simple model with our layer
        inputs = layers.Input(shape=(self.input_dim,))
        outputs = self.layer(inputs)
        model = Model(inputs=inputs, outputs=outputs)

        # Train the model for one step to ensure weights are built
        x = tf.random.normal((self.batch_size, self.input_dim))
        y = tf.random.normal((self.batch_size, self.units))
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

    def test_model_integration(self) -> None:
        """Test layer integration in a model."""
        # Create a simple sequential model
        model = Sequential([layers.Input(shape=(self.input_dim,)), self.layer])

        # Ensure model can be compiled and trained
        model.compile(optimizer="adam", loss="mse")

        # Generate dummy data
        x = tf.random.normal((100, self.input_dim))
        y = tf.random.normal((100, self.units))

        # Train for one epoch
        history = model.fit(x, y, epochs=1, verbose=0)
        self.assertTrue(history.history["loss"][0] > 0)

    def test_gating_mechanism(self) -> None:
        """Test that the gating mechanism works as expected."""
        # Create a layer with controlled weights
        layer = GatedLinearUnit(units=1)

        # Call the layer once to build it
        _ = layer(tf.zeros((1, 1)))

        # Set weights for linear and sigmoid layers
        # Linear: weight=1, bias=0 -> output = input
        # Sigmoid: weight=0, bias=0 -> output = 0.5
        layer.linear.set_weights([tf.ones((1, 1)), tf.zeros((1,))])
        layer.sigmoid.set_weights([tf.zeros((1, 1)), tf.zeros((1,))])

        # Test with a simple input
        test_input = tf.constant([[2.0]])
        output = layer(test_input)

        # Expected: linear(2.0) * sigmoid(2.0) = 2.0 * 0.5 = 1.0
        expected = tf.constant([[1.0]])

        # Check that output is close to expected
        self.assertTrue(tf.reduce_all(tf.abs(output - expected) < 1e-5))


if __name__ == "__main__":
    unittest.main()
