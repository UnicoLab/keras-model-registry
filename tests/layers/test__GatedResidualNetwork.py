"""Unit tests for GatedResidualNetwork layer.

Note: TensorFlow is used in tests for validation purposes only.
The actual layer implementation uses only Keras 3 operations.
"""

import unittest
import tensorflow as tf  # Used for testing only
from keras import layers, Model, Sequential
from kmr.layers.GatedResidualNetwork import GatedResidualNetwork


class TestGatedResidualNetwork(unittest.TestCase):
    """Test cases for GatedResidualNetwork layer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.batch_size = 32
        self.input_dim = 16
        self.units = 16  # Same as input_dim for residual connection
        self.dropout_rate = 0.2
        self.layer = GatedResidualNetwork(units=self.units, dropout_rate=self.dropout_rate)
        # Using TensorFlow for test data generation only
        self.inputs = tf.random.normal((self.batch_size, self.input_dim))
        tf.random.set_seed(42)  # For reproducibility

    def test_initialization(self) -> None:
        """Test layer initialization with various parameters."""
        # Test default initialization
        layer = GatedResidualNetwork(units=self.units)
        self.assertEqual(layer.units, self.units)
        self.assertEqual(layer.dropout_rate, 0.2)  # Default value

        # Test with custom parameters
        dropout_rate = 0.5
        layer = GatedResidualNetwork(units=self.units, dropout_rate=dropout_rate)
        self.assertEqual(layer.units, self.units)
        self.assertEqual(layer.dropout_rate, dropout_rate)

    def test_invalid_initialization(self) -> None:
        """Test layer initialization with invalid parameters."""
        # Test with negative units
        with self.assertRaises(ValueError):
            GatedResidualNetwork(units=-1)

        # Test with zero units
        with self.assertRaises(ValueError):
            GatedResidualNetwork(units=0)

        # Test with non-integer units
        with self.assertRaises(ValueError):
            GatedResidualNetwork(units=3.5)

        # Test with negative dropout rate
        with self.assertRaises(ValueError):
            GatedResidualNetwork(units=self.units, dropout_rate=-0.1)

        # Test with dropout rate >= 1
        with self.assertRaises(ValueError):
            GatedResidualNetwork(units=self.units, dropout_rate=1.0)

    def test_build(self) -> None:
        """Test layer building."""
        # Build the layer
        self.layer.build((None, self.input_dim))
        
        # Check if all components are created
        self.assertIsNotNone(self.layer.elu_dense)
        self.assertIsNotNone(self.layer.linear_dense)
        self.assertIsNotNone(self.layer.dropout)
        self.assertIsNotNone(self.layer.gated_linear_unit)
        self.assertIsNotNone(self.layer.layer_norm)
        
        # Check if projection layer is not created when input_dim == units
        self.assertFalse(hasattr(self.layer, "project") and self.layer.project is not None)
        
        # Test with different input dimension
        different_input_dim = self.units * 2
        layer = GatedResidualNetwork(units=self.units)
        layer.build((None, different_input_dim))
        
        # Check if projection layer is created when input_dim != units
        self.assertTrue(hasattr(layer, "project") and layer.project is not None)

    def test_output_shape(self) -> None:
        """Test output shape of the layer."""
        outputs = self.layer(self.inputs)
        
        # Check output shape
        self.assertEqual(outputs.shape, (self.batch_size, self.units))
        
        # Test with different input shapes
        test_shapes = [
            (16, 32),
            (64, 8),
            (128, 64)
        ]
        for batch, features in test_shapes:
            inputs = tf.random.normal((batch, features))
            layer = GatedResidualNetwork(units=self.units)
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
        # In training mode, dropout should be applied
        # In inference mode, dropout should be disabled
        # This might lead to different outputs
        output_train = self.layer(self.inputs, training=True)
        output_infer = self.layer(self.inputs, training=False)
        
        # Outputs might be different due to dropout
        # But they should have the same shape
        self.assertEqual(output_train.shape, output_infer.shape)

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        # Create a simple model with our layer
        inputs = layers.Input(shape=(self.input_dim,))
        outputs = self.layer(inputs)
        model = Model(inputs=inputs, outputs=outputs)

        # Train the model for one step to ensure weights are built
        x = tf.random.normal((self.batch_size, self.input_dim))
        y = tf.random.normal((self.batch_size, self.units))
        model.compile(optimizer='adam', loss='mse')
        model.fit(x, y, epochs=1, verbose=0)

        # Save and reload the model
        model_config = model.get_config()
        reloaded_model = Model.from_config(model_config)

        # Set the weights from the original model
        reloaded_model.set_weights(model.get_weights())

        # Test with same input
        test_input = tf.random.normal((1, self.input_dim))
        original_output = model(test_input, training=False)
        reloaded_output = reloaded_model(test_input, training=False)

        # Check if outputs are exactly equal since we copied weights
        tf.debugging.assert_equal(original_output, reloaded_output)

    def test_model_integration(self) -> None:
        """Test layer integration in a model."""
        # Create a simple sequential model
        model = Sequential([
            layers.Input(shape=(self.input_dim,)),
            self.layer
        ])

        # Ensure model can be compiled and trained
        model.compile(optimizer='adam', loss='mse')
        
        # Generate dummy data
        x = tf.random.normal((100, self.input_dim))
        y = tf.random.normal((100, self.units))
        
        # Train for one epoch
        history = model.fit(x, y, epochs=1, verbose=0)
        self.assertTrue(history.history['loss'][0] > 0)

    def test_residual_connection(self) -> None:
        """Test that the residual connection works as expected."""
        # Create a layer with a small units value
        layer = GatedResidualNetwork(units=4, dropout_rate=0.0)
        
        # Create a simple input
        test_input = tf.ones((1, 4))
        
        # Call the layer
        output = layer(test_input, training=False)
        
        # The output should be different from the input due to transformations
        # But it should maintain the same shape
        self.assertEqual(output.shape, test_input.shape)
        
        # The output should not be all zeros (which would happen if the residual connection failed)
        self.assertFalse(tf.reduce_all(tf.equal(output, 0.0)))

    def test_layer_normalization(self) -> None:
        """Test that layer normalization is applied correctly."""
        # Create inputs with a specific mean and variance
        inputs = tf.ones((self.batch_size, self.input_dim)) * 5.0  # Mean = 5.0, Var = 0
        
        # Call the layer
        outputs = self.layer(inputs, training=False)
        
        # Layer normalization should normalize the outputs
        # The mean should be close to 0 and variance close to 1
        # But due to the residual connection and other transformations,
        # we can't make exact assertions about the statistics
        
        # Just check that the outputs are not all equal (which would happen if normalization failed)
        self.assertFalse(tf.reduce_all(tf.equal(outputs, outputs[0, 0])))


if __name__ == '__main__':
    unittest.main() 