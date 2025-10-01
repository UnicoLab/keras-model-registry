"""
Unit tests for the GraphFeatureAggregation layer.
"""

import unittest
import numpy as np
import tensorflow as tf
from keras import Model, layers, ops
from kmr.layers import GraphFeatureAggregation


class TestGraphFeatureAggregation(unittest.TestCase):
    """Test cases for the GraphFeatureAggregation layer."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 16
        self.num_features = 10
        self.embed_dim = 8
        self.dropout_rate = 0.1
        self.leaky_relu_alpha = 0.2
        
        # Create random input data
        self.inputs = tf.random.normal((self.batch_size, self.num_features))
        
        # Create the layer
        self.layer = GraphFeatureAggregation(
            embed_dim=self.embed_dim,
            dropout_rate=self.dropout_rate,
            leaky_relu_alpha=self.leaky_relu_alpha
        )

    def test_initialization(self):
        """Test that the layer initializes with the correct parameters."""
        layer = GraphFeatureAggregation(
            embed_dim=16,
            dropout_rate=0.2,
            leaky_relu_alpha=0.3
        )
        
        self.assertEqual(layer.embed_dim, 16)
        self.assertEqual(layer.dropout_rate, 0.2)
        self.assertEqual(layer.leaky_relu_alpha, 0.3)
        
        # Test default values
        default_layer = GraphFeatureAggregation()
        self.assertEqual(default_layer.embed_dim, 8)
        self.assertEqual(default_layer.dropout_rate, 0.0)
        self.assertEqual(default_layer.leaky_relu_alpha, 0.2)

    def test_invalid_params(self):
        """Test that the layer raises appropriate errors for invalid parameters."""
        # Test invalid embed_dim
        with self.assertRaises(ValueError):
            GraphFeatureAggregation(embed_dim=0)
        
        with self.assertRaises(ValueError):
            GraphFeatureAggregation(embed_dim=-5)
        
        # Test invalid dropout_rate
        with self.assertRaises(ValueError):
            GraphFeatureAggregation(dropout_rate=-0.1)
        
        with self.assertRaises(ValueError):
            GraphFeatureAggregation(dropout_rate=1.0)
        
        # Test invalid leaky_relu_alpha
        with self.assertRaises(ValueError):
            GraphFeatureAggregation(leaky_relu_alpha=0)
        
        with self.assertRaises(ValueError):
            GraphFeatureAggregation(leaky_relu_alpha=-0.1)

    def test_build(self):
        """Test that the layer builds correctly with the given input shape."""
        # Build the layer
        self.layer.build((None, self.num_features))
        
        # Check that the layer attributes are set correctly
        self.assertEqual(self.layer.num_features, self.num_features)
        self.assertIsNotNone(self.layer.projection)
        self.assertIsNotNone(self.layer.attention_a)
        self.assertIsNotNone(self.layer.attention_bias)
        self.assertIsNotNone(self.layer.leaky_relu)
        self.assertIsNotNone(self.layer.out_proj)
        
        # Check the shape of the attention vector
        self.assertEqual(self.layer.attention_a.shape, (2 * self.embed_dim, 1))
        
        # Check that dropout layer is created when dropout_rate > 0
        self.assertIsNotNone(self.layer.dropout_layer)
        
        # Check that dropout layer is None when dropout_rate = 0
        layer_no_dropout = GraphFeatureAggregation(dropout_rate=0.0)
        layer_no_dropout.build((None, self.num_features))
        self.assertIsNone(layer_no_dropout.dropout_layer)

    def test_output_shape(self):
        """Test that the layer preserves the input shape."""
        # Call the layer
        outputs = self.layer(self.inputs)
        
        # Check that the output shape matches the input shape
        self.assertEqual(outputs.shape, self.inputs.shape)
        
        # Test with different batch sizes and feature dimensions
        for batch_size in [1, 8, 32]:
            for num_features in [5, 15, 30]:
                inputs = tf.random.normal((batch_size, num_features))
                layer = GraphFeatureAggregation()
                outputs = layer(inputs)
                self.assertEqual(outputs.shape, inputs.shape)

    def test_attention_mechanism(self):
        """Test the attention mechanism of the layer."""
        # Create a new layer for this test
        layer = GraphFeatureAggregation(
            embed_dim=self.embed_dim,
            dropout_rate=self.dropout_rate,
            leaky_relu_alpha=self.leaky_relu_alpha
        )
        
        # Build the layer
        layer.build((None, self.num_features))
        
        # Set non-zero bias to ensure non-zero outputs with zero inputs
        layer.attention_bias.assign([1.0])
        
        # Call the layer with non-zero inputs
        outputs = layer(self.inputs, training=True)
        
        # Check that the outputs are not the same as the inputs
        self.assertFalse(np.allclose(outputs.numpy(), self.inputs.numpy(), atol=1e-5))
        
        # Test that the attention mechanism is working by checking that
        # different input features produce different outputs
        # Create inputs with different patterns
        pattern1 = tf.ones((1, self.num_features))
        pattern2 = tf.concat([tf.ones((1, self.num_features // 2)), 
                             tf.zeros((1, self.num_features // 2))], axis=1)
        
        # Get outputs for different patterns
        output1 = layer(pattern1)
        output2 = layer(pattern2)
        
        # The outputs should be different due to the attention mechanism
        self.assertFalse(np.allclose(output1.numpy(), output2.numpy(), atol=1e-5))

    def test_training_mode(self):
        """Test that the layer behaves differently in training and inference modes."""
        # Only test if dropout_rate > 0
        if self.layer.dropout_rate > 0:
            # Set a seed for reproducibility
            tf.random.set_seed(42)
            
            # Call the layer in training mode
            outputs_training = self.layer(self.inputs, training=True)
            
            # Reset the seed
            tf.random.set_seed(42)
            
            # Call the layer in inference mode
            outputs_inference = self.layer(self.inputs, training=False)
            
            # Check that the outputs are different due to dropout
            self.assertFalse(np.allclose(outputs_training.numpy(), outputs_inference.numpy(), atol=1e-5))

    def test_serialization(self):
        """Test that the layer can be serialized and deserialized."""
        # Call the layer to build it
        _ = self.layer(self.inputs)
        
        # Get the config
        config = self.layer.get_config()
        
        # Check that the config contains the correct values
        self.assertEqual(config["embed_dim"], self.embed_dim)
        self.assertEqual(config["dropout_rate"], self.dropout_rate)
        self.assertEqual(config["leaky_relu_alpha"], self.leaky_relu_alpha)
        
        # Create a new layer from the config
        new_layer = GraphFeatureAggregation.from_config(config)
        
        # Check that the new layer has the same parameters
        self.assertEqual(new_layer.embed_dim, self.layer.embed_dim)
        self.assertEqual(new_layer.dropout_rate, self.layer.dropout_rate)
        self.assertEqual(new_layer.leaky_relu_alpha, self.layer.leaky_relu_alpha)

    def test_integration(self):
        """Test the layer in a simple model."""
        # Create a simple model with the GraphFeatureAggregation layer
        inputs = layers.Input(shape=(self.num_features,))
        x = GraphFeatureAggregation(
            embed_dim=self.embed_dim,
            dropout_rate=self.dropout_rate,
            leaky_relu_alpha=self.leaky_relu_alpha
        )(inputs)
        x = layers.Dense(32, activation="relu")(x)
        outputs = layers.Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile the model
        model.compile(optimizer="adam", loss="mse")
        
        # Create some dummy data
        x_train = tf.random.normal((100, self.num_features))
        y_train = tf.random.normal((100, 1))
        
        # Train the model for a few steps
        history = model.fit(x_train, y_train, epochs=2, batch_size=32, verbose=0)
        
        # Check that the loss decreased
        self.assertLess(history.history["loss"][1], history.history["loss"][0])
        
        # Make predictions
        test_inputs = tf.random.normal((10, self.num_features))
        predictions = model.predict(test_inputs)
        
        # Check that the predictions have the expected shape
        self.assertEqual(predictions.shape, (10, 1))

    def test_residual_connection(self):
        """Test that the residual connection works correctly."""
        # Create a test to directly verify the residual connection
        # We'll create a custom input and check that the output includes it
        
        # Create a layer with a specific initialization
        layer = GraphFeatureAggregation(embed_dim=self.embed_dim)
        
        # Build the layer
        layer.build((None, self.num_features))
        
        # Create a special input with a recognizable pattern
        test_input = tf.ones((1, self.num_features))
        
        # Call the layer
        output = layer(test_input)
        
        # Verify that the residual connection is working by checking
        # that the output is different from the input (due to the transformation)
        # but still contains the input (due to the residual connection)
        self.assertFalse(np.allclose(output.numpy(), test_input.numpy(), atol=1e-5))
        
        # The difference between output and input should be the transformation part
        transformation = output - test_input
        
        # Now we'll create a modified layer where we zero out the transformation
        # by setting the output projection weights to zero
        modified_layer = GraphFeatureAggregation(embed_dim=self.embed_dim)
        modified_layer.build((None, self.num_features))
        
        # Get the original output
        original_output = modified_layer(test_input)
        
        # Now modify the layer to have zero output projection
        if hasattr(modified_layer.out_proj, 'kernel'):
            modified_layer.out_proj.kernel.assign(tf.zeros_like(modified_layer.out_proj.kernel))
        if hasattr(modified_layer.out_proj, 'bias'):
            modified_layer.out_proj.bias.assign(tf.zeros_like(modified_layer.out_proj.bias))
        
        # Get the output with zero transformation
        zero_transform_output = modified_layer(test_input)
        
        # The output should now be equal to the input (due to the residual connection)
        # with a small tolerance for numerical precision
        self.assertTrue(np.allclose(zero_transform_output.numpy(), test_input.numpy(), atol=1e-5))


if __name__ == "__main__":
    unittest.main() 