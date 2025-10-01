"""Unit tests for the TabularMoELayer layer.

Note: TensorFlow is used in tests for validation purposes only.
The actual layer implementation uses only Keras 3 operations.
"""

import unittest
import numpy as np
import tensorflow as tf  # Used for testing only
from keras import layers, Model
from kmr.layers.TabularMoELayer import TabularMoELayer

class TestTabularMoELayer(unittest.TestCase):
    """Test cases for the TabularMoELayer layer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.batch_size = 32
        self.feature_dim = 16
        self.num_experts = 4
        self.expert_units = 8
        # Using TensorFlow for test data generation only
        self.test_input = tf.random.normal((self.batch_size, self.feature_dim))
        tf.random.set_seed(42)  # For reproducibility

    def test_initialization(self) -> None:
        """Test layer initialization with various parameters."""
        # Test default initialization
        layer = TabularMoELayer()
        self.assertEqual(layer.num_experts, 4)
        self.assertEqual(layer.expert_units, 16)

        # Test custom initialization
        layer = TabularMoELayer(
            num_experts=6,
            expert_units=32
        )
        self.assertEqual(layer.num_experts, 6)
        self.assertEqual(layer.expert_units, 32)

    def test_invalid_initialization(self) -> None:
        """Test layer initialization with invalid parameters."""
        # Test invalid num_experts
        with self.assertRaises(ValueError):
            TabularMoELayer(num_experts=0)
        with self.assertRaises(ValueError):
            TabularMoELayer(num_experts=-1)

        # Test invalid expert_units
        with self.assertRaises(ValueError):
            TabularMoELayer(expert_units=0)
        with self.assertRaises(ValueError):
            TabularMoELayer(expert_units=-1)

    def test_build(self) -> None:
        """Test layer building with different configurations."""
        # Test with default parameters
        layer = TabularMoELayer(
            num_experts=self.num_experts,
            expert_units=self.expert_units
        )
        layer.build(input_shape=(None, self.feature_dim))
        
        # Check if experts are created
        self.assertEqual(len(layer.experts), self.num_experts)
        self.assertEqual(len(layer.expert_outputs), self.num_experts)
        
        # Check if gate is created
        self.assertIsNotNone(layer.gate)
        
        # Check expert dimensions
        for expert in layer.experts:
            self.assertEqual(expert.units, self.expert_units)
        
        # Check expert output dimensions
        for expert_output in layer.expert_outputs:
            self.assertEqual(expert_output.units, self.feature_dim)

    def test_output_shape(self) -> None:
        """Test output shape preservation."""
        # Test with default input
        layer = TabularMoELayer(
            num_experts=self.num_experts,
            expert_units=self.expert_units
        )
        output = layer(self.test_input)
        self.assertEqual(output.shape, self.test_input.shape)

        # Test with different input shapes
        test_shapes = [
            (16, 8),
            (64, 32),
            (128, 64)
        ]
        for shape in test_shapes:
            # Create new layer instance for each shape
            layer = TabularMoELayer(
                num_experts=3,
                expert_units=shape[1] // 2
            )
            test_input = tf.random.normal((shape[0], shape[1]))
            output = layer(test_input)
            self.assertEqual(output.shape, test_input.shape)

    def test_expert_specialization(self) -> None:
        """Test that experts can specialize on different input patterns."""
        # Create a layer with a small number of experts for easier testing
        layer = TabularMoELayer(num_experts=2, expert_units=4)
        
        # Build the layer
        layer.build(input_shape=(None, self.feature_dim))
        
        # Create two distinct input patterns
        pattern1 = tf.ones((1, self.feature_dim))
        pattern2 = tf.concat([tf.ones((1, self.feature_dim // 2)), 
                             tf.zeros((1, self.feature_dim // 2))], axis=1)
        
        # Get outputs for both patterns
        output1 = layer(pattern1)
        output2 = layer(pattern2)
        
        # The outputs should be different due to the gating mechanism
        self.assertFalse(tf.reduce_all(tf.equal(output1, output2)))

    def test_gating_mechanism(self) -> None:
        """Test the gating mechanism of the layer."""
        layer = TabularMoELayer(num_experts=2, expert_units=4)
        
        # Call the layer once to build it
        _ = layer(self.test_input)
        
        # Get the original gate weights
        original_weights = layer.gate.get_weights()
        
        # Modify gate weights to strongly favor the first expert
        modified_weights = [np.zeros_like(original_weights[0]), np.zeros_like(original_weights[1])]
        # Set bias to strongly favor first expert
        modified_weights[1][0] = 10.0
        modified_weights[1][1] = -10.0
        layer.gate.set_weights(modified_weights)
        
        # Create a test input
        test_input = tf.ones((1, self.feature_dim))
        
        # Get output with modified gate
        output1 = layer(test_input)
        
        # Now modify gate weights to strongly favor the second expert
        modified_weights[1][0] = -10.0
        modified_weights[1][1] = 10.0
        layer.gate.set_weights(modified_weights)
        
        # Get output with new gate weights
        output2 = layer(test_input)
        
        # The outputs should be different due to the different gating
        self.assertFalse(tf.reduce_all(tf.equal(output1, output2)))

    def test_training_mode(self) -> None:
        """Test layer behavior in training and inference modes."""
        layer = TabularMoELayer(
            num_experts=self.num_experts,
            expert_units=self.expert_units
        )
        
        # In this layer, training mode doesn't affect the output directly
        # But we test it for completeness
        output_train = layer(self.test_input, training=True)
        output_infer = layer(self.test_input, training=False)
        
        # Shapes should be the same
        self.assertEqual(output_train.shape, output_infer.shape)

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        original_layer = TabularMoELayer(
            num_experts=6,
            expert_units=32
        )
        config = original_layer.get_config()

        # Create new layer from config
        restored_layer = TabularMoELayer.from_config(config)

        # Check if configurations match
        self.assertEqual(restored_layer.num_experts, original_layer.num_experts)
        self.assertEqual(restored_layer.expert_units, original_layer.expert_units)

    def test_integration(self) -> None:
        """Test integration with a simple model."""
        # Create a simple model with the MoE layer
        inputs = layers.Input(shape=(self.feature_dim,))
        x = TabularMoELayer(
            num_experts=self.num_experts,
            expert_units=self.expert_units
        )(inputs)
        outputs = layers.Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile the model
        model.compile(optimizer='adam', loss='mse')
        
        # Generate some dummy data
        x_data = tf.random.normal((100, self.feature_dim))
        y_data = tf.random.normal((100, 1))
        
        # Train for one step to ensure everything works
        history = model.fit(x_data, y_data, epochs=1, verbose=0)
        
        # Check that loss was computed
        self.assertIsNotNone(history.history['loss'])

if __name__ == "__main__":
    unittest.main() 