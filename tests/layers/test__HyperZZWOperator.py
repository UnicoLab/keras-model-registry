"""Unit tests for the HyperZZWOperator layer.

Note: TensorFlow is used in tests for validation purposes only.
The actual layer implementation uses only Keras 3 operations.
"""

import unittest
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from keras import layers, Model, ops, utils
from keras import random
from kmr.layers.HyperZZWOperator import HyperZZWOperator

class TestHyperZZWOperator(unittest.TestCase):
    """Test cases for the HyperZZWOperator layer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.batch_size = 32
        self.input_dim = 16
        self.context_dim = 8
        # Using Keras utils for random seed
        utils.set_random_seed(42)  # For reproducibility
        self.test_input = random.normal((self.batch_size, self.input_dim))
        self.test_context = random.normal((self.batch_size, self.context_dim))

    def test_initialization(self) -> None:
        """Test layer initialization with various parameters."""
        # Test default initialization
        layer = HyperZZWOperator(input_dim=self.input_dim)
        self.assertEqual(layer.input_dim, self.input_dim)
        self.assertIsNone(layer.context_dim)

        # Test custom initialization
        layer = HyperZZWOperator(input_dim=8, context_dim=4)
        self.assertEqual(layer.input_dim, 8)
        self.assertEqual(layer.context_dim, 4)

    def test_invalid_initialization(self) -> None:
        """Test layer initialization with invalid parameters."""
        # Test invalid input_dim
        with self.assertRaises(ValueError):
            HyperZZWOperator(input_dim=0)
        with self.assertRaises(ValueError):
            HyperZZWOperator(input_dim=-1)
            
        # Test invalid context_dim
        with self.assertRaises(ValueError):
            HyperZZWOperator(input_dim=self.input_dim, context_dim=0)
        with self.assertRaises(ValueError):
            HyperZZWOperator(input_dim=self.input_dim, context_dim=-1)

    def test_build(self) -> None:
        """Test layer building with different configurations."""
        # Test with default parameters
        layer = HyperZZWOperator(input_dim=self.input_dim)
        layer.build(input_shape=[(None, self.input_dim), (None, self.context_dim)])
        
        # Check if hyper_kernel is created
        self.assertIsNotNone(layer.hyper_kernel)
        
        # Check hyper_kernel dimensions
        self.assertEqual(layer.hyper_kernel.shape, (self.context_dim, self.input_dim))
        
        # Test with explicit context_dim
        layer = HyperZZWOperator(input_dim=self.input_dim, context_dim=self.context_dim)
        layer.build(input_shape=[(None, self.input_dim), (None, self.context_dim)])
        
        # Check hyper_kernel dimensions
        self.assertEqual(layer.hyper_kernel.shape, (self.context_dim, self.input_dim))

    def test_output_shape(self) -> None:
        """Test output shape preservation."""
        # Test with default input
        layer = HyperZZWOperator(input_dim=self.input_dim)
        output = layer([self.test_input, self.test_context])
        
        # Output should be batch_size x input_dim
        self.assertEqual(output.shape, (self.batch_size, self.input_dim))

        # Test with different input shapes
        test_shapes = [
            (16, 8, 4),  # batch_size, input_dim, context_dim
            (64, 32, 16),
            (128, 64, 32)
        ]
        for shape in test_shapes:
            # Create new layer instance for each shape
            layer = HyperZZWOperator(input_dim=shape[1], context_dim=shape[2])
            test_input = random.normal((shape[0], shape[1]))
            test_context = random.normal((shape[0], shape[2]))
            output = layer([test_input, test_context])
            self.assertEqual(output.shape, (shape[0], shape[1]))

    def test_forward_pass(self) -> None:
        """Test the forward pass of the layer."""
        layer = HyperZZWOperator(input_dim=self.input_dim, context_dim=self.context_dim)
        
        # Call the layer once to build it
        _ = layer([self.test_input, self.test_context])
        
        # Check that the output is different from the input
        # This is a basic test to ensure the layer is doing some transformation
        output = layer([self.test_input, self.test_context])
        self.assertFalse(ops.all(ops.equal(output, self.test_input)))

    def test_context_dependency(self) -> None:
        """Test that different contexts produce different outputs."""
        layer = HyperZZWOperator(input_dim=self.input_dim, context_dim=self.context_dim)
        
        # Generate two different contexts
        context1 = random.normal((self.batch_size, self.context_dim))
        context2 = random.normal((self.batch_size, self.context_dim))
        
        # Get outputs for the same input but different contexts
        output1 = layer([self.test_input, context1])
        output2 = layer([self.test_input, context2])
        
        # Outputs should be different
        self.assertFalse(ops.all(ops.equal(output1, output2)))

    def test_training_mode(self) -> None:
        """Test layer behavior in training and inference modes."""
        layer = HyperZZWOperator(input_dim=self.input_dim, context_dim=self.context_dim)
        
        # For this layer, training mode doesn't affect the output
        # But we test it for completeness
        output_train = layer([self.test_input, self.test_context], training=True)
        output_infer = layer([self.test_input, self.test_context], training=False)
        
        # Shapes should be the same
        self.assertEqual(output_train.shape, output_infer.shape)
        
        # Outputs should be the same since training doesn't affect this layer
        self.assertTrue(ops.all(ops.equal(output_train, output_infer)))

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        original_layer = HyperZZWOperator(input_dim=self.input_dim, context_dim=self.context_dim)
        config = original_layer.get_config()

        # Create new layer from config
        restored_layer = HyperZZWOperator.from_config(config)

        # Check if configurations match
        self.assertEqual(restored_layer.input_dim, original_layer.input_dim)
        self.assertEqual(restored_layer.context_dim, original_layer.context_dim)

    def test_integration(self) -> None:
        """Test integration with a simple model."""
        # Create a simple model with the HyperZZWOperator layer
        input_tensor = layers.Input(shape=(self.input_dim,))
        context_tensor = layers.Input(shape=(self.context_dim,))
        x = HyperZZWOperator(input_dim=self.input_dim, context_dim=self.context_dim)([input_tensor, context_tensor])
        outputs = layers.Dense(1)(x)
        model = Model(inputs=[input_tensor, context_tensor], outputs=outputs)
        
        # Compile the model
        model.compile(optimizer='adam', loss='mse')
        
        # Generate some dummy data
        x_data = random.normal((100, self.input_dim))
        c_data = random.normal((100, self.context_dim))
        y_data = random.normal((100, 1))
        
        # Train for one step to ensure everything works
        history = model.fit([x_data, c_data], y_data, epochs=1, verbose=0)
        
        # Check that loss was computed
        self.assertIsNotNone(history.history['loss'])

    def test_learnable_weights(self) -> None:
        """Test that the layer's weights are learnable."""
        layer = HyperZZWOperator(input_dim=self.input_dim, context_dim=self.context_dim)
        
        # Call the layer once to build it
        _ = layer([self.test_input, self.test_context])
        
        # Get initial weights
        initial_weights = layer.get_weights()[0].copy()
        
        # Create a simple model with the layer
        input_tensor = layers.Input(shape=(self.input_dim,))
        context_tensor = layers.Input(shape=(self.context_dim,))
        x = HyperZZWOperator(input_dim=self.input_dim, context_dim=self.context_dim)([input_tensor, context_tensor])
        outputs = layers.Dense(1)(x)
        model = Model(inputs=[input_tensor, context_tensor], outputs=outputs)
        
        # Compile the model
        model.compile(optimizer='adam', loss='mse')
        
        # Generate some dummy data
        x_data = random.normal((100, self.input_dim))
        c_data = random.normal((100, self.context_dim))
        y_data = random.normal((100, 1))
        
        # Train for a few steps
        model.fit([x_data, c_data], y_data, epochs=5, verbose=0)
        
        # Get updated weights
        updated_weights = model.layers[2].get_weights()[0]  # Index 2 should be the HyperZZWOperator
        
        # Weights should have changed
        self.assertFalse(np.array_equal(initial_weights, updated_weights))

if __name__ == "__main__":
    unittest.main() 