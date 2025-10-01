"""Unit tests for the TerminatorModel.

Note: TensorFlow is used in tests for validation purposes only.
The actual model implementation uses only Keras 3 operations.
"""

import unittest
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from keras import Model, Sequential, layers, ops
from keras import utils, random
from kmr.models.TerminatorModel import TerminatorModel

class TestTerminatorModel(unittest.TestCase):
    """Test cases for the TerminatorModel."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.batch_size = 32
        self.input_dim = 16
        self.context_dim = 8
        self.output_dim = 4
        self.hidden_dim = 32
        # Using Keras utils for random seed
        utils.set_random_seed(42)  # For reproducibility
        self.test_input = random.normal((self.batch_size, self.input_dim))
        self.test_context = random.normal((self.batch_size, self.context_dim))

    def test_initialization(self) -> None:
        """Test model initialization with various parameters."""
        # Test default initialization
        model = TerminatorModel(
            input_dim=self.input_dim,
            context_dim=self.context_dim,
            output_dim=self.output_dim
        )
        self.assertEqual(model.input_dim, self.input_dim)
        self.assertEqual(model.context_dim, self.context_dim)
        self.assertEqual(model.output_dim, self.output_dim)
        self.assertEqual(model.hidden_dim, 64)  # Default value
        self.assertEqual(model.num_layers, 2)  # Default value

        # Test custom initialization
        model = TerminatorModel(
            input_dim=8,
            context_dim=4,
            output_dim=2,
            hidden_dim=16,
            num_layers=3
        )
        self.assertEqual(model.input_dim, 8)
        self.assertEqual(model.context_dim, 4)
        self.assertEqual(model.output_dim, 2)
        self.assertEqual(model.hidden_dim, 16)
        self.assertEqual(model.num_layers, 3)

    def test_invalid_initialization(self) -> None:
        """Test model initialization with invalid parameters."""
        # Test invalid input_dim
        with self.assertRaises(ValueError):
            TerminatorModel(
                input_dim=0,
                context_dim=self.context_dim,
                output_dim=self.output_dim
            )
        with self.assertRaises(ValueError):
            TerminatorModel(
                input_dim=-1,
                context_dim=self.context_dim,
                output_dim=self.output_dim
            )

        # Test invalid context_dim
        with self.assertRaises(ValueError):
            TerminatorModel(
                input_dim=self.input_dim,
                context_dim=0,
                output_dim=self.output_dim
            )
        with self.assertRaises(ValueError):
            TerminatorModel(
                input_dim=self.input_dim,
                context_dim=-1,
                output_dim=self.output_dim
            )

        # Test invalid output_dim
        with self.assertRaises(ValueError):
            TerminatorModel(
                input_dim=self.input_dim,
                context_dim=self.context_dim,
                output_dim=0
            )
        with self.assertRaises(ValueError):
            TerminatorModel(
                input_dim=self.input_dim,
                context_dim=self.context_dim,
                output_dim=-1
            )

        # Test invalid hidden_dim
        with self.assertRaises(ValueError):
            TerminatorModel(
                input_dim=self.input_dim,
                context_dim=self.context_dim,
                output_dim=self.output_dim,
                hidden_dim=0
            )
        with self.assertRaises(ValueError):
            TerminatorModel(
                input_dim=self.input_dim,
                context_dim=self.context_dim,
                output_dim=self.output_dim,
                hidden_dim=-1
            )

        # Test invalid num_layers
        with self.assertRaises(ValueError):
            TerminatorModel(
                input_dim=self.input_dim,
                context_dim=self.context_dim,
                output_dim=self.output_dim,
                num_layers=0
            )
        with self.assertRaises(ValueError):
            TerminatorModel(
                input_dim=self.input_dim,
                context_dim=self.context_dim,
                output_dim=self.output_dim,
                num_layers=-1
            )

    def test_build(self) -> None:
        """Test model building with different configurations."""
        # Test with default parameters
        model = TerminatorModel(
            input_dim=self.input_dim,
            context_dim=self.context_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2
        )
        
        # Call the model once to build it
        _ = model([self.test_input, self.test_context])
        
        # Check if components are created
        self.assertIsNotNone(model.hyper_zzw)
        self.assertIsNotNone(model.slow_network)
        self.assertIsNotNone(model.output_layer)

    def test_output_shape(self) -> None:
        """Test output shape preservation."""
        # Test with default input
        model = TerminatorModel(
            input_dim=self.input_dim,
            context_dim=self.context_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2
        )
        output = model([self.test_input, self.test_context])
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

        # Test with different input shapes
        test_shapes = [
            (16, 8, 4, 2),  # batch_size, input_dim, context_dim, output_dim
            (64, 32, 16, 8),
            (128, 64, 32, 16)
        ]
        for shape in test_shapes:
            # Create new model instance for each shape
            model = TerminatorModel(
                input_dim=shape[1],
                context_dim=shape[2],
                output_dim=shape[3],
                hidden_dim=shape[1] * 2,
                num_layers=2
            )
            test_input = random.normal((shape[0], shape[1]))
            test_context = random.normal((shape[0], shape[2]))
            output = model([test_input, test_context])
            self.assertEqual(output.shape, (shape[0], shape[3]))

    def test_forward_pass(self) -> None:
        """Test the forward pass of the model."""
        model = TerminatorModel(
            input_dim=self.input_dim,
            context_dim=self.context_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2
        )
        
        # Call the model once to build it
        _ = model([self.test_input, self.test_context])
        
        # Check that the output has the correct shape
        output = model([self.test_input, self.test_context])
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_context_dependency(self) -> None:
        """Test that different contexts produce different outputs."""
        model = TerminatorModel(
            input_dim=self.input_dim,
            context_dim=self.context_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2
        )
        
        # Generate two different contexts
        context1 = random.normal((self.batch_size, self.context_dim))
        context2 = random.normal((self.batch_size, self.context_dim))
        
        # Get outputs for the same input but different contexts
        output1 = model([self.test_input, context1])
        output2 = model([self.test_input, context2])
        
        # Outputs should be different
        self.assertFalse(ops.all(ops.equal(output1, output2)))

    def test_training_mode(self) -> None:
        """Test model behavior in training and inference modes."""
        model = TerminatorModel(
            input_dim=self.input_dim,
            context_dim=self.context_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2
        )
        
        # For this model, training mode might affect the output due to dropout
        output_train = model([self.test_input, self.test_context], training=True)
        output_infer = model([self.test_input, self.test_context], training=False)
        
        # Shapes should be the same
        self.assertEqual(output_train.shape, output_infer.shape)

    def test_serialization(self) -> None:
        """Test model serialization and deserialization."""
        original_model = TerminatorModel(
            input_dim=self.input_dim,
            context_dim=self.context_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2
        )
        config = original_model.get_config()

        # Create new model from config
        restored_model = TerminatorModel.from_config(config)

        # Check if configurations match
        self.assertEqual(restored_model.input_dim, original_model.input_dim)
        self.assertEqual(restored_model.context_dim, original_model.context_dim)
        self.assertEqual(restored_model.output_dim, original_model.output_dim)
        self.assertEqual(restored_model.hidden_dim, original_model.hidden_dim)
        self.assertEqual(restored_model.num_layers, original_model.num_layers)

    def test_integration(self) -> None:
        """Test integration with a simple model."""
        # Create input layers
        input_tensor = layers.Input(shape=(self.input_dim,))
        context_tensor = layers.Input(shape=(self.context_dim,))
        
        # Create the model
        terminator = TerminatorModel(
            input_dim=self.input_dim,
            context_dim=self.context_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2
        )
        
        # Apply the model to the inputs
        outputs = terminator([input_tensor, context_tensor])
        
        # Create a Keras model
        model = Model(inputs=[input_tensor, context_tensor], outputs=outputs)
        
        # Compile the model
        model.compile(optimizer='adam', loss='mse')
        
        # Generate some dummy data
        x_data = random.normal((100, self.input_dim))
        c_data = random.normal((100, self.context_dim))
        y_data = random.normal((100, self.output_dim))
        
        # Train for one step to ensure everything works
        history = model.fit([x_data, c_data], y_data, epochs=1, verbose=0)
        
        # Check that loss was computed
        self.assertIsNotNone(history.history['loss'])

    def test_learnable_weights(self) -> None:
        """Test that the model's weights are learnable."""
        # Create a model instance
        model = TerminatorModel(
            input_dim=self.input_dim,
            context_dim=self.context_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2
        )
        
        # Call the model once to build it
        _ = model([self.test_input, self.test_context])
        
        # Get initial weights (just check one component)
        initial_weights = model.hyper_zzw.get_weights()[0].copy()
        
        # Create a Keras model
        input_tensor = layers.Input(shape=(self.input_dim,))
        context_tensor = layers.Input(shape=(self.context_dim,))
        outputs = model([input_tensor, context_tensor])
        keras_model = Model(inputs=[input_tensor, context_tensor], outputs=outputs)
        
        # Compile the model
        keras_model.compile(optimizer='adam', loss='mse')
        
        # Generate some dummy data
        x_data = random.normal((100, self.input_dim))
        c_data = random.normal((100, self.context_dim))
        y_data = random.normal((100, self.output_dim))
        
        # Train for a few steps
        keras_model.fit([x_data, c_data], y_data, epochs=5, verbose=0)
        
        # Get updated weights
        updated_weights = model.hyper_zzw.get_weights()[0]
        
        # Weights should have changed
        self.assertFalse(np.array_equal(initial_weights, updated_weights))

    def test_component_interaction(self) -> None:
        """Test that the components of the model interact correctly."""
        # Create a model with minimal configuration for testing
        model = TerminatorModel(
            input_dim=self.input_dim,
            context_dim=self.context_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
            num_blocks=1  # Just one SFNE block for simplicity
        )
        
        # Call the model once to build it
        _ = model([self.test_input, self.test_context])
        
        # Test the slow network
        slow_network_output = model.slow_network(self.test_context)
        self.assertEqual(slow_network_output.shape, (self.batch_size, self.context_dim))
        
        # Test the hyper_zzw operator
        input_layer_output = model.input_layer(self.test_input)
        hyper_zzw_output = model.hyper_zzw([input_layer_output, slow_network_output])
        self.assertEqual(hyper_zzw_output.shape, (self.batch_size, self.input_dim))
        
        # Test the SFNE block
        sfne_output = model.sfne_blocks[0](input_layer_output)
        self.assertEqual(sfne_output.shape, (self.batch_size, self.input_dim))
        
        # Test the output layer
        final_output = model.output_layer(sfne_output * hyper_zzw_output)
        self.assertEqual(final_output.shape, (self.batch_size, self.output_dim))

if __name__ == "__main__":
    unittest.main() 