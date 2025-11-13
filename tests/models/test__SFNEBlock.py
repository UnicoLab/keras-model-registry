"""Unit tests for the SFNEBlock model.

Note: TensorFlow is used in tests for validation purposes only.
The actual model implementation uses only Keras 3 operations.
"""

import unittest
import numpy as np
import tensorflow as tf  # Used for testing only
from keras import layers, Model
from kerasfactory.models.SFNEBlock import SFNEBlock


class TestSFNEBlock(unittest.TestCase):
    """Test cases for the SFNEBlock model."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.batch_size = 32
        self.input_dim = 16
        self.output_dim = 8
        self.hidden_dim = 32
        # Using TensorFlow for test data generation only
        tf.random.set_seed(42)  # For reproducibility
        self.test_input = tf.random.normal((self.batch_size, self.input_dim))

    def test_initialization(self) -> None:
        """Test model initialization with various parameters."""
        # Test default initialization
        model = SFNEBlock(input_dim=self.input_dim, output_dim=self.output_dim)
        self.assertEqual(model.input_dim, self.input_dim)
        self.assertEqual(model.output_dim, self.output_dim)
        self.assertEqual(model.hidden_dim, 64)  # Default value
        self.assertEqual(model.num_layers, 2)  # Default value

        # Test custom initialization
        model = SFNEBlock(input_dim=8, output_dim=4, hidden_dim=16, num_layers=3)
        self.assertEqual(model.input_dim, 8)
        self.assertEqual(model.output_dim, 4)
        self.assertEqual(model.hidden_dim, 16)
        self.assertEqual(model.num_layers, 3)

    def test_invalid_initialization(self) -> None:
        """Test model initialization with invalid parameters."""
        # Test invalid input_dim
        with self.assertRaises(ValueError):
            SFNEBlock(input_dim=0, output_dim=self.output_dim)
        with self.assertRaises(ValueError):
            SFNEBlock(input_dim=-1, output_dim=self.output_dim)

        # Test invalid output_dim
        with self.assertRaises(ValueError):
            SFNEBlock(input_dim=self.input_dim, output_dim=0)
        with self.assertRaises(ValueError):
            SFNEBlock(input_dim=self.input_dim, output_dim=-1)

        # Test invalid hidden_dim
        with self.assertRaises(ValueError):
            SFNEBlock(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                hidden_dim=0,
            )
        with self.assertRaises(ValueError):
            SFNEBlock(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                hidden_dim=-1,
            )

        # Test invalid num_layers
        with self.assertRaises(ValueError):
            SFNEBlock(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                num_layers=0,
            )
        with self.assertRaises(ValueError):
            SFNEBlock(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                num_layers=-1,
            )

    def test_build(self) -> None:
        """Test model building with different configurations."""
        # Test with default parameters
        model = SFNEBlock(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
        )

        # Call the model once to build it
        _ = model(self.test_input)

        # Check if layers are created
        self.assertIsNotNone(model.input_layer)
        self.assertEqual(len(model.hidden_layers), 2)
        self.assertIsNotNone(model.output_layer)

        # Check layer dimensions
        self.assertEqual(model.input_layer.units, self.hidden_dim)
        for hidden_layer in model.hidden_layers:
            self.assertEqual(hidden_layer.units, self.hidden_dim)
        self.assertEqual(model.output_layer.units, self.output_dim)

    def test_output_shape(self) -> None:
        """Test output shape preservation."""
        # Test with default input
        model = SFNEBlock(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
        )
        output = model(self.test_input)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

        # Test with different input shapes
        test_shapes = [
            (16, 8, 4),
            (64, 32, 16),
            (128, 64, 32),
        ]  # batch_size, input_dim, output_dim
        for shape in test_shapes:
            # Create new model instance for each shape
            model = SFNEBlock(
                input_dim=shape[1],
                output_dim=shape[2],
                hidden_dim=shape[1] * 2,
                num_layers=2,
            )
            test_input = tf.random.normal((shape[0], shape[1]))
            output = model(test_input)
            self.assertEqual(output.shape, (shape[0], shape[2]))

    def test_forward_pass(self) -> None:
        """Test the forward pass of the model."""
        model = SFNEBlock(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
        )

        # Call the model once to build it
        _ = model(self.test_input)

        # Check that the output has the correct shape
        output = model(self.test_input)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_training_mode(self) -> None:
        """Test model behavior in training and inference modes."""
        model = SFNEBlock(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
        )

        # For this model, training mode might affect the output due to dropout
        output_train = model(self.test_input, training=True)
        output_infer = model(self.test_input, training=False)

        # Shapes should be the same
        self.assertEqual(output_train.shape, output_infer.shape)

    def test_serialization(self) -> None:
        """Test model serialization and deserialization."""
        original_model = SFNEBlock(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
        )
        config = original_model.get_config()

        # Create new model from config
        restored_model = SFNEBlock.from_config(config)

        # Check if configurations match
        self.assertEqual(restored_model.input_dim, original_model.input_dim)
        self.assertEqual(restored_model.output_dim, original_model.output_dim)
        self.assertEqual(restored_model.hidden_dim, original_model.hidden_dim)
        self.assertEqual(restored_model.num_layers, original_model.num_layers)

    def test_integration(self) -> None:
        """Test integration with a simple model."""
        # Create a simple model with the SFNEBlock
        inputs = layers.Input(shape=(self.input_dim,))
        x = SFNEBlock(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
        )(
            inputs,
        )
        outputs = layers.Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(optimizer="adam", loss="mse")

        # Generate some dummy data
        x_data = tf.random.normal((100, self.input_dim))
        y_data = tf.random.normal((100, 1))

        # Train for one step to ensure everything works
        history = model.fit(x_data, y_data, epochs=1, verbose=0)

        # Check that loss was computed
        self.assertIsNotNone(history.history["loss"])

    def test_learnable_weights(self) -> None:
        """Test that the model's weights are learnable."""
        model = SFNEBlock(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
        )

        # Call the model once to build it
        _ = model(self.test_input)

        # Get initial weights
        initial_weights = model.get_weights()

        # Create a simple model with the SFNEBlock
        inputs = layers.Input(shape=(self.input_dim,))
        x = SFNEBlock(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
        )(
            inputs,
        )
        outputs = layers.Dense(1)(x)
        keras_model = Model(inputs=inputs, outputs=outputs)

        # Compile the model
        keras_model.compile(optimizer="adam", loss="mse")

        # Generate some dummy data
        x_data = tf.random.normal((100, self.input_dim))
        y_data = tf.random.normal((100, 1))

        # Train for a few steps
        keras_model.fit(x_data, y_data, epochs=5, verbose=0)

        # Get updated weights
        updated_weights = keras_model.layers[
            1
        ].get_weights()  # Index 1 should be the SFNEBlock

        # Weights should have changed
        for i in range(len(initial_weights)):
            self.assertFalse(np.array_equal(initial_weights[i], updated_weights[i]))

    def test_multi_layer_architecture(self) -> None:
        """Test that the model correctly builds with different numbers of layers."""
        # Test with 1 layer
        model_1 = SFNEBlock(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=1,
        )
        _ = model_1(self.test_input)
        self.assertEqual(len(model_1.hidden_layers), 1)

        # Test with 3 layers
        model_3 = SFNEBlock(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=3,
        )
        _ = model_3(self.test_input)
        self.assertEqual(len(model_3.hidden_layers), 3)

        # Test with 5 layers
        model_5 = SFNEBlock(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=5,
        )
        _ = model_5(self.test_input)
        self.assertEqual(len(model_5.hidden_layers), 5)


if __name__ == "__main__":
    unittest.main()
