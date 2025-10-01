"""Unit tests for the SlowNetwork layer.

Note: TensorFlow is used in tests for validation purposes only.
The actual layer implementation uses only Keras 3 operations.
"""

import unittest
import tensorflow as tf  # Used for testing only
from keras import layers, Model
from kmr.layers.SlowNetwork import SlowNetwork


class TestSlowNetwork(unittest.TestCase):
    """Test cases for the SlowNetwork layer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.batch_size = 32
        self.input_dim = 16
        self.num_layers = 3
        self.units = 64
        # Using TensorFlow for test data generation only
        self.test_input = tf.random.normal((self.batch_size, self.input_dim))
        tf.random.set_seed(42)  # For reproducibility

    def test_initialization(self) -> None:
        """Test layer initialization with various parameters."""
        # Test default initialization
        layer = SlowNetwork(input_dim=self.input_dim)
        self.assertEqual(layer.input_dim, self.input_dim)
        self.assertEqual(layer.num_layers, 3)  # Default value
        self.assertEqual(layer.units, 128)  # Default value

        # Test custom initialization
        layer = SlowNetwork(input_dim=8, num_layers=5, units=32)
        self.assertEqual(layer.input_dim, 8)
        self.assertEqual(layer.num_layers, 5)
        self.assertEqual(layer.units, 32)

    def test_invalid_initialization(self) -> None:
        """Test layer initialization with invalid parameters."""
        # Test invalid input_dim
        with self.assertRaises(ValueError):
            SlowNetwork(input_dim=0)
        with self.assertRaises(ValueError):
            SlowNetwork(input_dim=-1)

        # Test invalid num_layers
        with self.assertRaises(ValueError):
            SlowNetwork(input_dim=8, num_layers=0)
        with self.assertRaises(ValueError):
            SlowNetwork(input_dim=8, num_layers=-1)

        # Test invalid units
        with self.assertRaises(ValueError):
            SlowNetwork(input_dim=8, units=0)
        with self.assertRaises(ValueError):
            SlowNetwork(input_dim=8, units=-1)

    def test_build(self) -> None:
        """Test layer building with different configurations."""
        # Test with default parameters
        layer = SlowNetwork(
            input_dim=self.input_dim,
            num_layers=self.num_layers,
            units=self.units,
        )
        layer.build(input_shape=(None, self.input_dim))

        # Check if hidden layers are created
        self.assertEqual(len(layer.hidden_layers), self.num_layers)

        # Check if output layer is created
        self.assertIsNotNone(layer.output_layer)

        # Check hidden layer dimensions
        for hidden_layer in layer.hidden_layers:
            self.assertEqual(hidden_layer.units, self.units)

        # Check output layer dimensions
        self.assertEqual(layer.output_layer.units, self.input_dim)

    def test_output_shape(self) -> None:
        """Test output shape preservation."""
        # Test with default input
        layer = SlowNetwork(
            input_dim=self.input_dim,
            num_layers=self.num_layers,
            units=self.units,
        )
        output = layer(self.test_input)
        self.assertEqual(output.shape, self.test_input.shape)

        # Test with different input shapes
        test_shapes = [(16, 8), (64, 32), (128, 64)]
        for shape in test_shapes:
            # Create new layer instance for each shape
            layer = SlowNetwork(input_dim=shape[1], num_layers=2, units=shape[1] * 2)
            test_input = tf.random.normal((shape[0], shape[1]))
            output = layer(test_input)
            self.assertEqual(output.shape, test_input.shape)

    def test_forward_pass(self) -> None:
        """Test the forward pass of the layer."""
        layer = SlowNetwork(
            input_dim=self.input_dim,
            num_layers=self.num_layers,
            units=self.units,
        )

        # Call the layer once to build it
        _ = layer(self.test_input)

        # Check that the output is different from the input
        # This is a basic test to ensure the layer is doing some transformation
        output = layer(self.test_input)
        self.assertFalse(tf.reduce_all(tf.equal(output, self.test_input)))

    def test_training_mode(self) -> None:
        """Test layer behavior in training and inference modes."""
        layer = SlowNetwork(
            input_dim=self.input_dim,
            num_layers=self.num_layers,
            units=self.units,
        )

        # For this layer, training mode doesn't affect the output
        # But we test it for completeness
        output_train = layer(self.test_input, training=True)
        output_infer = layer(self.test_input, training=False)

        # Shapes should be the same
        self.assertEqual(output_train.shape, output_infer.shape)

        # Outputs should be the same since training doesn't affect this layer
        self.assertTrue(tf.reduce_all(tf.equal(output_train, output_infer)))

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        original_layer = SlowNetwork(
            input_dim=self.input_dim,
            num_layers=self.num_layers,
            units=self.units,
        )
        config = original_layer.get_config()

        # Create new layer from config
        restored_layer = SlowNetwork.from_config(config)

        # Check if configurations match
        self.assertEqual(restored_layer.input_dim, original_layer.input_dim)
        self.assertEqual(restored_layer.num_layers, original_layer.num_layers)
        self.assertEqual(restored_layer.units, original_layer.units)

    def test_integration(self) -> None:
        """Test integration with a simple model."""
        # Create a simple model with the SlowNetwork layer
        inputs = layers.Input(shape=(self.input_dim,))
        x = SlowNetwork(
            input_dim=self.input_dim,
            num_layers=self.num_layers,
            units=self.units,
        )(inputs)
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


if __name__ == "__main__":
    unittest.main()
