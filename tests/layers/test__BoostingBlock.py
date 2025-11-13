"""Unit tests for the BoostingBlock layer."""

import unittest
from keras import layers
import tensorflow as tf
from kerasfactory.layers.BoostingBlock import BoostingBlock


class TestBoostingBlock(unittest.TestCase):
    """Test cases for the BoostingBlock layer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.batch_size = 32
        self.input_dim = 16
        self.hidden_units = 8
        self.test_input = tf.random.normal((self.batch_size, self.input_dim))
        tf.random.set_seed(42)  # For reproducibility

    def test_initialization(self) -> None:
        """Test layer initialization with various parameters."""
        # Test default initialization
        layer = BoostingBlock()
        self.assertEqual(layer.hidden_units, [64])
        self.assertEqual(layer.hidden_activation, "relu")
        self.assertIsNone(layer.output_activation)
        self.assertTrue(layer.gamma_trainable)

        # Test custom initialization
        layer = BoostingBlock(
            hidden_units=[32, 16],
            hidden_activation="selu",
            output_activation="tanh",
            gamma_trainable=False,
            dropout_rate=0.1,
        )
        self.assertEqual(layer.hidden_units, [32, 16])
        self.assertEqual(layer.hidden_activation, "selu")
        self.assertEqual(layer.output_activation, "tanh")
        self.assertFalse(layer.gamma_trainable)
        self.assertEqual(layer.dropout_rate, 0.1)

    def test_invalid_initialization(self) -> None:
        """Test layer initialization with invalid parameters."""
        # Test invalid hidden units
        with self.assertRaises(ValueError):
            BoostingBlock(hidden_units=0)
        with self.assertRaises(ValueError):
            BoostingBlock(hidden_units=[-1])

        # Test invalid dropout rate
        with self.assertRaises(ValueError):
            BoostingBlock(dropout_rate=-0.1)
        with self.assertRaises(ValueError):
            BoostingBlock(dropout_rate=1.0)

    def test_build(self) -> None:
        """Test layer building with different configurations."""
        # Test single hidden layer
        layer = BoostingBlock(hidden_units=self.hidden_units)
        layer.build(input_shape=(None, self.input_dim))
        self.assertEqual(len(layer.hidden_layers), 1)
        self.assertIsInstance(layer.hidden_layers[0], layers.Dense)
        self.assertIsInstance(layer.output_layer, layers.Dense)
        self.assertTrue(layer.gamma.trainable)

        # Test multiple hidden layers with dropout
        layer = BoostingBlock(hidden_units=[8, 4], dropout_rate=0.1)
        layer.build(input_shape=(None, self.input_dim))
        self.assertEqual(len(layer.hidden_layers), 4)  # 2 dense + 2 dropout
        self.assertIsInstance(layer.hidden_layers[0], layers.Dense)
        self.assertIsInstance(layer.hidden_layers[1], layers.Dropout)

    def test_output_shape(self) -> None:
        """Test output shape preservation."""
        # Test with default input
        layer = BoostingBlock(hidden_units=self.hidden_units)
        output = layer(self.test_input)
        self.assertEqual(output.shape, self.test_input.shape)

        # Test with different input shapes
        test_shapes = [(16, 8), (64, 32), (128, 64)]
        for shape in test_shapes:
            # Create new layer instance for each shape
            layer = BoostingBlock(hidden_units=shape[1] // 2)
            test_input = tf.random.normal((shape[0], shape[1]))
            output = layer(test_input)
            self.assertEqual(output.shape, test_input.shape)

    def test_training_mode(self) -> None:
        """Test layer behavior in training and inference modes."""
        layer = BoostingBlock(hidden_units=self.hidden_units, dropout_rate=0.5)

        # Test that outputs are different in training vs inference
        output1 = layer(self.test_input, training=True)
        output2 = layer(self.test_input, training=True)
        layer(self.test_input, training=False)

        # Outputs should be different in training mode due to dropout
        self.assertFalse(tf.reduce_all(tf.equal(output1, output2)))
        # Output should be deterministic in inference mode
        self.assertTrue(
            tf.reduce_all(
                tf.equal(
                    layer(self.test_input, training=False),
                    layer(self.test_input, training=False),
                ),
            ),
        )

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        original_layer = BoostingBlock(
            hidden_units=[32, 16],
            hidden_activation="selu",
            output_activation="tanh",
            gamma_trainable=False,
            dropout_rate=0.1,
        )
        config = original_layer.get_config()

        # Create new layer from config
        restored_layer = BoostingBlock.from_config(config)

        # Check if configurations match
        self.assertEqual(restored_layer.hidden_units, original_layer.hidden_units)
        self.assertEqual(
            restored_layer.hidden_activation,
            original_layer.hidden_activation,
        )
        self.assertEqual(
            restored_layer.output_activation,
            original_layer.output_activation,
        )
        self.assertEqual(restored_layer.gamma_trainable, original_layer.gamma_trainable)
        self.assertEqual(restored_layer.dropout_rate, original_layer.dropout_rate)

    def test_gamma_effect(self) -> None:
        """Test the effect of gamma on the output."""
        layer = BoostingBlock(hidden_units=self.hidden_units)
        layer.build(input_shape=(None, self.input_dim))

        # Set gamma to zero
        layer.gamma.assign([0.0])
        output_zero_gamma = layer(self.test_input)
        # Output should equal input when gamma is zero
        self.assertTrue(tf.reduce_all(tf.equal(output_zero_gamma, self.test_input)))

        # Set gamma to one
        layer.gamma.assign([1.0])
        output_one_gamma = layer(self.test_input)
        # Output should be different from input when gamma is one
        self.assertFalse(tf.reduce_all(tf.equal(output_one_gamma, self.test_input)))


if __name__ == "__main__":
    unittest.main()
