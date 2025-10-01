"""Tests for the TabularAttention layer.

Note: TensorFlow is used for validation purposes only. The actual layer implementation
uses Keras 3 operations.
"""

import unittest
import numpy as np
from keras import Model, layers

from kmr.layers import TabularAttention


class TestTabularAttention(unittest.TestCase):
    """Test cases for the TabularAttention layer."""

    def setUp(self) -> None:
        """Set up test data."""
        # Create test data
        self.batch_size = 8
        self.num_samples = 10
        self.num_features = 5
        self.d_model = 16
        self.num_heads = 4

        # Generate random input data
        self.input_data = np.random.normal(
            size=(self.batch_size, self.num_samples, self.num_features),
        ).astype(
            np.float32,
        )

    def test_initialization(self) -> None:
        """Test initialization with default and custom parameters."""
        # Test with default parameters
        layer = TabularAttention(num_heads=4, d_model=16)
        self.assertEqual(layer.num_heads, 4)
        self.assertEqual(layer.d_model, 16)
        self.assertEqual(layer.dropout_rate, 0.1)

        # Test with custom parameters
        layer = TabularAttention(num_heads=8, d_model=32, dropout_rate=0.2)
        self.assertEqual(layer.num_heads, 8)
        self.assertEqual(layer.d_model, 32)
        self.assertEqual(layer.dropout_rate, 0.2)

    def test_invalid_initialization(self) -> None:
        """Test that invalid parameters raise appropriate exceptions."""
        # Test invalid num_heads
        with self.assertRaises(ValueError):
            TabularAttention(num_heads=0, d_model=16)

        # Test invalid d_model
        with self.assertRaises(ValueError):
            TabularAttention(num_heads=4, d_model=-16)

        # Test invalid dropout_rate
        with self.assertRaises(ValueError):
            TabularAttention(num_heads=4, d_model=16, dropout_rate=1.5)

        # Test d_model not divisible by num_heads
        with self.assertRaises(ValueError):
            TabularAttention(num_heads=3, d_model=10)

    def test_build(self) -> None:
        """Test that the layer builds correctly."""
        layer = TabularAttention(num_heads=4, d_model=16)
        layer.build((self.batch_size, self.num_samples, self.num_features))

        # Check that all internal layers are initialized
        self.assertIsNotNone(layer.input_projection)
        self.assertIsNotNone(layer.feature_attention)
        self.assertIsNotNone(layer.feature_layernorm)
        self.assertIsNotNone(layer.feature_dropout)
        self.assertIsNotNone(layer.sample_attention)
        self.assertIsNotNone(layer.sample_layernorm)
        self.assertIsNotNone(layer.sample_dropout)
        self.assertIsNotNone(layer.ffn_dense1)
        self.assertIsNotNone(layer.ffn_dense2)

    def test_invalid_input_shape(self) -> None:
        """Test that invalid input shapes raise appropriate exceptions."""
        layer = TabularAttention(num_heads=4, d_model=16)

        # Test 2D input (missing samples dimension)
        with self.assertRaises(ValueError):
            layer.build((self.batch_size, self.num_features))

        # Test during call
        layer.build((self.batch_size, self.num_samples, self.num_features))
        with self.assertRaises(ValueError):
            layer(np.random.normal(size=(self.batch_size, self.num_features)))

    def test_output_shape(self) -> None:
        """Test that the output shape is correct."""
        layer = TabularAttention(num_heads=4, d_model=16)
        output = layer(self.input_data)

        # Check output shape
        self.assertEqual(
            output.shape,
            (self.batch_size, self.num_samples, self.d_model),
        )

        # Test compute_output_shape method
        computed_shape = layer.compute_output_shape(
            (self.batch_size, self.num_samples, self.num_features),
        )
        self.assertEqual(
            computed_shape,
            (self.batch_size, self.num_samples, self.d_model),
        )

    def test_training_mode(self) -> None:
        """Test that the layer behaves differently in training and inference modes."""
        layer = TabularAttention(num_heads=4, d_model=16, dropout_rate=0.5)

        # Get output in training mode
        output_training = layer(self.input_data, training=True)

        # Get output in inference mode
        output_inference = layer(self.input_data, training=False)

        # The outputs should be different due to dropout
        self.assertFalse(np.allclose(output_training.numpy(), output_inference.numpy()))

    def test_serialization(self) -> None:
        """Test that the layer can be serialized and deserialized."""
        layer = TabularAttention(num_heads=4, d_model=16, dropout_rate=0.2)
        config = layer.get_config()

        # Check that the config contains all the expected keys
        self.assertEqual(config["num_heads"], 4)
        self.assertEqual(config["d_model"], 16)
        self.assertEqual(config["dropout_rate"], 0.2)

        # Recreate the layer from the config
        recreated_layer = TabularAttention.from_config(config)
        self.assertEqual(recreated_layer.num_heads, 4)
        self.assertEqual(recreated_layer.d_model, 16)
        self.assertEqual(recreated_layer.dropout_rate, 0.2)

    def test_model_integration(self) -> None:
        """Test that the layer can be integrated into a Keras model."""
        # Create a simple model with the TabularAttention layer
        inputs = layers.Input(shape=(self.num_samples, self.num_features))
        x = TabularAttention(num_heads=4, d_model=16)(inputs)
        outputs = layers.Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(optimizer="adam", loss="mse")

        # Generate some random data
        x_data = np.random.normal(
            size=(self.batch_size, self.num_samples, self.num_features),
        )
        y_data = np.random.normal(size=(self.batch_size, self.num_samples, 1))

        # Train the model for one step
        model.fit(x_data, y_data, batch_size=self.batch_size, epochs=1, verbose=0)

        # Make a prediction
        predictions = model.predict(x_data)
        self.assertEqual(predictions.shape, (self.batch_size, self.num_samples, 1))

    def test_attention_mechanism(self) -> None:
        """Test that the attention mechanism is working correctly."""
        # Create a simple input with a clear pattern
        # One feature has high values, others have low values
        pattern_input = np.zeros(
            (1, self.num_samples, self.num_features),
            dtype=np.float32,
        )
        pattern_input[0, :, 2] = 1.0  # Make the third feature stand out

        # Apply attention
        layer = TabularAttention(num_heads=4, d_model=16)
        output = layer(pattern_input)

        # The output should have captured the pattern
        # We can't make specific assertions about the values, but we can check
        # that the output is not all zeros
        self.assertFalse(np.allclose(output.numpy(), 0))


if __name__ == "__main__":
    unittest.main()
