"""Unit tests for the TransformerBlock layer.

Note: TensorFlow is used in tests for validation purposes only.
The actual layer implementation uses only Keras 3 operations.
"""

import unittest
import numpy as np
import tensorflow as tf  # Used for testing only
from keras import Model, layers, ops
from kmr.layers.TransformerBlock import TransformerBlock


class TestTransformerBlock(unittest.TestCase):
    """Test cases for the TransformerBlock layer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Using TensorFlow for test data generation only
        self.batch_size = 16
        self.seq_length = 10
        self.feature_dim = 32
        
        # Create test data
        self.test_3d_input = tf.random.normal(
            (self.batch_size, self.seq_length, self.feature_dim)
        )
        self.test_2d_input = tf.random.normal(
            (self.batch_size, self.feature_dim)
        )

    def test_initialization(self) -> None:
        """Test initialization with valid parameters."""
        # Default initialization
        transformer = TransformerBlock()
        self.assertEqual(transformer.dim_model, 32)
        self.assertEqual(transformer.num_heads, 3)
        self.assertEqual(transformer.ff_units, 16)
        self.assertEqual(transformer.dropout_rate, 0.2)
        
        # Custom initialization
        transformer = TransformerBlock(
            dim_model=64,
            num_heads=4,
            ff_units=128,
            dropout_rate=0.1,
            name="custom_transformer"
        )
        self.assertEqual(transformer.dim_model, 64)
        self.assertEqual(transformer.num_heads, 4)
        self.assertEqual(transformer.ff_units, 128)
        self.assertEqual(transformer.dropout_rate, 0.1)
        self.assertEqual(transformer.name, "custom_transformer")

    def test_invalid_initialization(self) -> None:
        """Test initialization with invalid parameters."""
        # Invalid dim_model
        with self.assertRaises(ValueError):
            TransformerBlock(dim_model=0)
        with self.assertRaises(ValueError):
            TransformerBlock(dim_model=-10)
        
        # Invalid num_heads
        with self.assertRaises(ValueError):
            TransformerBlock(num_heads=0)
        with self.assertRaises(ValueError):
            TransformerBlock(num_heads=-2)
        
        # Invalid ff_units
        with self.assertRaises(ValueError):
            TransformerBlock(ff_units=0)
        with self.assertRaises(ValueError):
            TransformerBlock(ff_units=-5)
        
        # Invalid dropout_rate
        with self.assertRaises(ValueError):
            TransformerBlock(dropout_rate=-0.1)
        with self.assertRaises(ValueError):
            TransformerBlock(dropout_rate=1.5)

    def test_build(self) -> None:
        """Test building the layer."""
        transformer = TransformerBlock(
            dim_model=self.feature_dim,
            num_heads=4,
            ff_units=64
        )
        
        # Build the layer
        transformer.build((None, self.seq_length, self.feature_dim))
        
        # Check that all layers are initialized
        self.assertIsNotNone(transformer.multihead_attention)
        self.assertIsNotNone(transformer.dropout1)
        self.assertIsNotNone(transformer.add1)
        self.assertIsNotNone(transformer.layer_norm1)
        self.assertIsNotNone(transformer.ff1)
        self.assertIsNotNone(transformer.dropout2)
        self.assertIsNotNone(transformer.ff2)
        self.assertIsNotNone(transformer.add2)
        self.assertIsNotNone(transformer.layer_norm2)

    def test_output_shape_3d(self) -> None:
        """Test output shape with 3D input."""
        transformer = TransformerBlock(
            dim_model=self.feature_dim,
            num_heads=4,
            ff_units=64
        )
        
        # Convert to Keras tensor
        inputs = ops.convert_to_tensor(self.test_3d_input)
        outputs = transformer(inputs)
        
        # Check output shape
        self.assertEqual(outputs.shape, (self.batch_size, self.seq_length, self.feature_dim))

    def test_output_shape_2d(self) -> None:
        """Test output shape with 2D input."""
        transformer = TransformerBlock(
            dim_model=self.feature_dim,
            num_heads=4,
            ff_units=64
        )
        
        # Convert to Keras tensor
        inputs = ops.convert_to_tensor(self.test_2d_input)
        outputs = transformer(inputs)
        
        # Check output shape
        self.assertEqual(outputs.shape, (self.batch_size, self.feature_dim))

    def test_compute_output_shape(self) -> None:
        """Test compute_output_shape method."""
        transformer = TransformerBlock(
            dim_model=self.feature_dim,
            num_heads=4,
            ff_units=64
        )
        
        # Test with 3D input shape
        input_shape = (None, self.seq_length, self.feature_dim)
        output_shape = transformer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, input_shape)
        
        # Test with 2D input shape
        input_shape = (None, self.feature_dim)
        output_shape = transformer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, input_shape)

    def test_training_mode(self) -> None:
        """Test behavior in training mode vs inference mode."""
        transformer = TransformerBlock(
            dim_model=self.feature_dim,
            num_heads=4,
            ff_units=64,
            dropout_rate=0.5  # High dropout for testing
        )
        
        # Convert to Keras tensor
        inputs = ops.convert_to_tensor(self.test_3d_input)
        
        # Run in training mode
        outputs_training = transformer(inputs, training=True)
        
        # Run in inference mode
        outputs_inference = transformer(inputs, training=False)
        
        # Outputs should be different due to dropout
        self.assertFalse(np.allclose(
            outputs_training.numpy(), 
            outputs_inference.numpy(),
            rtol=1e-5, atol=1e-5
        ))

    def test_serialization(self) -> None:
        """Test serialization and deserialization."""
        # Create and build the layer
        original_transformer = TransformerBlock(
            dim_model=64,
            num_heads=4,
            ff_units=128,
            dropout_rate=0.1,
            name="test_transformer"
        )
        
        # Get the config
        config = original_transformer.get_config()
        
        # Check config values
        self.assertEqual(config["dim_model"], 64)
        self.assertEqual(config["num_heads"], 4)
        self.assertEqual(config["ff_units"], 128)
        self.assertEqual(config["dropout_rate"], 0.1)
        self.assertEqual(config["name"], "test_transformer")
        
        # Create a new layer from the config
        reconstructed_transformer = TransformerBlock.from_config(config)
        
        # Check that the reconstructed layer has the same config
        self.assertEqual(reconstructed_transformer.dim_model, original_transformer.dim_model)
        self.assertEqual(reconstructed_transformer.num_heads, original_transformer.num_heads)
        self.assertEqual(reconstructed_transformer.ff_units, original_transformer.ff_units)
        self.assertEqual(reconstructed_transformer.dropout_rate, original_transformer.dropout_rate)
        self.assertEqual(reconstructed_transformer.name, original_transformer.name)

    def test_model_integration(self) -> None:
        """Test integration with a Keras model."""
        # Create a simple model with the transformer block
        inputs = layers.Input(shape=(self.seq_length, self.feature_dim))
        x = TransformerBlock(
            dim_model=self.feature_dim,
            num_heads=4,
            ff_units=64
        )(inputs)
        outputs = layers.Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile the model
        model.compile(optimizer="adam", loss="mse")
        
        # Generate some data
        x_data = np.random.normal(size=(32, self.seq_length, self.feature_dim))
        y_data = np.random.normal(size=(32, self.seq_length, 1))
        
        # Train for one step to ensure everything works
        model.fit(x_data, y_data, epochs=1, batch_size=16, verbose=0)
        
        # Make a prediction
        predictions = model.predict(x_data[:1], verbose=0)
        self.assertEqual(predictions.shape, (1, self.seq_length, 1))

    def test_attention_mechanism(self) -> None:
        """Test that the attention mechanism is working correctly."""
        # Create a transformer with a single head for easier testing
        transformer = TransformerBlock(
            dim_model=self.feature_dim,
            num_heads=1,
            ff_units=64,
            dropout_rate=0.0  # No dropout for deterministic testing
        )
        
        # Create input with a clear pattern
        # First half of sequence has high values in first half of features
        # Second half of sequence has high values in second half of features
        pattern_input = np.zeros((1, self.seq_length, self.feature_dim))
        half_seq = self.seq_length // 2
        half_feat = self.feature_dim // 2
        
        # Set pattern
        pattern_input[0, :half_seq, :half_feat] = 1.0
        pattern_input[0, half_seq:, half_feat:] = 1.0
        
        # Convert to Keras tensor
        inputs = ops.convert_to_tensor(pattern_input)
        
        # Process through transformer
        outputs = transformer(inputs)
        
        # The attention mechanism should cause information to flow between the two halves
        # So the output should have non-zero values in all regions
        self.assertTrue(np.any(outputs.numpy()[0, :half_seq, half_feat:] > 0))
        self.assertTrue(np.any(outputs.numpy()[0, half_seq:, :half_feat] > 0))


if __name__ == "__main__":
    unittest.main() 