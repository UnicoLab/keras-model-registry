"""Tests for the MultiResolutionTabularAttention layer.

Note: TensorFlow is used for validation purposes only. The actual layer implementation
uses Keras 3 operations.
"""

import unittest
import numpy as np
from keras import Model, layers

from kmr.layers import MultiResolutionTabularAttention


class TestMultiResolutionTabularAttention(unittest.TestCase):
    """Test cases for the MultiResolutionTabularAttention layer."""

    def setUp(self) -> None:
        """Set up test data."""
        # Create test data
        self.batch_size = 8
        self.num_samples = 10
        self.num_numerical_features = 5
        self.num_categorical_features = 3
        self.d_model = 16
        self.num_heads = 4

        # Generate random input data
        self.numerical_data = np.random.normal(
            size=(self.batch_size, self.num_samples, self.num_numerical_features),
        ).astype(np.float32)

        self.categorical_data = np.random.normal(
            size=(self.batch_size, self.num_samples, self.num_categorical_features),
        ).astype(np.float32)

    def test_initialization(self) -> None:
        """Test initialization with default and custom parameters."""
        # Test with default parameters
        layer = MultiResolutionTabularAttention(num_heads=4, d_model=16)
        self.assertEqual(layer.num_heads, 4)
        self.assertEqual(layer.d_model, 16)
        self.assertEqual(layer.dropout_rate, 0.1)

        # Test with custom parameters
        layer = MultiResolutionTabularAttention(
            num_heads=8,
            d_model=32,
            dropout_rate=0.2,
        )
        self.assertEqual(layer.num_heads, 8)
        self.assertEqual(layer.d_model, 32)
        self.assertEqual(layer.dropout_rate, 0.2)

    def test_invalid_initialization(self) -> None:
        """Test that invalid parameters raise appropriate exceptions."""
        # Test invalid num_heads
        with self.assertRaises(ValueError):
            MultiResolutionTabularAttention(num_heads=0, d_model=16)

        # Test invalid d_model
        with self.assertRaises(ValueError):
            MultiResolutionTabularAttention(num_heads=4, d_model=-16)

        # Test invalid dropout_rate
        with self.assertRaises(ValueError):
            MultiResolutionTabularAttention(num_heads=4, d_model=16, dropout_rate=1.5)

        # Test d_model not divisible by num_heads
        with self.assertRaises(ValueError):
            MultiResolutionTabularAttention(num_heads=3, d_model=10)

    def test_build(self) -> None:
        """Test that the layer builds correctly."""
        layer = MultiResolutionTabularAttention(num_heads=4, d_model=16)
        layer.build(
            [
                (self.batch_size, self.num_samples, self.num_numerical_features),
                (self.batch_size, self.num_samples, self.num_categorical_features),
            ],
        )

        # Check that all internal layers are initialized
        self.assertIsNotNone(layer.num_projection)
        self.assertIsNotNone(layer.cat_projection)
        self.assertIsNotNone(layer.num_attention)
        self.assertIsNotNone(layer.cat_attention)
        self.assertIsNotNone(layer.num_cat_attention)
        self.assertIsNotNone(layer.cat_num_attention)

    def test_invalid_input_shape(self) -> None:
        """Test that invalid input shapes raise appropriate exceptions."""
        layer = MultiResolutionTabularAttention(num_heads=4, d_model=16)

        # Test with single tensor instead of list
        with self.assertRaises(ValueError):
            layer.build(
                (self.batch_size, self.num_samples, self.num_numerical_features),
            )

        # Test with list of wrong length
        with self.assertRaises(ValueError):
            layer.build(
                [
                    (self.batch_size, self.num_samples, self.num_numerical_features),
                    (self.batch_size, self.num_samples, self.num_categorical_features),
                    (self.batch_size, self.num_samples, 2),
                ],
            )

        # Test with 2D input (missing samples dimension)
        with self.assertRaises(ValueError):
            layer.build(
                [
                    (self.batch_size, self.num_numerical_features),
                    (self.batch_size, self.num_categorical_features),
                ],
            )

        # Test during call
        layer.build(
            [
                (self.batch_size, self.num_samples, self.num_numerical_features),
                (self.batch_size, self.num_samples, self.num_categorical_features),
            ],
        )
        with self.assertRaises(ValueError):
            layer(self.numerical_data)  # Single tensor instead of list

    def test_output_shape(self) -> None:
        """Test that the output shape is correct."""
        layer = MultiResolutionTabularAttention(num_heads=4, d_model=16)
        numerical_output, categorical_output = layer(
            [self.numerical_data, self.categorical_data],
        )

        # Check output shapes
        self.assertEqual(
            numerical_output.shape,
            (self.batch_size, self.num_samples, self.d_model),
        )
        self.assertEqual(
            categorical_output.shape,
            (self.batch_size, self.num_samples, self.d_model),
        )

        # Test compute_output_shape method
        computed_shapes = layer.compute_output_shape(
            [
                (self.batch_size, self.num_samples, self.num_numerical_features),
                (self.batch_size, self.num_samples, self.num_categorical_features),
            ],
        )
        self.assertEqual(
            computed_shapes[0],
            (self.batch_size, self.num_samples, self.d_model),
        )
        self.assertEqual(
            computed_shapes[1],
            (self.batch_size, self.num_samples, self.d_model),
        )

    def test_training_mode(self) -> None:
        """Test that the layer behaves differently in training and inference modes."""
        layer = MultiResolutionTabularAttention(
            num_heads=4,
            d_model=16,
            dropout_rate=0.5,
        )

        # Get output in training mode
        numerical_training, categorical_training = layer(
            [self.numerical_data, self.categorical_data],
            training=True,
        )

        # Get output in inference mode
        numerical_inference, categorical_inference = layer(
            [self.numerical_data, self.categorical_data],
            training=False,
        )

        # The outputs should be different due to dropout
        self.assertFalse(
            np.allclose(numerical_training.numpy(), numerical_inference.numpy()),
        )
        self.assertFalse(
            np.allclose(categorical_training.numpy(), categorical_inference.numpy()),
        )

    def test_serialization(self) -> None:
        """Test that the layer can be serialized and deserialized."""
        layer = MultiResolutionTabularAttention(
            num_heads=4,
            d_model=16,
            dropout_rate=0.2,
        )
        config = layer.get_config()

        # Check that the config contains all the expected keys
        self.assertEqual(config["num_heads"], 4)
        self.assertEqual(config["d_model"], 16)
        self.assertEqual(config["dropout_rate"], 0.2)

        # Recreate the layer from the config
        recreated_layer = MultiResolutionTabularAttention.from_config(config)
        self.assertEqual(recreated_layer.num_heads, 4)
        self.assertEqual(recreated_layer.d_model, 16)
        self.assertEqual(recreated_layer.dropout_rate, 0.2)

    def test_model_integration(self) -> None:
        """Test that the layer can be integrated into a Keras model."""
        # Create inputs
        numerical_inputs = layers.Input(
            shape=(self.num_samples, self.num_numerical_features),
        )
        categorical_inputs = layers.Input(
            shape=(self.num_samples, self.num_categorical_features),
        )

        # Apply attention
        numerical_output, categorical_output = MultiResolutionTabularAttention(
            num_heads=4,
            d_model=16,
        )(
            [numerical_inputs, categorical_inputs],
        )

        # Combine outputs
        combined = layers.Concatenate(axis=-1)([numerical_output, categorical_output])
        outputs = layers.Dense(1)(combined)

        # Create model
        model = Model(inputs=[numerical_inputs, categorical_inputs], outputs=outputs)

        # Compile the model
        model.compile(optimizer="adam", loss="mse")

        # Generate some random target data
        y_data = np.random.normal(size=(self.batch_size, self.num_samples, 1))

        # Train the model for one step
        model.fit(
            [self.numerical_data, self.categorical_data],
            y_data,
            batch_size=self.batch_size,
            epochs=1,
            verbose=0,
        )

        # Make a prediction
        predictions = model.predict([self.numerical_data, self.categorical_data])
        self.assertEqual(predictions.shape, (self.batch_size, self.num_samples, 1))

    def test_cross_attention(self) -> None:
        """Test that the cross-attention mechanism is working correctly."""
        # Create a simple input with a clear pattern
        # One feature in numerical data has high correlation with one in categorical
        pattern_numerical = np.zeros(
            (1, self.num_samples, self.num_numerical_features),
            dtype=np.float32,
        )
        pattern_categorical = np.zeros(
            (1, self.num_samples, self.num_categorical_features),
            dtype=np.float32,
        )

        # Create a pattern where the first numerical feature and first categorical feature
        # have the same pattern
        for i in range(self.num_samples):
            value = (i % 5) / 5.0  # Create a repeating pattern
            pattern_numerical[0, i, 0] = value
            pattern_categorical[0, i, 0] = value

        # Apply attention
        layer = MultiResolutionTabularAttention(num_heads=4, d_model=16)
        numerical_output, categorical_output = layer(
            [pattern_numerical, pattern_categorical],
        )

        # The outputs should have captured the pattern
        # We can't make specific assertions about the values, but we can check
        # that the outputs are not all zeros
        self.assertFalse(np.allclose(numerical_output.numpy(), 0))
        self.assertFalse(np.allclose(categorical_output.numpy(), 0))


if __name__ == "__main__":
    unittest.main()
