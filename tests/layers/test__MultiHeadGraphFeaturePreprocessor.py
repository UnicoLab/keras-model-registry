"""Unit tests for the MultiHeadGraphFeaturePreprocessor layer.

Note: TensorFlow is used in tests for validation purposes only.
The actual layer implementation uses only Keras 3 operations.
"""

import unittest
import numpy as np
from keras import Model, layers, ops
from keras import utils, random
from kerasfactory.layers import MultiHeadGraphFeaturePreprocessor


class TestMultiHeadGraphFeaturePreprocessor(unittest.TestCase):
    """Test cases for the MultiHeadGraphFeaturePreprocessor layer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.batch_size = 32
        self.num_features = 10
        self.embed_dim = 16
        self.num_heads = 4
        self.dropout_rate = 0.1
        # Using Keras utils for random seed
        utils.set_random_seed(42)  # For reproducibility
        self.test_input = random.normal((self.batch_size, self.num_features))

    def test_initialization(self) -> None:
        """Test layer initialization with various parameters."""
        # Test default initialization
        layer = MultiHeadGraphFeaturePreprocessor()
        self.assertEqual(layer.embed_dim, 16)
        self.assertEqual(layer.num_heads, 4)
        self.assertEqual(layer.dropout_rate, 0.0)

        # Test custom initialization
        layer = MultiHeadGraphFeaturePreprocessor(
            embed_dim=32,
            num_heads=8,
            dropout_rate=0.2,
        )
        self.assertEqual(layer.embed_dim, 32)
        self.assertEqual(layer.num_heads, 8)
        self.assertEqual(layer.dropout_rate, 0.2)

    def test_invalid_initialization(self) -> None:
        """Test layer initialization with invalid parameters."""
        # Test invalid embed_dim
        with self.assertRaises(ValueError):
            MultiHeadGraphFeaturePreprocessor(embed_dim=0)
        with self.assertRaises(ValueError):
            MultiHeadGraphFeaturePreprocessor(embed_dim=-1)

        # Test invalid num_heads
        with self.assertRaises(ValueError):
            MultiHeadGraphFeaturePreprocessor(num_heads=0)
        with self.assertRaises(ValueError):
            MultiHeadGraphFeaturePreprocessor(num_heads=-1)

        # Test invalid embed_dim and num_heads combination
        with self.assertRaises(ValueError):
            MultiHeadGraphFeaturePreprocessor(
                embed_dim=15,
                num_heads=4,
            )  # Not divisible

        # Test invalid dropout_rate
        with self.assertRaises(ValueError):
            MultiHeadGraphFeaturePreprocessor(dropout_rate=-0.1)
        with self.assertRaises(ValueError):
            MultiHeadGraphFeaturePreprocessor(dropout_rate=1.0)

    def test_build(self) -> None:
        """Test layer building with different configurations."""
        # Test with default parameters
        layer = MultiHeadGraphFeaturePreprocessor(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )
        layer.build(input_shape=(None, self.num_features))

        # Check if components are created
        self.assertIsNotNone(layer.projection)
        self.assertIsNotNone(layer.q_dense)
        self.assertIsNotNone(layer.k_dense)
        self.assertIsNotNone(layer.v_dense)
        self.assertIsNotNone(layer.out_proj)
        self.assertIsNotNone(layer.final_dense)
        self.assertIsNotNone(layer.dropout_layer)

        # Check if dimensions are correct
        self.assertEqual(layer.num_features, self.num_features)
        self.assertEqual(layer.depth, self.embed_dim // self.num_heads)

        # Check if dropout layer is created only when needed
        layer_no_dropout = MultiHeadGraphFeaturePreprocessor(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout_rate=0.0,
        )
        layer_no_dropout.build(input_shape=(None, self.num_features))
        self.assertIsNone(layer_no_dropout.dropout_layer)

    def test_output_shape(self) -> None:
        """Test output shape preservation."""
        # Test with default input
        layer = MultiHeadGraphFeaturePreprocessor(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )
        output = layer(self.test_input)
        self.assertEqual(output.shape, self.test_input.shape)

        # Test with different input shapes
        test_shapes = [(16, 8), (64, 32), (128, 64)]
        for shape in test_shapes:
            # Create new layer instance for each shape
            layer = MultiHeadGraphFeaturePreprocessor(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
            )
            test_input = random.normal((shape[0], shape[1]))
            output = layer(test_input)
            self.assertEqual(output.shape, test_input.shape)

    def test_attention_mechanism(self) -> None:
        """Test that the attention mechanism works as expected."""
        layer = MultiHeadGraphFeaturePreprocessor(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout_rate=0.0,  # No dropout for deterministic testing
        )

        # Call the layer once to build it
        _ = layer(self.test_input)

        # Create a simple input with known patterns
        simple_input = ops.ones((2, 4))  # 2 samples, 4 features, all ones

        # The output should be different from the input due to the attention mechanism
        output = layer(simple_input)
        self.assertFalse(ops.all(ops.equal(output, simple_input)))

        # The output shape should be preserved
        self.assertEqual(output.shape, simple_input.shape)

    def test_training_mode(self) -> None:
        """Test layer behavior in training and inference modes."""
        layer = MultiHeadGraphFeaturePreprocessor(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,  # Use dropout to test training mode
        )

        # Outputs should be different in training mode due to dropout
        output_train = layer(self.test_input, training=True)
        output_infer = layer(self.test_input, training=False)

        # Shapes should be the same
        self.assertEqual(output_train.shape, output_infer.shape)

        # With dropout, outputs should be different
        self.assertFalse(ops.all(ops.equal(output_train, output_infer)))

        # Test with no dropout - outputs should be the same
        layer_no_dropout = MultiHeadGraphFeaturePreprocessor(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout_rate=0.0,
        )
        output_train = layer_no_dropout(self.test_input, training=True)
        output_infer = layer_no_dropout(self.test_input, training=False)
        self.assertTrue(ops.all(ops.equal(output_train, output_infer)))

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        original_layer = MultiHeadGraphFeaturePreprocessor(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )
        config = original_layer.get_config()

        # Create new layer from config
        restored_layer = MultiHeadGraphFeaturePreprocessor.from_config(config)

        # Check if configurations match
        self.assertEqual(restored_layer.embed_dim, original_layer.embed_dim)
        self.assertEqual(restored_layer.num_heads, original_layer.num_heads)
        self.assertEqual(restored_layer.dropout_rate, original_layer.dropout_rate)

        # Build both layers
        original_layer.build(input_shape=(None, self.num_features))
        restored_layer.build(input_shape=(None, self.num_features))

        # Check that outputs match
        original_output = original_layer(self.test_input)
        restored_output = restored_layer(self.test_input)

        # Since weights are initialized randomly, outputs won't match exactly
        # But shapes should match
        self.assertEqual(original_output.shape, restored_output.shape)

    def test_integration(self) -> None:
        """Test integration with a simple model."""
        # Create a simple model with the layer
        inputs = layers.Input(shape=(self.num_features,))
        x = MultiHeadGraphFeaturePreprocessor(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )(inputs)
        outputs = layers.Dense(1)(x)

        model = Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(optimizer="adam", loss="mse")

        # Generate some dummy data
        x_data = random.normal((100, self.num_features))
        y_data = random.normal((100, 1))

        # Train for one step to ensure everything works
        history = model.fit(x_data, y_data, epochs=1, verbose=0)

        # Check that loss was computed
        self.assertIsNotNone(history.history["loss"])

    def test_learnable_weights(self) -> None:
        """Test that the layer's weights are learnable."""
        # Create a layer instance
        layer = MultiHeadGraphFeaturePreprocessor(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )

        # Call the layer once to build it
        _ = layer(self.test_input)

        # Get initial weights (just check one component)
        initial_weights = layer.projection.get_weights()[0].copy()

        # Create a Keras model
        inputs = layers.Input(shape=(self.num_features,))
        outputs = layer(inputs)
        model = Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(optimizer="adam", loss="mse")

        # Generate some dummy data
        x_data = random.normal((100, self.num_features))
        y_data = random.normal((100, self.num_features))

        # Train for a few steps
        model.fit(x_data, y_data, epochs=5, verbose=0)

        # Get updated weights
        updated_weights = layer.projection.get_weights()[0]

        # Weights should have changed
        self.assertFalse(np.array_equal(initial_weights, updated_weights))

    def test_multi_head_behavior(self) -> None:
        """Test that the multi-head behavior works as expected."""
        # Create layers with different numbers of heads
        layer_single_head = MultiHeadGraphFeaturePreprocessor(
            embed_dim=self.embed_dim,
            num_heads=1,
            dropout_rate=0.0,
        )
        layer_multi_head = MultiHeadGraphFeaturePreprocessor(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout_rate=0.0,
        )

        # Call the layers once to build them
        output_single = layer_single_head(self.test_input)
        output_multi = layer_multi_head(self.test_input)

        # Both outputs should have the same shape
        self.assertEqual(output_single.shape, output_multi.shape)

        # But the outputs should be different due to different attention patterns
        self.assertFalse(ops.all(ops.equal(output_single, output_multi)))


if __name__ == "__main__":
    unittest.main()
