"""Tests for AdvancedGraphFeatureLayer.

This module contains unit tests for the AdvancedGraphFeatureLayer class, which
implements a graph-based feature attention mechanism for tabular data with optional
hierarchical aggregation.
"""

import unittest
import tensorflow as tf
from kerasfactory.layers.AdvancedGraphFeature import AdvancedGraphFeatureLayer


class TestAdvancedGraphFeatureLayer(unittest.TestCase):
    """Test cases for AdvancedGraphFeatureLayer.

    Tests cover initialization, shape handling, training behavior,
    serialization, and edge cases.
    """

    def setUp(self) -> None:
        """Initialize common test variables."""
        self.batch_size = 32
        self.num_features = 10
        self.embed_dim = 16
        self.num_heads = 4
        self.dropout_rate = 0.1
        self.num_groups = 4
        tf.random.set_seed(42)

        # Create standard test inputs
        self.test_inputs = tf.random.normal(
            (self.batch_size, self.num_features),
            mean=0,
            stddev=1,
        )

    def test_initialization(self) -> None:
        """Test layer initialization with various parameter combinations."""
        # Test basic initialization
        layer = AdvancedGraphFeatureLayer(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )
        self.assertEqual(layer.embed_dim, self.embed_dim)
        self.assertEqual(layer.num_heads, self.num_heads)
        self.assertEqual(layer.dropout_rate, self.dropout_rate)
        self.assertFalse(layer.hierarchical)

        # Test hierarchical initialization
        layer = AdvancedGraphFeatureLayer(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            hierarchical=True,
            num_groups=self.num_groups,
        )
        self.assertTrue(layer.hierarchical)
        self.assertEqual(layer.num_groups, self.num_groups)

        # Test invalid embed_dim/num_heads combination
        with self.assertRaises(ValueError):
            AdvancedGraphFeatureLayer(embed_dim=15, num_heads=4)

        # Test missing num_groups with hierarchical=True
        with self.assertRaises(ValueError):
            AdvancedGraphFeatureLayer(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                hierarchical=True,
            )

    def test_build_validation(self) -> None:
        """Test layer building and shape inference."""
        layer = AdvancedGraphFeatureLayer(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )
        layer.build(input_shape=(None, self.num_features))

        # Check if all required layers and weights are created
        self.assertIsNotNone(layer.projection)
        self.assertIsNotNone(layer.edge_attention_dense)
        self.assertIsNotNone(layer.out_proj)
        self.assertIsNotNone(layer.layer_norm)

        # Test hierarchical mode weights
        layer = AdvancedGraphFeatureLayer(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            hierarchical=True,
            num_groups=self.num_groups,
        )
        layer.build(input_shape=(None, self.num_features))
        self.assertIsNotNone(layer.grouping_matrix)
        self.assertEqual(
            layer.grouping_matrix.shape,
            (self.num_features, self.num_groups),
        )

    def test_output_shape(self) -> None:
        """Test output tensor shapes for various configurations."""
        # Test basic configuration
        layer = AdvancedGraphFeatureLayer(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
        )
        output = layer(self.test_inputs)
        self.assertEqual(output.shape, (self.batch_size, self.num_features))

        # Test hierarchical configuration
        layer = AdvancedGraphFeatureLayer(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            hierarchical=True,
            num_groups=self.num_groups,
        )
        output = layer(self.test_inputs)
        self.assertEqual(output.shape, (self.batch_size, self.num_features))

    def test_training_mode(self) -> None:
        """Test layer behavior in training vs inference modes."""
        layer = AdvancedGraphFeatureLayer(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout_rate=0.5,  # High dropout for visible effect
        )

        # Run multiple times to check for consistency in each mode
        training_outputs = [layer(self.test_inputs, training=True) for _ in range(3)]
        inference_outputs = [layer(self.test_inputs, training=False) for _ in range(3)]

        # Training outputs should differ due to dropout
        training_diffs = [
            tf.reduce_max(abs(training_outputs[0] - out))
            for out in training_outputs[1:]
        ]
        self.assertTrue(all(diff > 0 for diff in training_diffs))

        # Inference outputs should be deterministic
        inference_diffs = [
            tf.reduce_max(abs(inference_outputs[0] - out))
            for out in inference_outputs[1:]
        ]
        self.assertTrue(all(diff < 1e-6 for diff in inference_diffs))

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        original_layer = AdvancedGraphFeatureLayer(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            hierarchical=True,
            num_groups=self.num_groups,
        )
        original_output = original_layer(self.test_inputs)

        # Get config and create new layer
        config = original_layer.get_config()
        new_layer = AdvancedGraphFeatureLayer.from_config(config)

        # Check if config contains all expected parameters
        self.assertEqual(config["embed_dim"], self.embed_dim)
        self.assertEqual(config["num_heads"], self.num_heads)
        self.assertEqual(config["dropout_rate"], self.dropout_rate)
        self.assertTrue(config["hierarchical"])
        self.assertEqual(config["num_groups"], self.num_groups)

        # Check if new layer produces same output
        new_output = new_layer(self.test_inputs)
        self.assertTrue(tf.reduce_all(tf.abs(original_output - new_output) < 1e-6))

    def test_attention_weights(self) -> None:
        """Test attention weight properties."""
        layer = AdvancedGraphFeatureLayer(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
        )

        # Create test input and run layer
        inputs = tf.random.normal((1, self.num_features))
        layer(inputs)

        # Get attention weights
        attention = layer._last_attention

        # Check shape
        expected_shape = (1, self.num_heads, self.num_features, self.num_features)
        self.assertEqual(attention.shape, expected_shape)

        # Check properties
        attention_sum = tf.reduce_sum(attention, axis=-1)
        self.assertTrue(tf.reduce_all(tf.abs(attention_sum - 1.0) < 1e-6))


if __name__ == "__main__":
    unittest.main()
