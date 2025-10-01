"""Unit tests for the GatedFeatureFusion layer.

Note: TensorFlow is used in tests for validation purposes only.
The actual layer implementation uses only Keras 3 operations.
"""

import unittest
import tensorflow as tf  # Used for testing only
from keras import layers, Model
from kmr.layers.GatedFeatureFusion import GatedFeatureFusion


class TestGatedFeatureFusion(unittest.TestCase):
    """Test cases for the GatedFeatureFusion layer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.batch_size = 32
        self.feature_dim = 16
        # Using TensorFlow for test data generation only
        self.feat1 = tf.random.normal((self.batch_size, self.feature_dim))
        self.feat2 = tf.random.normal((self.batch_size, self.feature_dim))
        tf.random.set_seed(42)  # For reproducibility

    def test_initialization(self) -> None:
        """Test layer initialization with various parameters."""
        # Test default initialization
        layer = GatedFeatureFusion()
        self.assertEqual(layer.activation, "sigmoid")

        # Test custom initialization
        layer = GatedFeatureFusion(activation="relu")
        self.assertEqual(layer.activation, "relu")

    def test_build(self) -> None:
        """Test layer building with different configurations."""
        # Test with default parameters
        layer = GatedFeatureFusion()
        layer.build([(None, self.feature_dim), (None, self.feature_dim)])

        # Check if fusion_gate is created
        self.assertIsNotNone(layer.fusion_gate)
        self.assertEqual(layer.fusion_gate.units, self.feature_dim)
        self.assertEqual(layer.fusion_gate.activation.__name__, "sigmoid")

    def test_invalid_build(self) -> None:
        """Test layer building with invalid input shapes."""
        layer = GatedFeatureFusion()

        # Test with single input shape
        with self.assertRaises(ValueError):
            layer.build((None, self.feature_dim))

        # Test with mismatched feature dimensions
        with self.assertRaises(ValueError):
            layer.build([(None, self.feature_dim), (None, self.feature_dim * 2)])

    def test_call_validation(self) -> None:
        """Test input validation in call method."""
        layer = GatedFeatureFusion()
        layer.build([(None, self.feature_dim), (None, self.feature_dim)])

        # Test with single input
        with self.assertRaises(ValueError):
            layer(self.feat1)

        # Test with more than two inputs
        with self.assertRaises(ValueError):
            layer([self.feat1, self.feat2, self.feat1])

    def test_output_shape(self) -> None:
        """Test output shape preservation."""
        layer = GatedFeatureFusion()
        output = layer([self.feat1, self.feat2])
        self.assertEqual(output.shape, self.feat1.shape)

        # Test with different input shapes
        test_shapes = [(16, 8), (64, 32), (128, 64)]
        for shape in test_shapes:
            # Create new layer instance for each shape
            layer = GatedFeatureFusion()
            feat1 = tf.random.normal((shape[0], shape[1]))
            feat2 = tf.random.normal((shape[0], shape[1]))
            output = layer([feat1, feat2])
            self.assertEqual(output.shape, feat1.shape)

    def test_fusion_mechanism(self) -> None:
        """Test that the fusion mechanism works as expected."""
        layer = GatedFeatureFusion()

        # Call the layer once to build it properly
        _ = layer([self.feat1, self.feat2])

        # Now we can access the weights of the fusion_gate
        # Create weights that will produce a gate value of approximately 0.5
        weights = layer.fusion_gate.get_weights()

        # Set all weights to 0 and bias to 0
        # With sigmoid activation, this will produce a gate value of 0.5
        weights[0][:] = 0  # Weights
        weights[1][:] = 0  # Biases

        layer.fusion_gate.set_weights(weights)

        # Now test the output
        output = layer([self.feat1, self.feat2])
        expected = 0.5 * self.feat1 + 0.5 * self.feat2

        # Check that output is close to expected
        self.assertTrue(tf.reduce_all(tf.abs(output - expected) < 1e-5))

    def test_training_mode(self) -> None:
        """Test layer behavior in training and inference modes."""
        # For this layer, training mode doesn't affect the output
        # But we test it for completeness
        layer = GatedFeatureFusion()

        output_train = layer([self.feat1, self.feat2], training=True)
        output_infer = layer([self.feat1, self.feat2], training=False)

        # Outputs should be the same regardless of training mode
        self.assertTrue(tf.reduce_all(tf.equal(output_train, output_infer)))

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        original_layer = GatedFeatureFusion(activation="relu")
        config = original_layer.get_config()

        # Create new layer from config
        restored_layer = GatedFeatureFusion.from_config(config)

        # Check if configurations match
        self.assertEqual(restored_layer.activation, original_layer.activation)

        # Build both layers
        original_layer.build([(None, self.feature_dim), (None, self.feature_dim)])
        restored_layer.build([(None, self.feature_dim), (None, self.feature_dim)])

        # Check that outputs match
        original_output = original_layer([self.feat1, self.feat2])
        restored_output = restored_layer([self.feat1, self.feat2])

        # Since weights are initialized randomly, outputs won't match exactly
        # But shapes should match
        self.assertEqual(original_output.shape, restored_output.shape)

    def test_integration(self) -> None:
        """Test integration with a simple model."""
        # Create a simple model with the fusion layer
        input1 = layers.Input(shape=(self.feature_dim,), name="input1")
        input2 = layers.Input(shape=(self.feature_dim,), name="input2")

        fused = GatedFeatureFusion()([input1, input2])
        outputs = layers.Dense(1)(fused)

        model = Model(inputs=[input1, input2], outputs=outputs)

        # Compile the model
        model.compile(optimizer="adam", loss="mse")

        # Generate some dummy data
        x_data = {
            "input1": tf.random.normal((100, self.feature_dim)),
            "input2": tf.random.normal((100, self.feature_dim)),
        }
        y_data = tf.random.normal((100, 1))

        # Train for one step to ensure everything works
        history = model.fit(x_data, y_data, epochs=1, verbose=0)

        # Check that loss was computed
        self.assertIsNotNone(history.history["loss"])


if __name__ == "__main__":
    unittest.main()
