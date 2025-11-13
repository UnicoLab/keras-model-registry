"""Unit tests for the DifferentialPreprocessingLayer layer.

Note: TensorFlow is used in tests for validation purposes only.
The actual layer implementation uses only Keras 3 operations.
"""

import unittest
import numpy as np
import tensorflow as tf  # Used for testing only
from keras import layers, Model
from kerasfactory.layers.DifferentialPreprocessingLayer import (
    DifferentialPreprocessingLayer,
)


class TestDifferentialPreprocessingLayer(unittest.TestCase):
    """Test cases for the DifferentialPreprocessingLayer layer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.batch_size = 32
        self.num_features = 16
        self.mlp_hidden_units = 8
        # Using TensorFlow for test data generation only
        # Create data with some NaN values
        self.test_input = tf.random.normal((self.batch_size, self.num_features))
        # Add some NaN values (about 10% of the values)
        mask = tf.random.uniform((self.batch_size, self.num_features)) < 0.1
        self.test_input = tf.where(
            mask,
            tf.constant(float("nan"), dtype=tf.float32),
            self.test_input,
        )
        tf.random.set_seed(42)  # For reproducibility

    def test_initialization(self) -> None:
        """Test layer initialization with various parameters."""
        # Test default initialization
        layer = DifferentialPreprocessingLayer(num_features=self.num_features)
        self.assertEqual(layer.num_features, self.num_features)
        self.assertEqual(layer.mlp_hidden_units, 4)  # Default value
        self.assertEqual(layer.num_candidates, 4)  # Fixed value

        # Test custom initialization
        layer = DifferentialPreprocessingLayer(num_features=8, mlp_hidden_units=16)
        self.assertEqual(layer.num_features, 8)
        self.assertEqual(layer.mlp_hidden_units, 16)

    def test_invalid_initialization(self) -> None:
        """Test layer initialization with invalid parameters."""
        # Test invalid num_features
        with self.assertRaises(ValueError):
            DifferentialPreprocessingLayer(num_features=0)
        with self.assertRaises(ValueError):
            DifferentialPreprocessingLayer(num_features=-1)

        # Test invalid mlp_hidden_units
        with self.assertRaises(ValueError):
            DifferentialPreprocessingLayer(num_features=8, mlp_hidden_units=0)
        with self.assertRaises(ValueError):
            DifferentialPreprocessingLayer(num_features=8, mlp_hidden_units=-1)

    def test_build(self) -> None:
        """Test layer building with different configurations."""
        # Test with default parameters
        layer = DifferentialPreprocessingLayer(
            num_features=self.num_features,
            mlp_hidden_units=self.mlp_hidden_units,
        )
        layer.build(input_shape=(None, self.num_features))

        # Check if weights and layers are created
        self.assertIsNotNone(layer.impute)
        self.assertIsNotNone(layer.gamma)
        self.assertIsNotNone(layer.beta)
        self.assertIsNotNone(layer.mlp_hidden)
        self.assertIsNotNone(layer.mlp_output)
        self.assertIsNotNone(layer.alpha)

        # Check weight shapes
        self.assertEqual(layer.impute.shape, (self.num_features,))
        self.assertEqual(layer.gamma.shape, (self.num_features,))
        self.assertEqual(layer.beta.shape, (self.num_features,))
        self.assertEqual(layer.alpha.shape, (layer.num_candidates,))

        # Check MLP layers
        self.assertEqual(layer.mlp_hidden.units, self.mlp_hidden_units)
        self.assertEqual(layer.mlp_output.units, self.num_features)

    def test_output_shape(self) -> None:
        """Test output shape preservation."""
        # Test with default input
        layer = DifferentialPreprocessingLayer(
            num_features=self.num_features,
            mlp_hidden_units=self.mlp_hidden_units,
        )
        output = layer(self.test_input)
        self.assertEqual(output.shape, self.test_input.shape)

        # Test with different input shapes
        test_shapes = [(16, 8), (64, 32), (128, 64)]
        for shape in test_shapes:
            # Create new layer instance for each shape
            layer = DifferentialPreprocessingLayer(
                num_features=shape[1],
                mlp_hidden_units=shape[1] // 2,
            )
            test_input = tf.random.normal((shape[0], shape[1]))
            # Add some NaN values
            mask = tf.random.uniform((shape[0], shape[1])) < 0.1
            test_input = tf.where(
                mask,
                tf.constant(float("nan"), dtype=tf.float32),
                test_input,
            )
            output = layer(test_input)
            self.assertEqual(output.shape, test_input.shape)

    def test_imputation(self) -> None:
        """Test that the layer properly imputes missing values."""
        # Create a simple input with known NaN positions
        test_input = tf.constant(
            [[1.0, 2.0, float("nan"), 4.0], [float("nan"), 2.0, 3.0, 4.0]],
            dtype=tf.float32,
        )

        layer = DifferentialPreprocessingLayer(num_features=4, mlp_hidden_units=2)

        # Set imputation values
        layer.build(input_shape=(None, 4))
        impute_values = tf.constant([10.0, 20.0, 30.0, 40.0], dtype=tf.float32)
        layer.impute.assign(impute_values)

        # Process the input
        output = layer(test_input)

        # Check that output has no NaN values
        self.assertFalse(tf.reduce_any(tf.math.is_nan(output)))

    def test_candidate_transformations(self) -> None:
        """Test that all candidate transformations are applied correctly."""
        # Create a simple input with no NaN values
        test_input = tf.constant(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            dtype=tf.float32,
        )

        layer = DifferentialPreprocessingLayer(num_features=4, mlp_hidden_units=2)

        # Build the layer
        layer.build(input_shape=(None, 4))

        # Set alpha weights to strongly favor each candidate in turn
        # and check that the output changes accordingly

        # First, get the original output with default weights
        original_output = layer(test_input)

        # Test each candidate by setting its weight much higher than others
        for i in range(layer.num_candidates):
            # Create weights that strongly favor the i-th candidate
            alpha_values = tf.constant([-10.0] * layer.num_candidates, dtype=tf.float32)
            alpha_values = tf.tensor_scatter_nd_update(
                alpha_values,
                [[i]],
                [10.0],  # High weight for the i-th candidate
            )
            layer.alpha.assign(alpha_values)

            # Get output with these weights
            output_i = layer(test_input)

            # If we're not testing the first candidate (which is the default),
            # the output should be different
            if i > 0:
                # The outputs should be different due to the different transformation
                self.assertFalse(
                    tf.reduce_all(tf.abs(output_i - original_output) < 1e-5),
                )

    def test_training_mode(self) -> None:
        """Test layer behavior in training and inference modes."""
        layer = DifferentialPreprocessingLayer(
            num_features=self.num_features,
            mlp_hidden_units=self.mlp_hidden_units,
        )

        # In this layer, training mode might affect the MLP branch
        # But for simplicity, we just check that the shapes are the same
        output_train = layer(self.test_input, training=True)
        output_infer = layer(self.test_input, training=False)

        # Shapes should be the same
        self.assertEqual(output_train.shape, output_infer.shape)

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        original_layer = DifferentialPreprocessingLayer(
            num_features=self.num_features,
            mlp_hidden_units=self.mlp_hidden_units,
        )
        config = original_layer.get_config()

        # Create new layer from config
        restored_layer = DifferentialPreprocessingLayer.from_config(config)

        # Check if configurations match
        self.assertEqual(restored_layer.num_features, original_layer.num_features)
        self.assertEqual(
            restored_layer.mlp_hidden_units,
            original_layer.mlp_hidden_units,
        )

    def test_integration(self) -> None:
        """Test integration with a simple model."""
        # Create a simple model with the preprocessing layer
        inputs = layers.Input(shape=(self.num_features,))
        x = DifferentialPreprocessingLayer(
            num_features=self.num_features,
            mlp_hidden_units=self.mlp_hidden_units,
        )(
            inputs,
        )
        outputs = layers.Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(optimizer="adam", loss="mse")

        # Generate some dummy data
        x_data = self.test_input.numpy()
        y_data = tf.random.normal((self.batch_size, 1)).numpy()

        # Train for one step to ensure everything works
        history = model.fit(x_data, y_data, epochs=1, verbose=0)

        # Check that loss was computed
        self.assertIsNotNone(history.history["loss"])

    def test_learnable_weights(self) -> None:
        """Test that the layer's weights are learnable."""
        # Create a simple model with the preprocessing layer
        inputs = layers.Input(shape=(self.num_features,))
        preproc_layer = DifferentialPreprocessingLayer(
            num_features=self.num_features,
            mlp_hidden_units=self.mlp_hidden_units,
        )
        x = preproc_layer(inputs)
        outputs = layers.Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(optimizer="adam", loss="mse")

        # Get initial weights
        initial_impute = preproc_layer.impute.numpy().copy()
        initial_gamma = preproc_layer.gamma.numpy().copy()
        initial_beta = preproc_layer.beta.numpy().copy()
        initial_alpha = preproc_layer.alpha.numpy().copy()

        # Generate some dummy data
        x_data = self.test_input.numpy()
        y_data = tf.random.normal((self.batch_size, 1)).numpy()

        # Train for several steps to ensure weights change
        model.fit(x_data, y_data, epochs=5, verbose=0)

        # Check that weights have changed
        self.assertFalse(np.allclose(initial_impute, preproc_layer.impute.numpy()))
        self.assertFalse(np.allclose(initial_gamma, preproc_layer.gamma.numpy()))
        self.assertFalse(np.allclose(initial_beta, preproc_layer.beta.numpy()))
        self.assertFalse(np.allclose(initial_alpha, preproc_layer.alpha.numpy()))


if __name__ == "__main__":
    unittest.main()
