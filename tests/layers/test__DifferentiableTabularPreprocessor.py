"""Unit tests for the DifferentiableTabularPreprocessor layer.

Note: TensorFlow is used in tests for validation purposes only.
The actual layer implementation uses only Keras 3 operations.
"""

import unittest
import numpy as np
import tensorflow as tf  # Used for testing only
from keras import layers, Model
from kmr.layers.DifferentiableTabularPreprocessor import (
    DifferentiableTabularPreprocessor,
)


class TestDifferentiableTabularPreprocessor(unittest.TestCase):
    """Test cases for the DifferentiableTabularPreprocessor layer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.batch_size = 32
        self.num_features = 16
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
        layer = DifferentiableTabularPreprocessor(num_features=self.num_features)
        self.assertEqual(layer.num_features, self.num_features)

        # Test custom initialization
        layer = DifferentiableTabularPreprocessor(num_features=8)
        self.assertEqual(layer.num_features, 8)

    def test_invalid_initialization(self) -> None:
        """Test layer initialization with invalid parameters."""
        # Test invalid num_features
        with self.assertRaises(ValueError):
            DifferentiableTabularPreprocessor(num_features=0)
        with self.assertRaises(ValueError):
            DifferentiableTabularPreprocessor(num_features=-1)

    def test_build(self) -> None:
        """Test layer building with different configurations."""
        # Test with default parameters
        layer = DifferentiableTabularPreprocessor(num_features=self.num_features)
        layer.build(input_shape=(None, self.num_features))

        # Check if weights are created
        self.assertIsNotNone(layer.impute)
        self.assertIsNotNone(layer.gamma)
        self.assertIsNotNone(layer.beta)

        # Check weight shapes
        self.assertEqual(layer.impute.shape, (self.num_features,))
        self.assertEqual(layer.gamma.shape, (self.num_features,))
        self.assertEqual(layer.beta.shape, (self.num_features,))

    def test_output_shape(self) -> None:
        """Test output shape preservation."""
        # Test with default input
        layer = DifferentiableTabularPreprocessor(num_features=self.num_features)
        output = layer(self.test_input)
        self.assertEqual(output.shape, self.test_input.shape)

        # Test with different input shapes
        test_shapes = [(16, 8), (64, 32), (128, 64)]
        for shape in test_shapes:
            # Create new layer instance for each shape
            layer = DifferentiableTabularPreprocessor(num_features=shape[1])
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

        layer = DifferentiableTabularPreprocessor(num_features=4)

        # Set imputation values
        layer.build(input_shape=(None, 4))
        impute_values = tf.constant([10.0, 20.0, 30.0, 40.0], dtype=tf.float32)
        layer.impute.assign(impute_values)

        # Process the input
        output = layer(test_input)

        # Check that output has no NaN values
        self.assertFalse(tf.reduce_any(tf.math.is_nan(output)))

        # Check that NaN values were replaced with imputation values
        # For the first row, the third value should be imputed
        self.assertAlmostEqual(output[0, 2].numpy(), 30.0, places=5)
        # For the second row, the first value should be imputed
        self.assertAlmostEqual(output[0, 0].numpy(), 1.0, places=5)  # Not imputed
        self.assertAlmostEqual(output[1, 0].numpy(), 10.0, places=5)  # Imputed

    def test_normalization(self) -> None:
        """Test that the layer properly normalizes values."""
        # Create a simple input with no NaN values
        test_input = tf.constant(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            dtype=tf.float32,
        )

        layer = DifferentiableTabularPreprocessor(num_features=4)

        # Set normalization parameters
        layer.build(input_shape=(None, 4))
        gamma_values = tf.constant([2.0, 2.0, 2.0, 2.0], dtype=tf.float32)
        beta_values = tf.constant([1.0, 1.0, 1.0, 1.0], dtype=tf.float32)
        layer.gamma.assign(gamma_values)
        layer.beta.assign(beta_values)

        # Process the input
        output = layer(test_input)

        # Check that values are normalized: output = gamma * input + beta
        expected_output = tf.constant(
            [[3.0, 5.0, 7.0, 9.0], [11.0, 13.0, 15.0, 17.0]],
            dtype=tf.float32,  # 2*[1,2,3,4] + 1  # 2*[5,6,7,8] + 1
        )

        self.assertTrue(tf.reduce_all(tf.abs(output - expected_output) < 1e-5))

    def test_training_mode(self) -> None:
        """Test layer behavior in training and inference modes."""
        layer = DifferentiableTabularPreprocessor(num_features=self.num_features)

        # For this layer, training mode should not affect the output
        output_train = layer(self.test_input, training=True)
        output_infer = layer(self.test_input, training=False)

        # Outputs should be identical
        self.assertTrue(tf.reduce_all(tf.abs(output_train - output_infer) < 1e-5))

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        original_layer = DifferentiableTabularPreprocessor(
            num_features=self.num_features,
        )
        config = original_layer.get_config()

        # Create new layer from config
        restored_layer = DifferentiableTabularPreprocessor.from_config(config)

        # Check if configurations match
        self.assertEqual(restored_layer.num_features, original_layer.num_features)

    def test_integration(self) -> None:
        """Test integration with a simple model."""
        # Create a simple model with the preprocessing layer
        inputs = layers.Input(shape=(self.num_features,))
        x = DifferentiableTabularPreprocessor(num_features=self.num_features)(inputs)
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
        preproc_layer = DifferentiableTabularPreprocessor(
            num_features=self.num_features,
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

        # Generate some dummy data
        x_data = self.test_input.numpy()
        y_data = tf.random.normal((self.batch_size, 1)).numpy()

        # Train for several steps to ensure weights change
        model.fit(x_data, y_data, epochs=5, verbose=0)

        # Check that weights have changed
        self.assertFalse(np.allclose(initial_impute, preproc_layer.impute.numpy()))
        self.assertFalse(np.allclose(initial_gamma, preproc_layer.gamma.numpy()))
        self.assertFalse(np.allclose(initial_beta, preproc_layer.beta.numpy()))


if __name__ == "__main__":
    unittest.main()
