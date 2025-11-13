"""Tests for the VariableSelection layer.

Note: TensorFlow is used for validation purposes only. The actual layer implementation
uses Keras 3 operations.
"""

import unittest
import numpy as np
from keras import Model, layers

from kerasfactory.layers import VariableSelection


class TestVariableSelection(unittest.TestCase):
    """Test cases for the VariableSelection layer."""

    def setUp(self) -> None:
        """Set up test data."""
        # Create test data
        self.batch_size = 8
        self.nr_features = 5
        self.context_dim = 3
        self.units = 16

        # Generate random input data
        self.features = np.random.normal(
            size=(self.batch_size, self.nr_features),
        ).astype(np.float32)

        self.context = np.random.normal(
            size=(self.batch_size, self.context_dim),
        ).astype(np.float32)

    def test_initialization(self) -> None:
        """Test initialization with default and custom parameters."""
        # Test with default parameters
        layer = VariableSelection(nr_features=self.nr_features, units=self.units)
        self.assertEqual(layer.nr_features, self.nr_features)
        self.assertEqual(layer.units, self.units)
        self.assertEqual(layer.dropout_rate, 0.1)
        self.assertEqual(layer.use_context, False)

        # Test with custom parameters
        layer = VariableSelection(
            nr_features=self.nr_features,
            units=self.units,
            dropout_rate=0.2,
            use_context=True,
        )
        self.assertEqual(layer.nr_features, self.nr_features)
        self.assertEqual(layer.units, self.units)
        self.assertEqual(layer.dropout_rate, 0.2)
        self.assertEqual(layer.use_context, True)

    def test_invalid_initialization(self) -> None:
        """Test that invalid parameters raise appropriate exceptions."""
        # Test invalid nr_features
        with self.assertRaises(ValueError):
            VariableSelection(nr_features=0, units=self.units)

        # Test invalid units
        with self.assertRaises(ValueError):
            VariableSelection(nr_features=self.nr_features, units=-1)

        # Test invalid dropout_rate
        with self.assertRaises(ValueError):
            VariableSelection(
                nr_features=self.nr_features,
                units=self.units,
                dropout_rate=1.5,
            )

    def test_build_without_context(self) -> None:
        """Test that the layer builds correctly without context."""
        layer = VariableSelection(nr_features=self.nr_features, units=self.units)
        layer.build((self.batch_size, self.nr_features))

        # Check that all internal layers are initialized
        self.assertIsNotNone(layer.grn_var)
        self.assertEqual(len(layer.feature_grns), self.nr_features)
        self.assertIsNotNone(layer.softmax)

    def test_build_with_context(self) -> None:
        """Test that the layer builds correctly with context."""
        layer = VariableSelection(
            nr_features=self.nr_features,
            units=self.units,
            use_context=True,
        )
        layer.build(
            [(self.batch_size, self.nr_features), (self.batch_size, self.context_dim)],
        )

        # Check that all internal layers are initialized
        self.assertIsNotNone(layer.grn_var)
        self.assertEqual(len(layer.feature_grns), self.nr_features)
        self.assertIsNotNone(layer.softmax)

    def test_invalid_input_shape_without_context(self) -> None:
        """Test that invalid input shapes raise appropriate exceptions without context."""
        layer = VariableSelection(nr_features=self.nr_features, units=self.units)

        # Test with list instead of tensor
        with self.assertRaises(ValueError):
            layer.build(
                [
                    (self.batch_size, self.nr_features),
                    (self.batch_size, self.context_dim),
                ],
            )

        # Test with wrong feature dimension
        with self.assertRaises(ValueError):
            layer.build((self.batch_size, self.nr_features + 1))

        # Test during call
        layer.build((self.batch_size, self.nr_features))
        with self.assertRaises(ValueError):
            layer([self.features, self.context])  # List instead of tensor

    def test_invalid_input_shape_with_context(self) -> None:
        """Test that invalid input shapes raise appropriate exceptions with context."""
        layer = VariableSelection(
            nr_features=self.nr_features,
            units=self.units,
            use_context=True,
        )

        # Test with single tensor instead of list
        with self.assertRaises(ValueError):
            layer.build((self.batch_size, self.nr_features))

        # Test with list of wrong length
        with self.assertRaises(ValueError):
            layer.build(
                [
                    (self.batch_size, self.nr_features),
                    (self.batch_size, self.context_dim),
                    (self.batch_size, 2),
                ],
            )

        # Test with wrong feature dimension
        with self.assertRaises(ValueError):
            layer.build(
                [
                    (self.batch_size, self.nr_features + 1),
                    (self.batch_size, self.context_dim),
                ],
            )

        # Test during call
        layer.build(
            [(self.batch_size, self.nr_features), (self.batch_size, self.context_dim)],
        )
        with self.assertRaises(ValueError):
            layer(self.features)  # Single tensor instead of list

    def test_output_shape_without_context(self) -> None:
        """Test that the output shape is correct without context."""
        layer = VariableSelection(nr_features=self.nr_features, units=self.units)
        selected_features, feature_weights = layer(self.features)

        # Check output shapes
        self.assertEqual(selected_features.shape, (self.batch_size, self.units))
        self.assertEqual(feature_weights.shape, (self.batch_size, self.nr_features))

        # Test compute_output_shape method
        computed_shapes = layer.compute_output_shape(
            (self.batch_size, self.nr_features),
        )
        self.assertEqual(computed_shapes[0], (self.batch_size, self.units))
        self.assertEqual(computed_shapes[1], (self.batch_size, self.nr_features))

    def test_output_shape_with_context(self) -> None:
        """Test that the output shape is correct with context."""
        layer = VariableSelection(
            nr_features=self.nr_features,
            units=self.units,
            use_context=True,
        )
        selected_features, feature_weights = layer([self.features, self.context])

        # Check output shapes
        self.assertEqual(selected_features.shape, (self.batch_size, self.units))
        self.assertEqual(feature_weights.shape, (self.batch_size, self.nr_features))

        # Test compute_output_shape method
        computed_shapes = layer.compute_output_shape(
            [(self.batch_size, self.nr_features), (self.batch_size, self.context_dim)],
        )
        self.assertEqual(computed_shapes[0], (self.batch_size, self.units))
        self.assertEqual(computed_shapes[1], (self.batch_size, self.nr_features))

    def test_feature_weights_sum_to_one(self) -> None:
        """Test that feature weights sum to one."""
        # Without context
        layer_no_context = VariableSelection(
            nr_features=self.nr_features,
            units=self.units,
        )
        _, feature_weights_no_context = layer_no_context(self.features)

        # Check that weights sum to one for each batch
        weights_sum_no_context = np.sum(feature_weights_no_context.numpy(), axis=1)
        self.assertTrue(np.allclose(weights_sum_no_context, 1.0, atol=1e-5))

        # With context
        layer_with_context = VariableSelection(
            nr_features=self.nr_features,
            units=self.units,
            use_context=True,
        )
        _, feature_weights_with_context = layer_with_context(
            [self.features, self.context],
        )

        # Check that weights sum to one for each batch
        weights_sum_with_context = np.sum(feature_weights_with_context.numpy(), axis=1)
        self.assertTrue(np.allclose(weights_sum_with_context, 1.0, atol=1e-5))

    def test_training_mode(self) -> None:
        """Test that the layer behaves differently in training and inference modes."""
        layer = VariableSelection(
            nr_features=self.nr_features,
            units=self.units,
            dropout_rate=0.5,
        )

        # Get output in training mode
        selected_training, weights_training = layer(self.features, training=True)

        # Get output in inference mode
        selected_inference, weights_inference = layer(self.features, training=False)

        # The outputs should be different due to dropout
        self.assertFalse(
            np.allclose(selected_training.numpy(), selected_inference.numpy()),
        )

    def test_serialization(self) -> None:
        """Test that the layer can be serialized and deserialized."""
        layer = VariableSelection(
            nr_features=self.nr_features,
            units=self.units,
            dropout_rate=0.2,
            use_context=True,
        )
        config = layer.get_config()

        # Check that the config contains all the expected keys
        self.assertEqual(config["nr_features"], self.nr_features)
        self.assertEqual(config["units"], self.units)
        self.assertEqual(config["dropout_rate"], 0.2)
        self.assertEqual(config["use_context"], True)

        # Recreate the layer from the config
        recreated_layer = VariableSelection.from_config(config)
        self.assertEqual(recreated_layer.nr_features, self.nr_features)
        self.assertEqual(recreated_layer.units, self.units)
        self.assertEqual(recreated_layer.dropout_rate, 0.2)
        self.assertEqual(recreated_layer.use_context, True)

    def test_model_integration_without_context(self) -> None:
        """Test that the layer can be integrated into a Keras model without context."""
        # Create inputs
        inputs = layers.Input(shape=(self.nr_features,))

        # Apply variable selection
        selected_features, feature_weights = VariableSelection(
            nr_features=self.nr_features,
            units=self.units,
        )(inputs)

        # Use selected features for prediction
        outputs = layers.Dense(1)(selected_features)

        # Create model
        model = Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(optimizer="adam", loss="mse")

        # Generate some random target data
        y_data = np.random.normal(size=(self.batch_size, 1))

        # Train the model for one step
        model.fit(
            self.features,
            y_data,
            batch_size=self.batch_size,
            epochs=1,
            verbose=0,
        )

        # Make a prediction
        predictions = model.predict(self.features)
        self.assertEqual(predictions.shape, (self.batch_size, 1))

    def test_model_integration_with_context(self) -> None:
        """Test that the layer can be integrated into a Keras model with context."""
        # Create inputs
        feature_inputs = layers.Input(shape=(self.nr_features,))
        context_inputs = layers.Input(shape=(self.context_dim,))

        # Apply variable selection
        selected_features, feature_weights = VariableSelection(
            nr_features=self.nr_features,
            units=self.units,
            use_context=True,
        )([feature_inputs, context_inputs])

        # Use selected features for prediction
        outputs = layers.Dense(1)(selected_features)

        # Create model
        model = Model(inputs=[feature_inputs, context_inputs], outputs=outputs)

        # Compile the model
        model.compile(optimizer="adam", loss="mse")

        # Generate some random target data
        y_data = np.random.normal(size=(self.batch_size, 1))

        # Train the model for one step
        model.fit(
            [self.features, self.context],
            y_data,
            batch_size=self.batch_size,
            epochs=1,
            verbose=0,
        )

        # Make a prediction
        predictions = model.predict([self.features, self.context])
        self.assertEqual(predictions.shape, (self.batch_size, 1))


if __name__ == "__main__":
    unittest.main()
