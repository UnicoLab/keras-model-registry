"""Unit tests for the BoostingEnsembleLayer layer.

Note: TensorFlow is used in tests for validation purposes only.
The actual layer implementation uses only Keras 3 operations.
"""

import unittest
import numpy as np
import tensorflow as tf  # Used for testing only
from keras import layers, Model
from kerasfactory.layers.BoostingEnsembleLayer import BoostingEnsembleLayer


class TestBoostingEnsembleLayer(unittest.TestCase):
    """Test cases for the BoostingEnsembleLayer layer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.batch_size = 32
        self.input_dim = 16
        self.num_learners = 3
        self.learner_units = 8
        # Using TensorFlow for test data generation only
        self.test_input = tf.random.normal((self.batch_size, self.input_dim))
        tf.random.set_seed(42)  # For reproducibility

    def test_initialization(self) -> None:
        """Test layer initialization with various parameters."""
        # Test default initialization
        layer = BoostingEnsembleLayer()
        self.assertEqual(layer.num_learners, 3)
        self.assertEqual(layer.learner_units, 64)
        self.assertEqual(layer.hidden_activation, "relu")
        self.assertIsNone(layer.output_activation)
        self.assertTrue(layer.gamma_trainable)
        self.assertIsNone(layer.dropout_rate)

        # Test custom initialization
        layer = BoostingEnsembleLayer(
            num_learners=5,
            learner_units=[32, 16],
            hidden_activation="selu",
            output_activation="tanh",
            gamma_trainable=False,
            dropout_rate=0.1,
        )
        self.assertEqual(layer.num_learners, 5)
        self.assertEqual(layer.learner_units, [32, 16])
        self.assertEqual(layer.hidden_activation, "selu")
        self.assertEqual(layer.output_activation, "tanh")
        self.assertFalse(layer.gamma_trainable)
        self.assertEqual(layer.dropout_rate, 0.1)

    def test_invalid_initialization(self) -> None:
        """Test layer initialization with invalid parameters."""
        # Test invalid num_learners
        with self.assertRaises(ValueError):
            BoostingEnsembleLayer(num_learners=0)
        with self.assertRaises(ValueError):
            BoostingEnsembleLayer(num_learners=-1)

        # Test invalid dropout rate
        with self.assertRaises(ValueError):
            BoostingEnsembleLayer(dropout_rate=-0.1)
        with self.assertRaises(ValueError):
            BoostingEnsembleLayer(dropout_rate=1.0)

    def test_build(self) -> None:
        """Test layer building with different configurations."""
        # Test with default parameters
        layer = BoostingEnsembleLayer(
            num_learners=self.num_learners,
            learner_units=self.learner_units,
        )
        layer.build(input_shape=(None, self.input_dim))

        # Check if learners are created
        self.assertEqual(len(layer.learners), self.num_learners)
        # Check if alpha weights are created
        self.assertIsNotNone(layer.alpha)
        self.assertEqual(layer.alpha.shape, (self.num_learners,))
        self.assertTrue(layer.alpha.trainable)

    def test_output_shape(self) -> None:
        """Test output shape preservation."""
        # Test with default input
        layer = BoostingEnsembleLayer(
            num_learners=self.num_learners,
            learner_units=self.learner_units,
        )
        output = layer(self.test_input)
        self.assertEqual(output.shape, self.test_input.shape)

        # Test with different input shapes
        test_shapes = [(16, 8), (64, 32), (128, 64)]
        for shape in test_shapes:
            # Create new layer instance for each shape
            layer = BoostingEnsembleLayer(num_learners=3, learner_units=shape[1] // 2)
            test_input = tf.random.normal((shape[0], shape[1]))
            output = layer(test_input)
            self.assertEqual(output.shape, test_input.shape)

    def test_training_mode(self) -> None:
        """Test layer behavior in training and inference modes."""
        layer = BoostingEnsembleLayer(
            num_learners=self.num_learners,
            learner_units=self.learner_units,
            dropout_rate=0.5,
        )

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
        original_layer = BoostingEnsembleLayer(
            num_learners=5,
            learner_units=[32, 16],
            hidden_activation="selu",
            output_activation="tanh",
            gamma_trainable=False,
            dropout_rate=0.1,
        )
        config = original_layer.get_config()

        # Create new layer from config
        restored_layer = BoostingEnsembleLayer.from_config(config)

        # Check if configurations match
        self.assertEqual(restored_layer.num_learners, original_layer.num_learners)
        self.assertEqual(restored_layer.learner_units, original_layer.learner_units)
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

    def test_ensemble_weights(self) -> None:
        """Test the effect of ensemble weights on the output."""
        layer = BoostingEnsembleLayer(num_learners=2, learner_units=self.learner_units)
        layer.build(input_shape=(None, self.input_dim))

        # Set alpha weights to strongly favor the first learner
        layer.alpha.assign([10.0, -10.0])
        output1 = layer(self.test_input)

        # Set alpha weights to strongly favor the second learner
        layer.alpha.assign([-10.0, 10.0])
        output2 = layer(self.test_input)

        # Outputs should be different when weights are different
        self.assertFalse(tf.reduce_all(tf.equal(output1, output2)))

    def test_integration(self) -> None:
        """Test integration with a simple model."""
        # Create a simple model with the ensemble layer
        inputs = layers.Input(shape=(self.input_dim,))
        x = BoostingEnsembleLayer(
            num_learners=self.num_learners,
            learner_units=self.learner_units,
        )(inputs)
        outputs = layers.Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(optimizer="adam", loss="mse")

        # Generate some dummy data
        x_data = tf.random.normal((100, self.input_dim))
        y_data = tf.random.normal((100, 1))

        # Train for one step to ensure everything works
        history = model.fit(x_data, y_data, epochs=1, verbose=0)

        # Check that loss was computed
        self.assertIsNotNone(history.history["loss"])

    def test_learnable_weights(self) -> None:
        """Test that the layer's weights are learnable."""
        # Create a layer instance
        layer = BoostingEnsembleLayer(
            num_learners=self.num_learners,
            learner_units=self.learner_units,
        )

        # Call the layer once to build it
        _ = layer(self.test_input)

        # Get initial weights
        initial_alpha = layer.alpha.numpy().copy()

        # Create a simple model with the layer
        inputs = layers.Input(shape=(self.input_dim,))
        x = BoostingEnsembleLayer(
            num_learners=self.num_learners,
            learner_units=self.learner_units,
        )(inputs)
        outputs = layers.Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(optimizer="adam", loss="mse")

        # Generate some dummy data
        x_data = tf.random.normal((100, self.input_dim))
        y_data = tf.random.normal((100, 1))

        # Train for a few steps
        model.fit(x_data, y_data, epochs=5, verbose=0)

        # Get updated weights
        updated_alpha = model.layers[
            1
        ].alpha.numpy()  # Index 1 should be the BoostingEnsembleLayer

        # Alpha weights should have changed
        self.assertFalse(np.array_equal(initial_alpha, updated_alpha))

    def test_boosting_block_interaction(self) -> None:
        """Test that the boosting blocks are properly interacting with the ensemble."""
        # Create a layer with 2 learners for simplicity
        layer = BoostingEnsembleLayer(num_learners=2, learner_units=self.learner_units)

        # Call the layer once to build it
        _ = layer(self.test_input)

        # Check that each learner produces different outputs
        learner_outputs = [learner(self.test_input) for learner in layer.learners]
        self.assertFalse(
            tf.reduce_all(tf.equal(learner_outputs[0], learner_outputs[1])),
        )

        # Check that the ensemble output is a weighted combination of learner outputs
        # First, set alpha to known values
        layer.alpha.assign([0.5, 0.5])  # Equal weights

        # Get ensemble output
        ensemble_output = layer(self.test_input)

        # Manually compute the expected output
        weights = tf.nn.softmax(tf.constant([0.5, 0.5]))
        expected_output = (
            weights[0] * learner_outputs[0] + weights[1] * learner_outputs[1]
        )

        # Check that outputs are close
        self.assertTrue(tf.reduce_all(tf.abs(ensemble_output - expected_output) < 1e-5))

    def test_multi_layer_architecture(self) -> None:
        """Test that the layer works with multi-layer boosting blocks."""
        # Create a layer with multi-layer boosting blocks
        layer = BoostingEnsembleLayer(
            num_learners=2,
            learner_units=[32, 16, 8],  # Three hidden layers
            hidden_activation="relu",
            output_activation="linear",
        )

        # Call the layer to build it
        output = layer(self.test_input)

        # Check output shape
        self.assertEqual(output.shape, self.test_input.shape)

        # Check that each learner has the expected architecture
        for learner in layer.learners:
            # The BoostingBlock should have multiple hidden layers
            self.assertGreater(len(learner.hidden_layers), 1)

            # Check the dimensions of each hidden layer
            self.assertEqual(learner.hidden_layers[0].units, 32)
            self.assertEqual(learner.hidden_layers[1].units, 16)
            self.assertEqual(learner.hidden_layers[2].units, 8)


if __name__ == "__main__":
    unittest.main()
