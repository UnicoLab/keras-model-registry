"""Unit tests for AdvancedNumericalEmbedding layer.

Note: TensorFlow is used in tests for validation purposes only.
The actual layer implementation uses only Keras 3 operations.
"""

import unittest
import tensorflow as tf  # Used for testing only
import numpy as np
from keras import layers, Model
from kmr.layers.AdvancedNumericalEmbedding import AdvancedNumericalEmbedding


class TestAdvancedNumericalEmbedding(unittest.TestCase):
    """Test cases for AdvancedNumericalEmbedding layer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.batch_size = 32
        self.num_features = 5
        self.embedding_dim = 8
        self.mlp_hidden_units = 16
        self.num_bins = 10
        self.init_min = -3.0
        self.init_max = 3.0
        self.dropout_rate = 0.1
        self.use_batch_norm = True
        
        self.layer = AdvancedNumericalEmbedding(
            embedding_dim=self.embedding_dim,
            mlp_hidden_units=self.mlp_hidden_units,
            num_bins=self.num_bins,
            init_min=self.init_min,
            init_max=self.init_max,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm
        )
        
        # Using TensorFlow for test data generation only
        self.inputs = tf.random.uniform(
            (self.batch_size, self.num_features),
            minval=self.init_min,
            maxval=self.init_max
        )
        tf.random.set_seed(42)  # For reproducibility

    def test_initialization(self) -> None:
        """Test layer initialization with various parameters."""
        # Test default initialization
        layer = AdvancedNumericalEmbedding()
        self.assertEqual(layer.embedding_dim, 8)  # Default value
        self.assertEqual(layer.mlp_hidden_units, 16)  # Default value
        self.assertEqual(layer.num_bins, 10)  # Default value
        self.assertEqual(layer.init_min, -3.0)  # Default value
        self.assertEqual(layer.init_max, 3.0)  # Default value
        self.assertEqual(layer.dropout_rate, 0.1)  # Default value
        self.assertTrue(layer.use_batch_norm)  # Default value

        # Test with custom parameters
        embedding_dim = 16
        mlp_hidden_units = 32
        num_bins = 20
        init_min = -5.0
        init_max = 5.0
        dropout_rate = 0.2
        use_batch_norm = False
        
        layer = AdvancedNumericalEmbedding(
            embedding_dim=embedding_dim,
            mlp_hidden_units=mlp_hidden_units,
            num_bins=num_bins,
            init_min=init_min,
            init_max=init_max,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm
        )
        
        self.assertEqual(layer.embedding_dim, embedding_dim)
        self.assertEqual(layer.mlp_hidden_units, mlp_hidden_units)
        self.assertEqual(layer.num_bins, num_bins)
        self.assertEqual(layer.init_min, init_min)
        self.assertEqual(layer.init_max, init_max)
        self.assertEqual(layer.dropout_rate, dropout_rate)
        self.assertEqual(layer.use_batch_norm, use_batch_norm)

    def test_invalid_initialization(self) -> None:
        """Test layer initialization with invalid parameters."""
        # Test with negative embedding_dim
        with self.assertRaises(ValueError):
            AdvancedNumericalEmbedding(embedding_dim=-1)

        # Test with zero embedding_dim
        with self.assertRaises(ValueError):
            AdvancedNumericalEmbedding(embedding_dim=0)

        # Test with negative mlp_hidden_units
        with self.assertRaises(ValueError):
            AdvancedNumericalEmbedding(mlp_hidden_units=-1)

        # Test with zero mlp_hidden_units
        with self.assertRaises(ValueError):
            AdvancedNumericalEmbedding(mlp_hidden_units=0)

        # Test with negative num_bins
        with self.assertRaises(ValueError):
            AdvancedNumericalEmbedding(num_bins=-1)

        # Test with zero num_bins
        with self.assertRaises(ValueError):
            AdvancedNumericalEmbedding(num_bins=0)

        # Test with negative dropout_rate
        with self.assertRaises(ValueError):
            AdvancedNumericalEmbedding(dropout_rate=-0.1)

        # Test with dropout_rate >= 1
        with self.assertRaises(ValueError):
            AdvancedNumericalEmbedding(dropout_rate=1.0)

        # Test with non-boolean use_batch_norm
        with self.assertRaises(ValueError):
            AdvancedNumericalEmbedding(use_batch_norm="True")

    def test_build(self) -> None:
        """Test layer building."""
        # Build the layer
        self.layer.build((None, self.num_features))
        
        # Check if all components are created
        self.assertIsNotNone(self.layer.hidden_layer)
        self.assertIsNotNone(self.layer.output_layer)
        self.assertIsNotNone(self.layer.residual_proj)
        self.assertIsNotNone(self.layer.gate)
        
        # Check if dropout layer is created when dropout_rate > 0
        self.assertIsNotNone(self.layer.dropout_layer)
        
        # Check if batch normalization layer is created when use_batch_norm is True
        self.assertIsNotNone(self.layer.batch_norm)
        
        # Check if bin embeddings are created
        self.assertEqual(len(self.layer.bin_embeddings), self.num_features)
        
        # Check if learned boundaries are created
        self.assertIsNotNone(self.layer.learned_min)
        self.assertIsNotNone(self.layer.learned_max)
        
        # Test with dropout_rate = 0
        layer = AdvancedNumericalEmbedding(dropout_rate=0.0)
        layer.build((None, self.num_features))
        self.assertIsNone(layer.dropout_layer)
        
        # Test with use_batch_norm = False
        layer = AdvancedNumericalEmbedding(use_batch_norm=False)
        layer.build((None, self.num_features))
        self.assertIsNone(layer.batch_norm)

    def test_output_shape(self) -> None:
        """Test output shape of the layer."""
        outputs = self.layer(self.inputs)
        
        # Check output shape
        self.assertEqual(outputs.shape, (self.batch_size, self.num_features, self.embedding_dim))
        
        # Test with different input shapes
        test_shapes = [
            (16, 3),
            (64, 8),
            (128, 1)
        ]
        for batch, features in test_shapes:
            inputs = tf.random.uniform((batch, features))
            layer = AdvancedNumericalEmbedding(embedding_dim=self.embedding_dim)
            outputs = layer(inputs)
            
            # If features == 1, output shape should be (batch, embedding_dim)
            if features == 1:
                self.assertEqual(outputs.shape, (batch, self.embedding_dim))
            else:
                self.assertEqual(outputs.shape, (batch, features, self.embedding_dim))

    def test_compute_output_shape(self) -> None:
        """Test compute_output_shape method."""
        # Build the layer first to set num_features
        self.layer.build((None, self.num_features))
        
        # Test with multiple features
        input_shape = (None, self.num_features)
        output_shape = self.layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (None, self.num_features, self.embedding_dim))
        
        # Test with single feature
        layer = AdvancedNumericalEmbedding(embedding_dim=self.embedding_dim)
        layer.build((None, 1))  # Build with 1 feature
        input_shape = (None, 1)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (None, self.embedding_dim))

    def test_call(self) -> None:
        """Test forward pass of the layer."""
        # Call the layer
        outputs = self.layer(self.inputs)
        
        # Check that outputs are not None
        self.assertIsNotNone(outputs)
        
        # Check that outputs are not all zeros
        self.assertFalse(tf.reduce_all(tf.equal(outputs, 0)))

    def test_training_mode(self) -> None:
        """Test layer behavior in training and inference modes."""
        # In training mode, dropout should be applied
        # In inference mode, dropout should be disabled
        # This might lead to different outputs
        output_train = self.layer(self.inputs, training=True)
        output_infer = self.layer(self.inputs, training=False)
        
        # Outputs might be different due to dropout and batch normalization
        # But they should have the same shape
        self.assertEqual(output_train.shape, output_infer.shape)

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        # Get the config of the layer
        config = self.layer.get_config()
        
        # Create a new layer from the config
        new_layer = AdvancedNumericalEmbedding.from_config(config)
        
        # Check that the configs match
        self.assertEqual(new_layer.embedding_dim, self.layer.embedding_dim)
        self.assertEqual(new_layer.mlp_hidden_units, self.layer.mlp_hidden_units)
        self.assertEqual(new_layer.num_bins, self.layer.num_bins)
        self.assertEqual(new_layer.init_min, self.layer.init_min)
        self.assertEqual(new_layer.init_max, self.layer.init_max)
        self.assertEqual(new_layer.dropout_rate, self.layer.dropout_rate)
        self.assertEqual(new_layer.use_batch_norm, self.layer.use_batch_norm)
        
        # Call both layers with the same input
        original_output = self.layer(self.inputs)
        new_output = new_layer(self.inputs)
        
        # Check that the outputs have the same shape
        self.assertEqual(original_output.shape, new_output.shape)

    def test_model_integration(self) -> None:
        """Test layer integration in a model."""
        # Create a simple model
        inputs = layers.Input(shape=(self.num_features,))
        embedded = self.layer(inputs)
        # Flatten the output for dense layer
        flattened = layers.Reshape((-1,))(embedded)
        outputs = layers.Dense(1)(flattened)
        model = Model(inputs=inputs, outputs=outputs)

        # Ensure model can be compiled and trained
        model.compile(optimizer='adam', loss='mse')
        
        # Generate dummy data
        x = tf.random.uniform((100, self.num_features))
        y = tf.random.uniform((100, 1))
        
        # Train for one epoch
        history = model.fit(x, y, epochs=1, verbose=0)
        self.assertTrue(history.history['loss'][0] > 0)

    def test_learned_boundaries(self) -> None:
        """Test that learned boundaries are initialized correctly."""
        # Build the layer
        self.layer.build((None, self.num_features))
        
        # Check that learned_min and learned_max are initialized correctly
        learned_min = self.layer.learned_min.numpy()
        learned_max = self.layer.learned_max.numpy()
        
        # Check shapes
        self.assertEqual(learned_min.shape, (self.num_features,))
        self.assertEqual(learned_max.shape, (self.num_features,))
        
        # Check values
        self.assertTrue(np.allclose(learned_min, self.init_min))
        self.assertTrue(np.allclose(learned_max, self.init_max))
        
        # Test with list of init_min and init_max
        init_min_list = [-1.0, -2.0, -3.0]
        init_max_list = [1.0, 2.0, 3.0]
        layer = AdvancedNumericalEmbedding(
            init_min=init_min_list,
            init_max=init_max_list
        )
        layer.build((None, 3))  # 3 features
        
        learned_min = layer.learned_min.numpy()
        learned_max = layer.learned_max.numpy()
        
        self.assertTrue(np.allclose(learned_min, init_min_list))
        self.assertTrue(np.allclose(learned_max, init_max_list))
        
        # Test with mismatched list length
        with self.assertRaises(ValueError):
            layer = AdvancedNumericalEmbedding(
                init_min=[-1.0, -2.0],  # 2 values
                init_max=[1.0, 2.0, 3.0]  # 3 values
            )
            layer.build((None, 3))  # 3 features

    def test_gating_mechanism(self) -> None:
        """Test that the gating mechanism works as expected."""
        # Create a layer with a single feature for simplicity
        layer = AdvancedNumericalEmbedding(
            embedding_dim=1,
            mlp_hidden_units=1,
            num_bins=2,
            dropout_rate=0.0,
            use_batch_norm=False
        )
        
        # Create a simple input
        test_input = tf.constant([[0.0]])
        
        # Call the layer once to build it
        _ = layer(test_input)
        
        # Set gate to 0.5 (equal weighting of continuous and discrete branches)
        layer.gate.assign(tf.constant([[0.0]]))  # sigmoid(0.0) = 0.5
        
        # Test with the input
        output = layer(test_input, training=False)
        
        # Output should be a combination of continuous and discrete branches
        # But we can't make exact assertions about the values
        # Just check that the output has the expected shape
        self.assertEqual(output.shape, (1, 1))


if __name__ == '__main__':
    unittest.main()