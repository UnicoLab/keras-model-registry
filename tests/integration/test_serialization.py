"""
Integration tests for KMR serialization and deserialization.

These tests verify that KMR layers and models can be properly serialized
and deserialized using Keras serialization mechanisms.
"""

import unittest
import tempfile
import os
import numpy as np
import keras
from keras import Model, layers

from kmr.layers import (
    TabularAttention,
    AdvancedNumericalEmbedding,
    GatedFeatureFusion,
    VariableSelection,
    TransformerBlock,
    BoostingBlock,
    StochasticDepth,
    DateEncodingLayer,
    DateParsingLayer
)
from kmr.models import TerminatorModel, BaseFeedForwardModel


class TestLayerSerialization(unittest.TestCase):
    """Test serialization of individual KMR layers."""

    def setUp(self):
        """Set up test data."""
        self.batch_size = 8
        self.num_features = 10
        self.embed_dim = 16
        self.num_heads = 4
        
        # Create test data
        self.x = keras.random.normal((self.batch_size, self.num_features))
        self.context = keras.random.normal((self.batch_size, 5))

    def test_tabular_attention_serialization(self):
        """Test TabularAttention serialization."""
        # Create layer
        layer = TabularAttention(
            num_heads=self.num_heads,
            d_model=self.embed_dim,
            dropout_rate=0.1
        )
        
        # Reshape data for attention
        x_reshaped = keras.ops.reshape(self.x, (self.batch_size, 1, self.num_features))
        
        # Test forward pass
        output = layer(x_reshaped)
        
        # Test serialization
        config = layer.get_config()
        self.assertIsInstance(config, dict)
        self.assertEqual(config['num_heads'], self.num_heads)
        self.assertEqual(config['d_model'], self.embed_dim)
        self.assertEqual(config['dropout_rate'], 0.1)
        
        # Test deserialization
        reconstructed = TabularAttention.from_config(config)
        reconstructed_output = reconstructed(x_reshaped)
        
        # Check that outputs match
        np.testing.assert_allclose(output, reconstructed_output, rtol=1e-5)

    def test_advanced_numerical_embedding_serialization(self):
        """Test AdvancedNumericalEmbedding serialization."""
        # Create layer
        layer = AdvancedNumericalEmbedding(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads
        )
        
        # Test forward pass
        output = layer(self.x)
        
        # Test serialization
        config = layer.get_config()
        self.assertIsInstance(config, dict)
        self.assertEqual(config['embed_dim'], self.embed_dim)
        self.assertEqual(config['num_heads'], self.num_heads)
        
        # Test deserialization
        reconstructed = AdvancedNumericalEmbedding.from_config(config)
        reconstructed_output = reconstructed(self.x)
        
        # Check that outputs match
        np.testing.assert_allclose(output, reconstructed_output, rtol=1e-5)

    def test_gated_feature_fusion_serialization(self):
        """Test GatedFeatureFusion serialization."""
        # Create layer
        layer = GatedFeatureFusion()
        
        # Create two inputs for GatedFeatureFusion
        x1 = keras.random.normal((self.batch_size, self.num_features))
        x2 = keras.random.normal((self.batch_size, self.num_features))
        
        # Test forward pass
        output = layer([x1, x2])
        
        # Test serialization
        config = layer.get_config()
        self.assertIsInstance(config, dict)
        self.assertEqual(config['activation'], 'sigmoid')
        
        # Test deserialization
        reconstructed = GatedFeatureFusion.from_config(config)
        reconstructed_output = reconstructed([x1, x2])
        
        # Check that outputs match
        np.testing.assert_allclose(output, reconstructed_output, rtol=1e-5)

    def test_variable_selection_serialization(self):
        """Test VariableSelection serialization."""
        # Create layer
        layer = VariableSelection(
            nr_features=self.num_features,
            units=16,
            use_context=True
        )
        
        # Create 3D input for VariableSelection
        x_3d = keras.ops.expand_dims(self.x, axis=-1)
        
        # Test forward pass
        output = layer([x_3d, self.context])
        
        # Test serialization
        config = layer.get_config()
        self.assertIsInstance(config, dict)
        self.assertEqual(config['nr_features'], self.num_features)
        self.assertEqual(config['units'], 16)
        self.assertEqual(config['use_context'], True)
        
        # Test deserialization
        reconstructed = VariableSelection.from_config(config)
        reconstructed_output = reconstructed([x_3d, self.context])
        
        # Check that outputs match (VariableSelection returns tuple)
        self.assertEqual(len(output), 2)
        self.assertEqual(len(reconstructed_output), 2)
        np.testing.assert_allclose(output[0], reconstructed_output[0], rtol=1e-5)
        np.testing.assert_allclose(output[1], reconstructed_output[1], rtol=1e-5)

    def test_transformer_block_serialization(self):
        """Test TransformerBlock serialization."""
        # Create layer
        layer = TransformerBlock(
            dim_model=self.embed_dim,
            num_heads=self.num_heads,
            ff_units=32,
            dropout_rate=0.1
        )
        
        # Test forward pass
        output = layer(self.x)
        
        # Test serialization
        config = layer.get_config()
        self.assertIsInstance(config, dict)
        self.assertEqual(config['dim_model'], self.embed_dim)
        self.assertEqual(config['num_heads'], self.num_heads)
        self.assertEqual(config['ff_units'], 32)
        self.assertEqual(config['dropout_rate'], 0.1)
        
        # Test deserialization
        reconstructed = TransformerBlock.from_config(config)
        reconstructed_output = reconstructed(self.x)
        
        # Check that outputs match
        np.testing.assert_allclose(output, reconstructed_output, rtol=1e-5)

    def test_boosting_block_serialization(self):
        """Test BoostingBlock serialization."""
        # Create layer
        layer = BoostingBlock(
            num_estimators=3,
            hidden_units=16,
            dropout_rate=0.1
        )
        
        # Test forward pass
        output = layer(self.x)
        
        # Test serialization
        config = layer.get_config()
        self.assertIsInstance(config, dict)
        self.assertEqual(config['num_estimators'], 3)
        self.assertEqual(config['hidden_units'], 16)
        self.assertEqual(config['dropout_rate'], 0.1)
        
        # Test deserialization
        reconstructed = BoostingBlock.from_config(config)
        reconstructed_output = reconstructed(self.x)
        
        # Check that outputs match
        np.testing.assert_allclose(output, reconstructed_output, rtol=1e-5)


class TestModelSerialization(unittest.TestCase):
    """Test serialization of KMR models."""

    def setUp(self):
        """Set up test data."""
        self.batch_size = 8
        self.input_dim = 12
        self.context_dim = 6
        self.num_features = 10
        
        # Create test data
        self.x = keras.random.normal((self.batch_size, self.input_dim))
        self.context = keras.random.normal((self.batch_size, self.context_dim))
        self.x_features = keras.random.normal((self.batch_size, self.num_features))

    def test_terminator_model_serialization(self):
        """Test TerminatorModel serialization."""
        # Create model
        model = TerminatorModel(
            input_dim=self.input_dim,
            context_dim=self.context_dim,
            output_dim=1,
            num_blocks=2,
            hidden_dim=32
        )
        
        # Build model
        model.build(input_shape=[(self.batch_size, self.input_dim), (self.batch_size, self.context_dim)])
        
        # Test forward pass
        output = model([self.x, self.context])
        
        # Test serialization
        config = model.get_config()
        self.assertIsInstance(config, dict)
        self.assertEqual(config['input_dim'], self.input_dim)
        self.assertEqual(config['context_dim'], self.context_dim)
        self.assertEqual(config['output_dim'], 1)
        self.assertEqual(config['num_blocks'], 2)
        self.assertEqual(config['hidden_dim'], 32)
        
        # Test deserialization
        reconstructed = TerminatorModel.from_config(config)
        reconstructed.build(input_shape=[(self.batch_size, self.input_dim), (self.batch_size, self.context_dim)])
        reconstructed_output = reconstructed([self.x, self.context])
        
        # Check that outputs match
        np.testing.assert_allclose(output, reconstructed_output, rtol=1e-5)

    def test_feed_forward_model_serialization(self):
        """Test BaseFeedForwardModel serialization."""
        # Create model
        model = BaseFeedForwardModel(
            feature_names=[f'feature_{i}' for i in range(self.num_features)],
            hidden_units=[32, 16],
            output_units=1,
            dropout_rate=0.1
        )
        
        # BaseFeedForwardModel expects individual inputs for each feature
        feature_inputs = [keras.ops.expand_dims(self.x_features[:, i], axis=-1) for i in range(self.num_features)]
        
        # Test forward pass
        output = model(feature_inputs)
        
        # Test serialization
        config = model.get_config()
        self.assertIsInstance(config, dict)
        self.assertEqual(config['hidden_units'], [32, 16])
        self.assertEqual(config['output_units'], 1)
        self.assertEqual(config['dropout_rate'], 0.1)
        
        # Test deserialization
        reconstructed = BaseFeedForwardModel.from_config(config)
        reconstructed_output = reconstructed(feature_inputs)
        
        # Check that outputs match
        np.testing.assert_allclose(output, reconstructed_output, rtol=1e-5)


class TestModelSaving(unittest.TestCase):
    """Test saving and loading of KMR models."""

    def setUp(self):
        """Set up test data."""
        self.batch_size = 8
        self.num_features = 10
        
        # Create test data
        self.x = keras.random.normal((self.batch_size, self.num_features))

    def test_model_save_load_keras_format(self):
        """Test saving and loading model in Keras format."""
        # Create a model with KMR layers
        inputs = keras.Input(shape=(self.num_features,))
        embedded = AdvancedNumericalEmbedding(embed_dim=8, num_heads=2)(inputs)
        # Use a simple dense layer instead of GatedFeatureFusion
        dense = layers.Dense(16, activation='relu')(embedded)
        outputs = layers.Dense(1)(dense)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Test forward pass
        original_output = model(self.x)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as f:
            model_path = f.name
        
        try:
            model.save(model_path)
            
            # Load model
            loaded_model = keras.models.load_model(model_path)
            
            # Test loaded model
            loaded_output = loaded_model(self.x)
            
            # Check that outputs match
            np.testing.assert_allclose(original_output, loaded_output, rtol=1e-5)
            
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_model_save_load_tf_format(self):
        """Test saving and loading model in TensorFlow format."""
        # Create a model with KMR layers
        inputs = keras.Input(shape=(self.num_features,))
        embedded = AdvancedNumericalEmbedding(embed_dim=8, num_heads=2)(inputs)
        # Use a simple dense layer instead of GatedFeatureFusion
        dense = layers.Dense(16, activation='relu')(embedded)
        outputs = layers.Dense(1)(dense)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Test forward pass
        original_output = model(self.x)
        
        # Save model
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'model')
            model.save(model_path, save_format='tf')
            
            # Load model
            loaded_model = keras.models.load_model(model_path)
            
            # Test loaded model
            loaded_output = loaded_model(self.x)
            
            # Check that outputs match
            np.testing.assert_allclose(original_output, loaded_output, rtol=1e-5)

    def test_complex_model_save_load(self):
        """Test saving and loading a complex model with multiple KMR layers."""
        # Create complex model
        inputs = keras.Input(shape=(self.num_features,))
        
        # Stage 1: Embedding
        embedded = AdvancedNumericalEmbedding(embed_dim=8, num_heads=2)(inputs)
        
        # Stage 2: Transformer
        transformer = TransformerBlock(
            dim_model=8,
            num_heads=2,
            ff_units=16,
            dropout_rate=0.1
        )(embedded)
        
        # Stage 3: Stochastic depth
        residual = layers.Dense(8)(transformer)
        stochastic = StochasticDepth(survival_prob=0.8)([transformer, residual])
        
        # Stage 4: Output
        outputs = layers.Dense(1, activation='sigmoid')(stochastic)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Test forward pass
        original_output = model(self.x)
        
        # Save and load model
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as f:
            model_path = f.name
        
        try:
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)
            loaded_output = loaded_model(self.x)
            
            # Check that outputs match
            np.testing.assert_allclose(original_output, loaded_output, rtol=1e-5)
            
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.unlink(model_path)


if __name__ == '__main__':
    unittest.main()
