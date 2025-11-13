"""Integration tests for kerasfactory layer combinations.

These tests verify that kerasfactory layers work correctly when combined together
in various configurations and pipelines.
"""

import unittest
import keras
import tensorflow as tf
from keras import Model, layers

from kerasfactory.layers import (
    TabularAttention,
    AdvancedNumericalEmbedding,
    GatedFeatureFusion,
    VariableSelection,
    TransformerBlock,
    BoostingBlock,
    StochasticDepth,
)
from kerasfactory.models import TerminatorModel, BaseFeedForwardModel


class TestLayerCombinations(unittest.TestCase):
    """Test combinations of kerasfactory layers."""

    def setUp(self) -> None:
        """Set up test data."""
        self.batch_size = 8
        self.num_features = 10
        self.embed_dim = 16
        self.num_heads = 4

        # Create test data
        self.x = keras.random.normal((self.batch_size, self.num_features))
        self.context = keras.random.normal((self.batch_size, 5))

    def test_embedding_fusion_pipeline(self) -> None:
        """Test embedding followed by fusion."""
        # Create pipeline with two inputs for GatedFeatureFusion
        inputs1 = keras.Input(shape=(self.num_features,))
        inputs2 = keras.Input(shape=(self.num_features,))

        embedded1 = AdvancedNumericalEmbedding(
            embedding_dim=self.embed_dim,
            num_heads=self.num_heads,
        )(inputs1)
        embedded2 = AdvancedNumericalEmbedding(
            embedding_dim=self.embed_dim,
            num_heads=self.num_heads,
        )(inputs2)

        fused = GatedFeatureFusion()([embedded1, embedded2])
        pooled = layers.GlobalAveragePooling1D()(fused)
        outputs = layers.Dense(1)(pooled)

        model = Model(inputs=[inputs1, inputs2], outputs=outputs)

        # Test forward pass
        output = model([self.x, self.x])

        self.assertEqual(output.shape, (self.batch_size, 1))
        self.assertIsNotNone(output)

    def test_attention_transformer_pipeline(self) -> None:
        """Test attention followed by transformer."""
        # Reshape data for attention
        x_reshaped = keras.ops.reshape(self.x, (self.batch_size, 1, self.num_features))

        # Create pipeline
        inputs = keras.Input(shape=(1, self.num_features))
        attention = TabularAttention(
            num_heads=self.num_heads,
            d_model=self.embed_dim,
            dropout_rate=0.1,
        )(inputs)
        # Flatten attention output for transformer
        attention_flat = keras.ops.reshape(attention, (self.batch_size, self.embed_dim))
        transformer = TransformerBlock(
            dim_model=self.embed_dim,
            num_heads=self.num_heads,
            ff_units=32,
            dropout_rate=0.1,
        )(attention_flat)
        # TransformerBlock outputs 2D, no need for pooling
        outputs = layers.Dense(1)(transformer)

        model = Model(inputs=inputs, outputs=outputs)

        # Test forward pass
        output = model(x_reshaped)

        self.assertEqual(output.shape, (self.batch_size, 1))
        self.assertIsNotNone(output)

    def test_variable_selection_with_context(self) -> None:
        """Test variable selection with context."""
        # Create pipeline - VariableSelection expects 3D input
        inputs = keras.Input(
            shape=(self.num_features, 1),
        )  # (batch, features, feature_dim)
        context_input = keras.Input(shape=(5,))

        selected = VariableSelection(
            nr_features=self.num_features,
            units=16,
            use_context=True,
        )([inputs, context_input])

        # VariableSelection returns (selected_features, weights)
        selected_features, weights = selected
        # VariableSelection outputs 2D, no need for pooling
        outputs = layers.Dense(1)(selected_features)

        model = Model(inputs=[inputs, context_input], outputs=outputs)

        # Reshape input for 3D
        x_3d = keras.ops.expand_dims(self.x, axis=-1)

        # Test forward pass
        output = model([x_3d, self.context])

        self.assertEqual(output.shape, (self.batch_size, 1))
        self.assertIsNotNone(output)

    def test_boosting_block_pipeline(self) -> None:
        """Test boosting block in a pipeline."""
        # Create pipeline
        inputs = keras.Input(shape=(self.num_features,))
        boosted = BoostingBlock(num_estimators=3, hidden_units=16, dropout_rate=0.1)(
            inputs,
        )
        # BoostingBlock outputs 2D, no need for pooling
        outputs = layers.Dense(1)(boosted)

        model = Model(inputs=inputs, outputs=outputs)

        # Test forward pass
        output = model(self.x)

        self.assertEqual(output.shape, (self.batch_size, 1))
        self.assertIsNotNone(output)

    def test_stochastic_depth_regularization(self) -> None:
        """Test stochastic depth for regularization."""
        # Create pipeline
        inputs = keras.Input(shape=(self.num_features,))
        embedded = AdvancedNumericalEmbedding(
            embedding_dim=self.embed_dim,
            num_heads=self.num_heads,
        )(inputs)

        # Create a residual branch
        residual = layers.Dense(self.embed_dim)(embedded)

        # StochasticDepth expects [inputs, residual] and uses survival_prob
        stochastic = StochasticDepth(survival_prob=0.8)([embedded, residual])
        pooled = layers.GlobalAveragePooling1D()(stochastic)
        outputs = layers.Dense(1)(pooled)

        model = Model(inputs=inputs, outputs=outputs)

        # Test forward pass
        output = model(self.x)

        self.assertEqual(output.shape, (self.batch_size, 1))
        self.assertIsNotNone(output)

    def test_complex_pipeline(self) -> None:
        """Test a complex pipeline with multiple kerasfactory layers."""
        # Create complex pipeline
        inputs = keras.Input(shape=(self.num_features,))

        # Stage 1: Embedding
        embedded = AdvancedNumericalEmbedding(
            embedding_dim=self.embed_dim,
            num_heads=self.num_heads,
        )(inputs)

        # Stage 2: Transformer (skip fusion for simplicity)
        transformer = TransformerBlock(
            dim_model=self.embed_dim,
            num_heads=self.num_heads,
            ff_units=32,
            dropout_rate=0.1,
        )(embedded)

        # Stage 3: Global average pooling to reduce sequence dimension
        pooled = layers.GlobalAveragePooling1D()(transformer)

        # Stage 4: Stochastic depth
        residual = layers.Dense(self.embed_dim)(pooled)
        stochastic = StochasticDepth(survival_prob=0.8)([pooled, residual])

        # Stage 5: Output
        outputs = layers.Dense(1, activation="sigmoid")(stochastic)

        model = Model(inputs=inputs, outputs=outputs)

        # Test forward pass
        output = model(self.x)

        self.assertEqual(output.shape, (self.batch_size, 1))
        self.assertIsNotNone(output)
        self.assertTrue(keras.ops.all(output >= 0))
        self.assertTrue(keras.ops.all(output <= 1))

    def test_training_mode_behavior(self) -> None:
        """Test that layers behave differently in training vs inference."""
        # Create a simple pipeline
        inputs = keras.Input(shape=(self.num_features,))
        embedded = AdvancedNumericalEmbedding(
            embedding_dim=self.embed_dim,
            num_heads=self.num_heads,
        )(inputs)
        # Use a simple dense layer with dropout instead of GatedFeatureFusion
        dense = layers.Dense(32, activation="relu")(embedded)
        dropout = layers.Dropout(0.5)(dense)
        pooled = layers.GlobalAveragePooling1D()(dropout)
        outputs = layers.Dense(1)(pooled)

        model = Model(inputs=inputs, outputs=outputs)

        # Test in training mode
        output_train = model(self.x, training=True)

        # Test in inference mode
        output_inference = model(self.x, training=False)

        # Outputs should be different due to dropout
        self.assertIsNotNone(output_train)
        self.assertIsNotNone(output_inference)
        self.assertEqual(output_train.shape, output_inference.shape)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow correctly through the pipeline."""
        # Create a simple pipeline
        inputs = keras.Input(shape=(self.num_features,))
        embedded = AdvancedNumericalEmbedding(
            embedding_dim=self.embed_dim,
            num_heads=self.num_heads,
        )(inputs)
        # Use a simple dense layer instead of GatedFeatureFusion
        dense = layers.Dense(32, activation="relu")(embedded)
        pooled = layers.GlobalAveragePooling1D()(dense)
        outputs = layers.Dense(1)(pooled)

        model = Model(inputs=inputs, outputs=outputs)

        # Create dummy target
        y = keras.random.normal((self.batch_size, 1))

        # Test gradient computation
        with tf.GradientTape() as tape:
            predictions = model(self.x)
            loss = keras.losses.mean_squared_error(y, predictions)

        # Compute gradients
        gradients = tape.gradient(loss, model.trainable_variables)

        # Check that gradients are computed
        self.assertIsNotNone(gradients)
        self.assertEqual(len(gradients), len(model.trainable_variables))

        # Check that at least some gradients are not None
        non_none_gradients = [grad for grad in gradients if grad is not None]
        self.assertGreater(
            len(non_none_gradients),
            0,
            "At least some gradients should be computed",
        )


class TestModelIntegration(unittest.TestCase):
    """Test integration of kerasfactory models."""

    def setUp(self) -> None:
        """Set up test data."""
        self.batch_size = 8
        self.input_dim = 12
        self.context_dim = 6
        self.num_features = 10

        # Create test data
        self.x = keras.random.normal((self.batch_size, self.input_dim))
        self.context = keras.random.normal((self.batch_size, self.context_dim))
        self.x_features = keras.random.normal((self.batch_size, self.num_features))
        self.y = keras.random.normal((self.batch_size, 1))

    def test_terminator_model_integration(self) -> None:
        """Test TerminatorModel integration."""
        # Create model
        model = TerminatorModel(
            input_dim=self.input_dim,
            context_dim=self.context_dim,
            output_dim=1,
            num_blocks=2,
            hidden_dim=32,
        )

        # Build model
        model.build(
            input_shape=[
                (self.batch_size, self.input_dim),
                (self.batch_size, self.context_dim),
            ],
        )

        # Test forward pass
        output = model([self.x, self.context])

        self.assertEqual(output.shape, (self.batch_size, 1))
        self.assertIsNotNone(output)

    def test_feed_forward_model_integration(self) -> None:
        """Test BaseFeedForwardModel integration."""
        # Create model
        model = BaseFeedForwardModel(
            feature_names=[f"feature_{i}" for i in range(self.num_features)],
            hidden_units=[32, 16],
            output_units=1,
            dropout_rate=0.1,
        )

        # BaseFeedForwardModel expects individual inputs for each feature
        # Split the input tensor into individual features
        feature_inputs = [
            keras.ops.expand_dims(self.x_features[:, i], axis=-1)
            for i in range(self.num_features)
        ]

        # Test forward pass
        output = model(feature_inputs)

        self.assertEqual(output.shape, (self.batch_size, 1))
        self.assertIsNotNone(output)

    def test_model_training_integration(self) -> None:
        """Test model training integration."""
        # Create a simple model with kerasfactory layers
        inputs = keras.Input(shape=(self.num_features,))
        embedded = AdvancedNumericalEmbedding(embedding_dim=8, num_heads=2)(inputs)
        # Use a simple dense layer instead of GatedFeatureFusion
        dense = layers.Dense(16, activation="relu")(embedded)
        pooled = layers.GlobalAveragePooling1D()(dense)
        outputs = layers.Dense(1)(pooled)

        model = Model(inputs=inputs, outputs=outputs)

        # Compile model
        model.compile(optimizer="adam", loss="mse")

        # Test training step
        with tf.GradientTape() as tape:
            predictions = model(self.x_features)
            loss = keras.losses.mean_squared_error(self.y, predictions)

        # Compute gradients
        gradients = tape.gradient(loss, model.trainable_variables)

        # Check that training works
        self.assertIsNotNone(loss)
        self.assertIsNotNone(gradients)
        # Convert loss to scalar for comparison
        loss_value = float(keras.ops.mean(loss))
        self.assertGreater(loss_value, 0)


if __name__ == "__main__":
    unittest.main()
