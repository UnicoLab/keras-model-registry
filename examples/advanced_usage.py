#!/usr/bin/env python
"""Advanced usage examples for KMR layers and models.

This script demonstrates advanced usage patterns including model training,
serialization, and complex layer combinations.
"""

import keras
import numpy as np
import tempfile
import os
from kmr.layers import (
    AdvancedNumericalEmbedding,
    GatedFeatureFusion,
    VariableSelection,
    TransformerBlock,
    BoostingBlock,
    StochasticDepth,
)
from kmr.models import TerminatorModel


def example_model_training() -> None:
    """Example of training a KMR model."""
    # Generate synthetic training data
    np.random.seed(42)
    batch_size, num_features = 100, 20
    x_train = keras.random.normal((batch_size, num_features))
    y_train = keras.random.normal((batch_size, 1))

    # Create a simple model using KMR layers
    inputs = keras.Input(shape=(num_features,))

    # Apply KMR layers
    embedding = AdvancedNumericalEmbedding(embed_dim=16, num_heads=4)(inputs)
    fusion = GatedFeatureFusion(units=32, dropout_rate=0.1)(embedding)
    outputs = keras.layers.Dense(1)(fusion)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile and train
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # Train the model
    model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=16,
        validation_split=0.2,
        verbose=0,
    )


def example_model_serialization() -> None:
    """Example of serializing and deserializing KMR models."""
    # Create a model with KMR layers
    inputs = keras.Input(shape=(10,))
    embedding = AdvancedNumericalEmbedding(embed_dim=8, num_heads=2)(inputs)
    fusion = GatedFeatureFusion(units=16, dropout_rate=0.1)(embedding)
    outputs = keras.layers.Dense(1)(fusion)

    original_model = keras.Model(inputs=inputs, outputs=outputs)

    # Save the model
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as f:
        model_path = f.name

    try:
        original_model.save(model_path)

        # Load the model
        loaded_model = keras.models.load_model(model_path)

        # Test that both models produce the same output
        test_input = keras.random.normal((1, 10))
        original_model(test_input)
        loaded_model(test_input)

    finally:
        # Clean up
        if os.path.exists(model_path):
            os.unlink(model_path)


def example_complex_layer_pipeline() -> None:
    """Example of a complex pipeline with multiple KMR layers."""
    # Create sample data
    batch_size, num_features = 32, 15
    x = keras.random.normal((batch_size, num_features))

    # Create a complex pipeline
    inputs = keras.Input(shape=(num_features,))

    # Stage 1: Advanced numerical embedding
    embedded = AdvancedNumericalEmbedding(embed_dim=16, num_heads=4)(inputs)

    # Stage 2: Gated feature fusion
    fused = GatedFeatureFusion(units=32, dropout_rate=0.1)(embedded)

    # Stage 3: Transformer block
    transformer = TransformerBlock(
        embed_dim=32,
        num_heads=4,
        ff_dim=64,
        dropout_rate=0.1,
    )(fused)

    # Stage 4: Stochastic depth for regularization
    stochastic = StochasticDepth(dropout_rate=0.1)(transformer)

    # Stage 5: Final output
    outputs = keras.layers.Dense(1, activation="sigmoid")(stochastic)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Test the pipeline
    model(x)


def example_terminator_model_advanced() -> None:
    """Advanced example of using TerminatorModel."""
    # Create sample data with context
    batch_size, input_dim, context_dim = 64, 20, 8
    x = keras.random.normal((batch_size, input_dim))
    context = keras.random.normal((batch_size, context_dim))
    y = keras.random.normal((batch_size, 1))

    # Create TerminatorModel with custom configuration
    model = TerminatorModel(
        input_dim=input_dim,
        context_dim=context_dim,
        output_dim=1,
        num_blocks=3,
        hidden_dim=64,
        slow_network_layers=2,
        slow_network_units=128,
    )

    # Build the model
    model.build(input_shape=[(batch_size, input_dim), (batch_size, context_dim)])

    # Compile and train
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # Train the model
    model.fit(
        [x, context],
        y,
        epochs=3,
        batch_size=16,
        validation_split=0.2,
        verbose=0,
    )


def example_boosting_block() -> None:
    """Example of using BoostingBlock for ensemble learning."""
    # Create sample data
    batch_size, num_features = 32, 12
    x = keras.random.normal((batch_size, num_features))

    # Create boosting block
    boosting_block = BoostingBlock(num_estimators=3, hidden_units=32, dropout_rate=0.1)

    # Apply boosting block
    boosting_block(x)


def example_variable_selection_advanced() -> None:
    """Advanced example of using VariableSelection with context."""
    # Create sample data with context
    batch_size, num_features, context_dim = 16, 10, 5
    x = keras.random.normal((batch_size, num_features))
    context = keras.random.normal((batch_size, context_dim))

    # Create variable selection layer
    selector = VariableSelection(
        num_features=num_features,
        context_dim=context_dim,
        hidden_units=32,
    )

    # Apply variable selection
    selector([x, context])


def example_custom_model_with_kmr() -> None:
    """Example of creating a custom model using KMR components."""
    # Create sample data
    batch_size, num_features = 50, 18
    x = keras.random.normal((batch_size, num_features))
    y = keras.random.normal((batch_size, 1))

    # Create custom model architecture
    inputs = keras.Input(shape=(num_features,))

    # Feature preprocessing with KMR layers
    embedded = AdvancedNumericalEmbedding(embed_dim=12, num_heads=3)(inputs)
    fused = GatedFeatureFusion(units=24, dropout_rate=0.1)(embedded)

    # Add some standard Keras layers
    dense1 = keras.layers.Dense(32, activation="relu")(fused)
    dropout1 = keras.layers.Dropout(0.2)(dense1)

    # More KMR layers
    transformer = TransformerBlock(
        embed_dim=32,
        num_heads=4,
        ff_dim=64,
        dropout_rate=0.1,
    )(dropout1)

    # Final layers
    dense2 = keras.layers.Dense(16, activation="relu")(transformer)
    outputs = keras.layers.Dense(1)(dense2)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile and train
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # Train the model
    model.fit(x, y, epochs=3, batch_size=16, validation_split=0.2, verbose=0)


def main() -> None:
    """Run all advanced examples."""
    example_model_training()
    example_model_serialization()
    example_complex_layer_pipeline()
    example_terminator_model_advanced()
    example_boosting_block()
    example_variable_selection_advanced()
    example_custom_model_with_kmr()


if __name__ == "__main__":
    main()
