#!/usr/bin/env python
"""Basic usage examples for KMR layers and models.

This script demonstrates fundamental usage patterns for the most commonly used
KMR components including layers, models, and utilities.
"""

import keras
import numpy as np
from kmr.layers import (
    TabularAttention,
    AdvancedNumericalEmbedding,
    GatedFeatureFusion,
    VariableSelection,
)
from kmr.models import BaseFeedForwardModel, TerminatorModel
from kmr.utils import analyze_data


def example_tabular_attention() -> None:
    """Example of using TabularAttention layer."""
    # Create sample tabular data
    batch_size, num_samples, num_features = 32, 100, 20
    x = keras.random.normal((batch_size, num_samples, num_features))

    # Create and apply tabular attention
    attention = TabularAttention(num_heads=4, d_model=32, dropout_rate=0.1)
    attention(x)


def example_advanced_numerical_embedding() -> None:
    """Example of using AdvancedNumericalEmbedding layer."""
    # Create sample numerical data
    batch_size, num_features = 64, 15
    x = keras.random.normal((batch_size, num_features))

    # Create and apply advanced numerical embedding
    embedding = AdvancedNumericalEmbedding(embed_dim=16, num_heads=4)
    embedding(x)


def example_gated_feature_fusion() -> None:
    """Example of using GatedFeatureFusion layer."""
    # Create sample feature data
    batch_size, num_features = 32, 10
    x = keras.random.normal((batch_size, num_features))

    # Create and apply gated feature fusion
    fusion = GatedFeatureFusion(units=64, dropout_rate=0.1)
    fusion(x)


def example_variable_selection() -> None:
    """Example of using VariableSelection layer."""
    # Create sample data with context
    batch_size, num_features = 16, 8
    x = keras.random.normal((batch_size, num_features))
    context = keras.random.normal((batch_size, 4))

    # Create and apply variable selection
    selector = VariableSelection(num_features=num_features, context_dim=4)
    selector([x, context])


def example_feed_forward_model() -> None:
    """Example of using BaseFeedForwardModel."""
    # Create sample data
    batch_size, num_features = 32, 10
    keras.random.normal((batch_size, num_features))
    keras.random.normal((batch_size, 1))

    # Create model
    model = BaseFeedForwardModel(
        feature_names=[f"feature_{i}" for i in range(num_features)],
        hidden_units=[64, 32],
        output_units=1,
        dropout_rate=0.1,
    )

    # Build the model
    model.build(input_shape=(batch_size, num_features))


def example_terminator_model() -> None:
    """Example of using TerminatorModel."""
    # Create sample data
    batch_size, input_dim, context_dim = 16, 12, 6
    keras.random.normal((batch_size, input_dim))
    keras.random.normal((batch_size, context_dim))

    # Create model
    model = TerminatorModel(
        input_dim=input_dim,
        context_dim=context_dim,
        output_dim=1,
        num_blocks=2,
        hidden_dim=32,
    )

    # Build the model
    model.build(input_shape=[(batch_size, input_dim), (batch_size, context_dim)])


def example_layer_combination() -> None:
    """Example of combining multiple KMR layers."""
    # Create sample data
    batch_size, num_features = 32, 15
    x = keras.random.normal((batch_size, num_features))

    # Create a pipeline of layers
    embedding = AdvancedNumericalEmbedding(embed_dim=16, num_heads=4)
    fusion = GatedFeatureFusion(units=32, dropout_rate=0.1)

    # Apply layers sequentially
    embedded = embedding(x)
    fusion(embedded)


def example_data_analyzer() -> None:
    """Example of using the data analyzer utility."""
    # Create a sample CSV file for demonstration
    import pandas as pd
    import tempfile
    import os

    # Generate sample data
    np.random.seed(42)
    data = {
        "numeric_1": np.random.normal(0, 1, 100),
        "numeric_2": np.random.normal(10, 5, 100),
        "categorical_1": np.random.choice(["A", "B", "C"], 100),
        "categorical_2": np.random.choice(["X", "Y"], 100),
        "date_1": pd.date_range("2023-01-01", periods=100, freq="D"),
        "target": np.random.normal(0, 1, 100),
    }

    df = pd.DataFrame(data)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_file = f.name

    try:
        # Analyze the data
        results = analyze_data(temp_file)

        for _rec in results["recommendations"][:5]:  # Show first 5 recommendations
            pass
    finally:
        # Clean up
        os.unlink(temp_file)


def main() -> None:
    """Run all examples."""
    example_tabular_attention()
    example_advanced_numerical_embedding()
    example_gated_feature_fusion()
    example_variable_selection()
    example_feed_forward_model()
    example_terminator_model()
    example_layer_combination()
    example_data_analyzer()


if __name__ == "__main__":
    main()
