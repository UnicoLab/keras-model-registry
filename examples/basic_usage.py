#!/usr/bin/env python
"""
Basic usage examples for KMR layers and models.

This script demonstrates fundamental usage patterns for the most commonly used
KMR components including layers, models, and utilities.
"""

import keras
import numpy as np
from kmr.layers import (
    TabularAttention,
    AdvancedNumericalEmbedding,
    GatedFeatureFusion,
    DateEncodingLayer,
    VariableSelection
)
from kmr.models import BaseFeedForwardModel, TerminatorModel
from kmr.utils import analyze_data

def example_tabular_attention():
    """Example of using TabularAttention layer."""
    print("=== TabularAttention Example ===")
    
    # Create sample tabular data
    batch_size, num_samples, num_features = 32, 100, 20
    x = keras.random.normal((batch_size, num_samples, num_features))
    
    # Create and apply tabular attention
    attention = TabularAttention(num_heads=4, d_model=32, dropout_rate=0.1)
    output = attention(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention heads: {attention.num_heads}")
    print(f"Model dimension: {attention.d_model}")
    print()

def example_advanced_numerical_embedding():
    """Example of using AdvancedNumericalEmbedding layer."""
    print("=== AdvancedNumericalEmbedding Example ===")
    
    # Create sample numerical data
    batch_size, num_features = 64, 15
    x = keras.random.normal((batch_size, num_features))
    
    # Create and apply advanced numerical embedding
    embedding = AdvancedNumericalEmbedding(embed_dim=16, num_heads=4)
    output = embedding(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Embedding dimension: {embedding.embed_dim}")
    print(f"Number of heads: {embedding.num_heads}")
    print()

def example_gated_feature_fusion():
    """Example of using GatedFeatureFusion layer."""
    print("=== GatedFeatureFusion Example ===")
    
    # Create sample feature data
    batch_size, num_features = 32, 10
    x = keras.random.normal((batch_size, num_features))
    
    # Create and apply gated feature fusion
    fusion = GatedFeatureFusion(units=64, dropout_rate=0.1)
    output = fusion(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Fusion units: {fusion.units}")
    print(f"Dropout rate: {fusion.dropout_rate}")
    print()

def example_variable_selection():
    """Example of using VariableSelection layer."""
    print("=== VariableSelection Example ===")
    
    # Create sample data with context
    batch_size, num_features = 16, 8
    x = keras.random.normal((batch_size, num_features))
    context = keras.random.normal((batch_size, 4))
    
    # Create and apply variable selection
    selector = VariableSelection(num_features=num_features, context_dim=4)
    output = selector([x, context])
    
    print(f"Input shape: {x.shape}")
    print(f"Context shape: {context.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of features: {selector.num_features}")
    print(f"Context dimension: {selector.context_dim}")
    print()

def example_feed_forward_model():
    """Example of using BaseFeedForwardModel."""
    print("=== BaseFeedForwardModel Example ===")
    
    # Create sample data
    batch_size, num_features = 32, 10
    x = keras.random.normal((batch_size, num_features))
    y = keras.random.normal((batch_size, 1))
    
    # Create model
    model = BaseFeedForwardModel(
        feature_names=[f'feature_{i}' for i in range(num_features)],
        hidden_units=[64, 32],
        output_units=1,
        dropout_rate=0.1
    )
    
    # Build the model
    model.build(input_shape=(batch_size, num_features))
    
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    print(f"Hidden units: {model.hidden_units}")
    print(f"Number of parameters: {model.count_params():,}")
    print()

def example_terminator_model():
    """Example of using TerminatorModel."""
    print("=== TerminatorModel Example ===")
    
    # Create sample data
    batch_size, input_dim, context_dim = 16, 12, 6
    x = keras.random.normal((batch_size, input_dim))
    context = keras.random.normal((batch_size, context_dim))
    
    # Create model
    model = TerminatorModel(
        input_dim=input_dim,
        context_dim=context_dim,
        output_dim=1,
        num_blocks=2,
        hidden_dim=32
    )
    
    # Build the model
    model.build(input_shape=[(batch_size, input_dim), (batch_size, context_dim)])
    
    print(f"Input dimension: {input_dim}")
    print(f"Context dimension: {context_dim}")
    print(f"Output dimension: {model.output_dim}")
    print(f"Number of blocks: {model.num_blocks}")
    print(f"Number of parameters: {model.count_params():,}")
    print()

def example_layer_combination():
    """Example of combining multiple KMR layers."""
    print("=== Layer Combination Example ===")
    
    # Create sample data
    batch_size, num_features = 32, 15
    x = keras.random.normal((batch_size, num_features))
    
    # Create a pipeline of layers
    embedding = AdvancedNumericalEmbedding(embed_dim=16, num_heads=4)
    fusion = GatedFeatureFusion(units=32, dropout_rate=0.1)
    
    # Apply layers sequentially
    embedded = embedding(x)
    fused = fusion(embedded)
    
    print(f"Original shape: {x.shape}")
    print(f"After embedding: {embedded.shape}")
    print(f"After fusion: {fused.shape}")
    print()

def example_data_analyzer():
    """Example of using the data analyzer utility."""
    print("=== Data Analyzer Example ===")
    
    # Create a sample CSV file for demonstration
    import pandas as pd
    import tempfile
    import os
    
    # Generate sample data
    np.random.seed(42)
    data = {
        'numeric_1': np.random.normal(0, 1, 100),
        'numeric_2': np.random.normal(10, 5, 100),
        'categorical_1': np.random.choice(['A', 'B', 'C'], 100),
        'categorical_2': np.random.choice(['X', 'Y'], 100),
        'date_1': pd.date_range('2023-01-01', periods=100, freq='D'),
        'target': np.random.normal(0, 1, 100)
    }
    
    df = pd.DataFrame(data)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_file = f.name
    
    try:
        # Analyze the data
        results = analyze_data(temp_file)
        
        print("Data Analysis Results:")
        print(f"Number of rows: {results['statistics']['num_rows']}")
        print(f"Number of columns: {results['statistics']['num_columns']}")
        print(f"Feature types: {results['statistics']['feature_types']}")
        print("\nRecommended layers:")
        for rec in results['recommendations'][:5]:  # Show first 5 recommendations
            print(f"- {rec['layer_name']}: {rec['description']}")
    finally:
        # Clean up
        os.unlink(temp_file)
    
    print()

def main():
    """Run all examples."""
    print("🌟 KMR Basic Usage Examples 🌟\n")
    
    example_tabular_attention()
    example_advanced_numerical_embedding()
    example_gated_feature_fusion()
    example_variable_selection()
    example_feed_forward_model()
    example_terminator_model()
    example_layer_combination()
    example_data_analyzer()
    
    print("✅ All examples completed successfully!")

if __name__ == "__main__":
    main()
