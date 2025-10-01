#!/usr/bin/env python
"""
Advanced usage examples for KMR layers and models.

This script demonstrates advanced usage patterns including model training,
serialization, and complex layer combinations.
"""

import keras
import numpy as np
import tempfile
import os
from kmr.layers import (
    TabularAttention,
    AdvancedNumericalEmbedding,
    GatedFeatureFusion,
    VariableSelection,
    TransformerBlock,
    BoostingBlock,
    StochasticDepth
)
from kmr.models import TerminatorModel, BaseFeedForwardModel
from kmr.utils import analyze_data

def example_model_training():
    """Example of training a KMR model."""
    print("=== Model Training Example ===")
    
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
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print(f"Model created with {model.count_params():,} parameters")
    print("Training for 5 epochs...")
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=16,
        validation_split=0.2,
        verbose=0
    )
    
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print()

def example_model_serialization():
    """Example of serializing and deserializing KMR models."""
    print("=== Model Serialization Example ===")
    
    # Create a model with KMR layers
    inputs = keras.Input(shape=(10,))
    embedding = AdvancedNumericalEmbedding(embed_dim=8, num_heads=2)(inputs)
    fusion = GatedFeatureFusion(units=16, dropout_rate=0.1)(embedding)
    outputs = keras.layers.Dense(1)(fusion)
    
    original_model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Save the model
    with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as f:
        model_path = f.name
    
    try:
        original_model.save(model_path)
        print(f"Model saved to: {model_path}")
        
        # Load the model
        loaded_model = keras.models.load_model(model_path)
        print("Model loaded successfully")
        
        # Test that both models produce the same output
        test_input = keras.random.normal((1, 10))
        original_output = original_model(test_input)
        loaded_output = loaded_model(test_input)
        
        print(f"Original model output shape: {original_output.shape}")
        print(f"Loaded model output shape: {loaded_output.shape}")
        print(f"Outputs match: {np.allclose(original_output, loaded_output)}")
        
    finally:
        # Clean up
        if os.path.exists(model_path):
            os.unlink(model_path)
    
    print()

def example_complex_layer_pipeline():
    """Example of a complex pipeline with multiple KMR layers."""
    print("=== Complex Layer Pipeline Example ===")
    
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
        dropout_rate=0.1
    )(fused)
    
    # Stage 4: Stochastic depth for regularization
    stochastic = StochasticDepth(dropout_rate=0.1)(transformer)
    
    # Stage 5: Final output
    outputs = keras.layers.Dense(1, activation='sigmoid')(stochastic)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    print(f"Pipeline model created with {model.count_params():,} parameters")
    print("Pipeline stages:")
    print("1. Advanced Numerical Embedding")
    print("2. Gated Feature Fusion")
    print("3. Transformer Block")
    print("4. Stochastic Depth")
    print("5. Dense Output")
    
    # Test the pipeline
    test_output = model(x)
    print(f"Pipeline output shape: {test_output.shape}")
    print()

def example_terminator_model_advanced():
    """Advanced example of using TerminatorModel."""
    print("=== Advanced TerminatorModel Example ===")
    
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
        slow_network_units=128
    )
    
    # Build the model
    model.build(input_shape=[(batch_size, input_dim), (batch_size, context_dim)])
    
    # Compile and train
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print(f"TerminatorModel created with {model.count_params():,} parameters")
    print(f"Number of SFNE blocks: {model.num_blocks}")
    print(f"Hidden dimension: {model.hidden_dim}")
    
    # Train the model
    print("Training TerminatorModel for 3 epochs...")
    history = model.fit(
        [x, context], y,
        epochs=3,
        batch_size=16,
        validation_split=0.2,
        verbose=0
    )
    
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print()

def example_boosting_block():
    """Example of using BoostingBlock for ensemble learning."""
    print("=== BoostingBlock Example ===")
    
    # Create sample data
    batch_size, num_features = 32, 12
    x = keras.random.normal((batch_size, num_features))
    
    # Create boosting block
    boosting_block = BoostingBlock(
        num_estimators=3,
        hidden_units=32,
        dropout_rate=0.1
    )
    
    # Apply boosting block
    output = boosting_block(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of estimators: {boosting_block.num_estimators}")
    print(f"Hidden units: {boosting_block.hidden_units}")
    print()

def example_variable_selection_advanced():
    """Advanced example of using VariableSelection with context."""
    print("=== Advanced VariableSelection Example ===")
    
    # Create sample data with context
    batch_size, num_features, context_dim = 16, 10, 5
    x = keras.random.normal((batch_size, num_features))
    context = keras.random.normal((batch_size, context_dim))
    
    # Create variable selection layer
    selector = VariableSelection(
        num_features=num_features,
        context_dim=context_dim,
        hidden_units=32
    )
    
    # Apply variable selection
    selected_features = selector([x, context])
    
    print(f"Input features shape: {x.shape}")
    print(f"Context shape: {context.shape}")
    print(f"Selected features shape: {selected_features.shape}")
    print(f"Number of input features: {selector.num_features}")
    print(f"Context dimension: {selector.context_dim}")
    print()

def example_custom_model_with_kmr():
    """Example of creating a custom model using KMR components."""
    print("=== Custom Model with KMR Components ===")
    
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
    dense1 = keras.layers.Dense(32, activation='relu')(fused)
    dropout1 = keras.layers.Dropout(0.2)(dense1)
    
    # More KMR layers
    transformer = TransformerBlock(
        embed_dim=32,
        num_heads=4,
        ff_dim=64,
        dropout_rate=0.1
    )(dropout1)
    
    # Final layers
    dense2 = keras.layers.Dense(16, activation='relu')(transformer)
    outputs = keras.layers.Dense(1)(dense2)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile and train
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print(f"Custom model created with {model.count_params():,} parameters")
    print("Model architecture:")
    print("- Advanced Numerical Embedding")
    print("- Gated Feature Fusion")
    print("- Dense + Dropout")
    print("- Transformer Block")
    print("- Dense + Output")
    
    # Train the model
    print("Training custom model for 3 epochs...")
    history = model.fit(
        x, y,
        epochs=3,
        batch_size=16,
        validation_split=0.2,
        verbose=0
    )
    
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print()

def main():
    """Run all advanced examples."""
    print("🚀 KMR Advanced Usage Examples 🚀\n")
    
    example_model_training()
    example_model_serialization()
    example_complex_layer_pipeline()
    example_terminator_model_advanced()
    example_boosting_block()
    example_variable_selection_advanced()
    example_custom_model_with_kmr()
    
    print("✅ All advanced examples completed successfully!")

if __name__ == "__main__":
    main()
