# 📖 Rich Docstrings Showcase

Comprehensive examples demonstrating KerasFactory layers with detailed documentation, best practices, and real-world use cases.

## 🎯 Overview

This showcase provides in-depth examples of KerasFactory layers with rich documentation, showing how to build production-ready tabular models. Each example includes:

- **Detailed explanations** of layer functionality
- **Best practices** for parameter selection
- **Real-world use cases** and applications
- **Performance considerations** and optimization tips
- **Complete code examples** ready to run

## 🧠 Attention Mechanisms

### TabularAttention - Dual Attention for Tabular Data

```python
import keras
import numpy as np
from loguru import logger
from typing import Tuple, Dict, Any, Optional
from kerasfactory.layers import TabularAttention

def create_tabular_attention_model(input_dim: int, num_classes: int) -> keras.Model:
    """Create a model using TabularAttention for dual attention mechanisms.
    
    TabularAttention implements both inter-feature and inter-sample attention,
    making it ideal for capturing complex relationships in tabular data.
    
    Args:
        input_dim: Number of input features.
        num_classes: Number of output classes.
    
    Returns:
        keras.Model: Compiled model ready for training.
        
    Example:
        ```python
        import keras
        model = create_tabular_attention_model(input_dim=20, num_classes=3)
        ```
    """
    
    # Input layer
    inputs = keras.Input(shape=(input_dim,), name='tabular_input')
    
    # TabularAttention layer with comprehensive configuration
    attention_layer = TabularAttention(
        num_heads=8,                    # 8 attention heads for rich representation
        key_dim=64,                     # 64-dimensional key vectors
        dropout=0.1,                    # 10% dropout for regularization
        use_attention_weights=True,     # Return attention weights for interpretation
        attention_activation='softmax', # Softmax activation for attention weights
        name='tabular_attention'
    )
    
    # Apply attention
    x = attention_layer(inputs)
    
    # Output layer
    outputs = keras.layers.Dense(
        num_classes, 
        activation='softmax',
        name='predictions'
    )(x)
    
    # Create and compile model
    model = keras.Model(inputs, outputs, name='tabular_attention_model')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Usage example
def demonstrate_tabular_attention() -> Tuple[keras.Model, keras.callbacks.History]:
    """Demonstrate TabularAttention with sample data.
    
    Creates and trains a TabularAttention model on random sample data,
    evaluating its performance and returning the trained model and history.
    
    Returns:
        Tuple[keras.Model, keras.callbacks.History]: Trained model and training history.
        
    Example:
        ```python
        model, history = demonstrate_tabular_attention()
        ```
    """
    
    # Create sample data
    X_train = np.random.random((1000, 20))
    y_train = np.random.randint(0, 3, (1000,))
    y_train = keras.utils.to_categorical(y_train, 3)
    
    # Create model
    model = create_tabular_attention_model(input_dim=20, num_classes=3)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_train, y_train, verbose=0)
    logger.info(f"Model accuracy: {test_accuracy:.4f}")
    
    return model, history

# Run demonstration
# model, history = demonstrate_tabular_attention()
```

### MultiResolutionTabularAttention - Multi-Resolution Processing

```python
from kerasfactory.layers import MultiResolutionTabularAttention

def create_multi_resolution_model(input_dim: int, num_classes: int) -> keras.Model:
    """Create a model using MultiResolutionTabularAttention for different feature scales.
    
    This layer processes numerical and categorical features separately with different
    attention mechanisms, then combines them with cross-attention.
    
    Args:
        input_dim: Number of input features.
        num_classes: Number of output classes.
    
    Returns:
        keras.Model: Compiled model ready for training.
        
    Example:
        ```python
        import keras
        model = create_multi_resolution_model(input_dim=20, num_classes=3)
        ```
    """
    
    inputs = keras.Input(shape=(input_dim,), name='multi_resolution_input')
    
    # Multi-resolution attention with separate processing
    attention_layer = MultiResolutionTabularAttention(
        num_heads=8,                    # Total attention heads
        key_dim=64,                     # Key dimension
        dropout=0.1,                    # Dropout rate
        numerical_heads=4,              # Heads for numerical features
        categorical_heads=4,            # Heads for categorical features
        name='multi_resolution_attention'
    )
    
    # Apply multi-resolution attention
    x = attention_layer(inputs)
    
    # Output layer
    outputs = keras.layers.Dense(
        num_classes, 
        activation='softmax',
        name='predictions'
    )(x)
    
    # Create and compile model
    model = keras.Model(inputs, outputs, name='multi_resolution_model')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Usage example
def demonstrate_multi_resolution() -> Tuple[keras.Model, keras.callbacks.History]:
    """Demonstrate MultiResolutionTabularAttention with mixed data types.
    
    Creates and trains a model that handles mixed numerical and categorical features
    using multi-resolution attention mechanisms.
    
    Returns:
        Tuple[keras.Model, keras.callbacks.History]: Trained model and training history.
        
    Example:
        ```python
        model, history = demonstrate_multi_resolution()
        ```
    """
    
    # Create sample data with mixed types
    X_numerical = np.random.random((1000, 10))
    X_categorical = np.random.randint(0, 5, (1000, 10))
    X_mixed = np.concatenate([X_numerical, X_categorical], axis=1)
    
    y = np.random.randint(0, 3, (1000,))
    y = keras.utils.to_categorical(y, 3)
    
    # Create model
    model = create_multi_resolution_model(input_dim=20, num_classes=3)
    
    # Train model
    history = model.fit(
        X_mixed, y,
        validation_split=0.2,
        epochs=10,
        batch_size=32,
        verbose=1
    )
    
    return model, history

# Run demonstration
# model, history = demonstrate_multi_resolution()
```

## 🔧 Feature Engineering

### VariableSelection - Intelligent Feature Selection

```python
from kerasfactory.layers import VariableSelection

def create_variable_selection_model(input_dim: int, num_classes: int) -> keras.Model:
    """Create a model using VariableSelection for intelligent feature selection.
    
    VariableSelection uses gated residual networks to learn feature importance
    and select the most relevant features for the task.
    
    Args:
        input_dim: Number of input features.
        num_classes: Number of output classes.
    
    Returns:
        keras.Model: Compiled model ready for training.
        
    Example:
        ```python
        import keras
        model = create_variable_selection_model(input_dim=20, num_classes=3)
        ```
    """
    
    inputs = keras.Input(shape=(input_dim,), name='variable_selection_input')
    
    # Variable selection with context
    selection_layer = VariableSelection(
        hidden_dim=64,                  # Hidden dimension for GRN
        dropout=0.1,                    # Dropout rate
        use_context=True,               # Use context for selection
        context_dim=32,                 # Context dimension
        name='variable_selection'
    )
    
    # Apply variable selection
    x = selection_layer(inputs)
    
    # Additional processing
    x = keras.layers.Dense(128, activation='relu', name='dense_1')(x)
    x = keras.layers.Dropout(0.2, name='dropout_1')(x)
    x = keras.layers.Dense(64, activation='relu', name='dense_2')(x)
    
    # Output layer
    outputs = keras.layers.Dense(
        num_classes, 
        activation='softmax',
        name='predictions'
    )(x)
    
    # Create and compile model
    model = keras.Model(inputs, outputs, name='variable_selection_model')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Usage example
def demonstrate_variable_selection() -> Tuple[keras.Model, keras.callbacks.History]:
    """Demonstrate VariableSelection with feature importance analysis.
    
    Trains a model with VariableSelection layer and analyzes learned feature importance weights.
    
    Returns:
        Tuple[keras.Model, keras.callbacks.History]: Trained model and training history.
        
    Example:
        ```python
        model, history = demonstrate_variable_selection()
        ```
    """
    
    # Create sample data
    X_train = np.random.random((1000, 20))
    y_train = np.random.randint(0, 3, (1000,))
    y_train = keras.utils.to_categorical(y_train, 3)
    
    # Create model
    model = create_variable_selection_model(input_dim=20, num_classes=3)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=32,
        verbose=1
    )
    
    # Analyze feature importance
    selection_layer = model.get_layer('variable_selection')
    feature_weights = selection_layer.get_weights()
    
    logger.info(f"Feature selection weights shape: {feature_weights[0].shape}")
    
    return model, history

# Run demonstration
# model, history = demonstrate_variable_selection()
```

### AdvancedNumericalEmbedding - Rich Numerical Representations

```python
from kerasfactory.layers import AdvancedNumericalEmbedding

def create_advanced_embedding_model(input_dim: int, num_classes: int) -> keras.Model:
    """Create a model using AdvancedNumericalEmbedding for rich numerical representations.
    
    This layer combines continuous MLP processing with discrete binning/embedding,
    providing a dual-branch architecture for numerical features.
    
    Args:
        input_dim: Number of input features.
        num_classes: Number of output classes.
    
    Returns:
        keras.Model: Compiled model ready for training.
        
    Example:
        ```python
        import keras
        model = create_advanced_embedding_model(input_dim=20, num_classes=3)
        ```
    """
    
    inputs = keras.Input(shape=(input_dim,), name='embedding_input')
    
    # Advanced numerical embedding
    embedding_layer = AdvancedNumericalEmbedding(
        embedding_dim=64,               # Embedding dimension
        num_bins=10,                    # Number of bins for discretization
        hidden_dim=128,                 # Hidden dimension for MLP
        dropout=0.1,                    # Dropout rate
        name='advanced_embedding'
    )
    
    # Apply embedding
    x = embedding_layer(inputs)
    
    # Additional processing
    x = keras.layers.Dense(128, activation='relu', name='dense_1')(x)
    x = keras.layers.Dropout(0.2, name='dropout_1')(x)
    x = keras.layers.Dense(64, activation='relu', name='dense_2')(x)
    
    # Output layer
    outputs = keras.layers.Dense(
        num_classes, 
        activation='softmax',
        name='predictions'
    )(x)
    
    # Create and compile model
    model = keras.Model(inputs, outputs, name='advanced_embedding_model')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Usage example
def demonstrate_advanced_embedding() -> Tuple[keras.Model, keras.callbacks.History]:
    """Demonstrate AdvancedNumericalEmbedding with numerical data.
    
    Trains a model using AdvancedNumericalEmbedding layer on numerical data
    with both continuous and binned representations.
    
    Returns:
        Tuple[keras.Model, keras.callbacks.History]: Trained model and training history.
        
    Example:
        ```python
        model, history = demonstrate_advanced_embedding()
        ```
    """
    
    # Create sample numerical data
    X_train = np.random.normal(0, 1, (1000, 20))
    y_train = np.random.randint(0, 3, (1000,))
    y_train = keras.utils.to_categorical(y_train, 3)
    
    # Create model
    model = create_advanced_embedding_model(input_dim=20, num_classes=3)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=32,
        verbose=1
    )
    
    return model, history

# Run demonstration
# model, history = demonstrate_advanced_embedding()
```

## ⚙️ Preprocessing

### DifferentiableTabularPreprocessor - End-to-End Preprocessing

```python
from kerasfactory.layers import DifferentiableTabularPreprocessor

def create_preprocessing_model(input_dim: int, num_classes: int) -> keras.Model:
    """Create a model using DifferentiableTabularPreprocessor for end-to-end preprocessing.
    
    This layer integrates preprocessing into the model, allowing for learnable
    imputation and normalization strategies.
    
    Args:
        input_dim: Number of input features.
        num_classes: Number of output classes.
    
    Returns:
        keras.Model: Compiled model ready for training.
        
    Example:
        ```python
        import keras
        model = create_preprocessing_model(input_dim=20, num_classes=3)
        ```
    """
    
    inputs = keras.Input(shape=(input_dim,), name='preprocessing_input')
    
    # Differentiable preprocessing
    preprocessor = DifferentiableTabularPreprocessor(
        imputation_strategy='learnable',    # Learnable imputation
        normalization='learnable',          # Learnable normalization
        dropout=0.1,                        # Dropout rate
        name='tabular_preprocessor'
    )
    
    # Apply preprocessing
    x = preprocessor(inputs)
    
    # Additional processing
    x = keras.layers.Dense(128, activation='relu', name='dense_1')(x)
    x = keras.layers.Dropout(0.2, name='dropout_1')(x)
    x = keras.layers.Dense(64, activation='relu', name='dense_2')(x)
    
    # Output layer
    outputs = keras.layers.Dense(
        num_classes, 
        activation='softmax',
        name='predictions'
    )(x)
    
    # Create and compile model
    model = keras.Model(inputs, outputs, name='preprocessing_model')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Usage example
def demonstrate_preprocessing() -> Tuple[keras.Model, keras.callbacks.History]:
    """Demonstrate DifferentiableTabularPreprocessor with missing data.
    
    Creates and trains a model that handles missing values using learnable
    preprocessing strategies.
    
    Returns:
        Tuple[keras.Model, keras.callbacks.History]: Trained model and training history.
        
    Example:
        ```python
        model, history = demonstrate_preprocessing()
        ```
    """
    
    # Create sample data with missing values
    X_train = np.random.random((1000, 20))
    # Introduce missing values
    missing_mask = np.random.random((1000, 20)) < 0.1
    X_train[missing_mask] = np.nan
    
    y_train = np.random.randint(0, 3, (1000,))
    y_train = keras.utils.to_categorical(y_train, 3)
    
    # Create model
    model = create_preprocessing_model(input_dim=20, num_classes=3)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=32,
        verbose=1
    )
    
    return model, history

# Run demonstration
# model, history = demonstrate_preprocessing()
```

## 🏗️ Specialized Architectures

### GatedResidualNetwork - Advanced Residual Processing

```python
from kerasfactory.layers import GatedResidualNetwork

def create_gated_residual_model(input_dim: int, num_classes: int) -> keras.Model:
    """Create a model using GatedResidualNetwork for advanced residual processing.
    
    This layer combines residual connections with gated linear units for
    improved gradient flow and feature transformation.
    
    Args:
        input_dim: Number of input features.
        num_classes: Number of output classes.
    
    Returns:
        keras.Model: Compiled model ready for training.
        
    Example:
        ```python
        import keras
        model = create_gated_residual_model(input_dim=20, num_classes=3)
        ```
    """
    
    inputs = keras.Input(shape=(input_dim,), name='gated_residual_input')
    
    # Gated residual networks
    x = GatedResidualNetwork(
        units=64,                        # Number of units
        dropout_rate=0.1,                # Dropout rate
        name='grn_1'
    )(inputs)
    
    x = GatedResidualNetwork(
        units=64,
        dropout_rate=0.1,
        name='grn_2'
    )(x)
    
    x = GatedResidualNetwork(
        units=64,
        dropout_rate=0.1,
        name='grn_3'
    )(x)
    
    # Output layer
    outputs = keras.layers.Dense(
        num_classes, 
        activation='softmax',
        name='predictions'
    )(x)
    
    # Create and compile model
    model = keras.Model(inputs, outputs, name='gated_residual_model')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Usage example
def demonstrate_gated_residual() -> Tuple[keras.Model, keras.callbacks.History]:
    """Demonstrate GatedResidualNetwork with deep architecture.
    
    Creates and trains a deep residual network using GatedResidualNetwork layers
    for improved gradient flow and feature transformation.
    
    Returns:
        Tuple[keras.Model, keras.callbacks.History]: Trained model and training history.
        
    Example:
        ```python
        model, history = demonstrate_gated_residual()
        ```
    """
    
    # Create sample data
    X_train = np.random.random((1000, 20))
    y_train = np.random.randint(0, 3, (1000,))
    y_train = keras.utils.to_categorical(y_train, 3)
    
    # Create model
    model = create_gated_residual_model(input_dim=20, num_classes=3)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=32,
        verbose=1
    )
    
    return model, history

# Run demonstration
# model, history = demonstrate_gated_residual()
```

### TabularMoELayer - Mixture of Experts

```python
from kerasfactory.layers import TabularMoELayer

def create_moe_model(input_dim: int, num_classes: int) -> keras.Model:
    """Create a model using TabularMoELayer for mixture of experts architecture.
    
    This layer routes input features through multiple expert sub-networks
    and aggregates their outputs via a learnable gating mechanism.
    
    Args:
        input_dim: Number of input features.
        num_classes: Number of output classes.
    
    Returns:
        keras.Model: Compiled model ready for training.
        
    Example:
        ```python
        import keras
        model = create_moe_model(input_dim=20, num_classes=3)
        ```
    """
    
    inputs = keras.Input(shape=(input_dim,), name='moe_input')
    
    # Mixture of experts
    moe_layer = TabularMoELayer(
        num_experts=4,                   # Number of expert networks
        expert_units=16,                 # Units per expert
        name='tabular_moe'
    )
    
    # Apply MoE
    x = moe_layer(inputs)
    
    # Additional processing
    x = keras.layers.Dense(128, activation='relu', name='dense_1')(x)
    x = keras.layers.Dropout(0.2, name='dropout_1')(x)
    x = keras.layers.Dense(64, activation='relu', name='dense_2')(x)
    
    # Output layer
    outputs = keras.layers.Dense(
        num_classes, 
        activation='softmax',
        name='predictions'
    )(x)
    
    # Create and compile model
    model = keras.Model(inputs, outputs, name='moe_model')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Usage example
def demonstrate_moe() -> Tuple[keras.Model, keras.callbacks.History]:
    """Demonstrate TabularMoELayer with expert routing.
    
    Creates and trains a mixture of experts model where multiple expert networks
    process input features and are combined via learned gating.
    
    Returns:
        Tuple[keras.Model, keras.callbacks.History]: Trained model and training history.
        
    Example:
        ```python
        model, history = demonstrate_moe()
        ```
    """
    
    # Create sample data
    X_train = np.random.random((1000, 20))
    y_train = np.random.randint(0, 3, (1000,))
    y_train = keras.utils.to_categorical(y_train, 3)
    
    # Create model
    model = create_moe_model(input_dim=20, num_classes=3)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=32,
        verbose=1
    )
    
    return model, history

# Run demonstration
# model, history = demonstrate_moe()
```

## 🔍 Model Interpretation and Analysis

### Attention Weight Analysis

```python
def analyze_attention_weights(model: keras.Model, X_test: np.ndarray, layer_name: str = 'tabular_attention') -> Dict[str, Any]:
    """Analyze attention weights to understand model behavior.
    
    Extracts attention weights from a specified attention layer and computes
    statistical measures including mean, standard deviation, and feature importance.
    
    Args:
        model: Trained model with attention layer.
        X_test: Test feature array of shape (n_samples, n_features).
        layer_name: Name of the attention layer to analyze. Defaults to 'tabular_attention'.
    
    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'attention_weights': Raw attention weight matrices
            - 'mean_attention': Mean attention weights
            - 'std_attention': Standard deviation of attention weights
            - 'feature_importance': Computed feature importance scores
        
    Example:
        ```python
        import numpy as np
        X_test = np.random.rand(100, 20)
        analysis = analyze_attention_weights(model, X_test)
        ```
    """
    
    # Get attention layer
    attention_layer = model.get_layer(layer_name)
    
    # Create model that outputs attention weights
    attention_model = keras.Model(
        inputs=model.input,
        outputs=attention_layer.output
    )
    
    # Get attention weights
    attention_weights = attention_model.predict(X_test)
    
    # Analyze attention patterns
    mean_attention = np.mean(attention_weights, axis=0)
    std_attention = np.std(attention_weights, axis=0)
    
    # Feature importance
    feature_importance = np.mean(attention_weights, axis=(0, 1))
    
    analysis = {
        'attention_weights': attention_weights,
        'mean_attention': mean_attention,
        'std_attention': std_attention,
        'feature_importance': feature_importance
    }
    
    return analysis

# Usage example
def demonstrate_attention_analysis() -> Dict[str, Any]:
    """Demonstrate attention weight analysis.
    
    Analyzes attention weights from a trained model and logs feature importance scores.
    
    Returns:
        Dict[str, Any]: Analysis results containing attention weights and importance scores.
        
    Example:
        ```python
        analysis = demonstrate_attention_analysis()
        ```
    """
    
    # Create sample data
    X_test = np.random.random((100, 20))
    
    # Analyze attention weights
    analysis = analyze_attention_weights(model, X_test)
    
    logger.info("Feature importance scores:")
    for i, importance in enumerate(analysis['feature_importance']):
        logger.info(f"Feature {i}: {importance:.4f}")
    
    return analysis

# Run demonstration
# analysis = demonstrate_attention_analysis()
```

## 📊 Performance Optimization

### Memory-Efficient Training

```python
from kerasfactory.layers import GatedFeatureFusion

def create_memory_efficient_model(input_dim: int, num_classes: int) -> keras.Model:
    """Create a memory-efficient model for large datasets.
    
    This model uses smaller dimensions and fewer parameters to reduce
    memory usage while maintaining good performance. Ideal for deployment
    on memory-constrained devices.
    
    Args:
        input_dim: Number of input features.
        num_classes: Number of output classes.
    
    Returns:
        keras.Model: Compiled memory-efficient model.
        
    Example:
        ```python
        import keras
        model = create_memory_efficient_model(input_dim=50, num_classes=3)
        ```
    """
    
    inputs = keras.Input(shape=(input_dim,), name='memory_efficient_input')
    
    # Use smaller dimensions
    x = VariableSelection(hidden_dim=32)(inputs)
    x = TabularAttention(num_heads=4, key_dim=32)(x)
    x = GatedFeatureFusion(hidden_dim=64)(x)
    
    # Output layer
    outputs = keras.layers.Dense(
        num_classes, 
        activation='softmax',
        name='predictions'
    )(x)
    
    # Create and compile model
    model = keras.Model(inputs, outputs, name='memory_efficient_model')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Usage example
def demonstrate_memory_efficiency() -> Tuple[keras.Model, keras.callbacks.History]:
    """Demonstrate memory-efficient training on large datasets.
    
    Creates and trains a memory-efficient model on a large dataset,
    demonstrating reduced memory consumption while maintaining performance.
    
    Returns:
        Tuple[keras.Model, keras.callbacks.History]: Trained model and training history.
        
    Example:
        ```python
        model, history = demonstrate_memory_efficiency()
        ```
    """
    
    # Create large dataset
    X_train = np.random.random((10000, 50))
    y_train = np.random.randint(0, 3, (10000,))
    y_train = keras.utils.to_categorical(y_train, 3)
    
    logger.info("Starting memory-efficient training...")
    
    # Create memory-efficient model
    model = create_memory_efficient_model(input_dim=50, num_classes=3)
    
    # Train with smaller batch size
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=16,  # Smaller batch size
        verbose=1
    )
    
    logger.info("Memory-efficient training completed.")
    
    return model, history

# Run demonstration
# model, history = demonstrate_memory_efficiency()
```

## 📚 Next Steps

1. **BaseFeedForwardModel Guide**: Learn about feed-forward architectures
2. **KDP Integration Guide**: Integrate with Keras Data Processor
3. **Data Analyzer Examples**: Explore data analysis workflows
4. **API Reference**: Deep dive into layer parameters

---

**Ready for more examples?** Check out the [BaseFeedForwardModel Guide](feed_forward_guide.md) next!