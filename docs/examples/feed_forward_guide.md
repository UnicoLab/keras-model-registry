# 🏗️ BaseFeedForwardModel Guide

Learn how to build feed-forward models using KerasFactory layers. This guide covers the fundamentals of creating efficient feed-forward architectures for tabular data.

## 📋 Table of Contents

1. [Basic Feed-Forward Architecture](#basic-feed-forward-architecture)
2. [Advanced Feed-Forward Patterns](#advanced-feed-forward-patterns)
3. [Performance Optimization](#performance-optimization)
4. [Real-World Examples](#real-world-examples)

## 🏛️ Basic Feed-Forward Architecture

### Simple Feed-Forward Model

```python
import keras
import numpy as np
from loguru import logger
from typing import Optional, Tuple
from kerasfactory.layers import VariableSelection, GatedFeatureFusion

def create_basic_feedforward(input_dim: int, num_classes: int) -> keras.Model:
    """Create a basic feed-forward model with KerasFactory layers.
    
    Constructs a simple feed-forward neural network using VariableSelection layer
    for feature selection followed by dense layers for classification.
    
    Args:
        input_dim: Dimension of input features.
        num_classes: Number of output classes for classification.
    
    Returns:
        keras.Model: Compiled feed-forward model.
        
    Example:
        ```python
        import keras
        model = create_basic_feedforward(input_dim=20, num_classes=3)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        ```
    """
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Feature selection
    x = VariableSelection(hidden_dim=64)(inputs)
    
    # Dense layers
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    
    # Output
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

# Usage
# model = create_basic_feedforward(input_dim=20, num_classes=3)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### Feed-Forward with Feature Engineering

```python
from kerasfactory.layers import (
    DifferentiableTabularPreprocessor,
    AdvancedNumericalEmbedding,
    GatedFeatureFusion
)

def create_engineered_feedforward(input_dim: int, num_classes: int) -> keras.Model:
    """Create a feed-forward model with feature engineering.
    
    Builds a feed-forward network that includes preprocessing, feature embedding,
    and gated feature fusion for improved feature interactions.
    
    Args:
        input_dim: Dimension of input features.
        num_classes: Number of output classes for classification.
    
    Returns:
        keras.Model: Feed-forward model with feature engineering layers.
        
    Example:
        ```python
        import keras
        model = create_engineered_feedforward(input_dim=20, num_classes=3)
        ```
    """
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Preprocessing
    x = DifferentiableTabularPreprocessor()(inputs)
    
    # Feature engineering
    x = AdvancedNumericalEmbedding(embedding_dim=64)(x)
    x = GatedFeatureFusion(hidden_dim=128)(x)
    
    # Dense layers
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    
    # Output
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)
```

## 🚀 Advanced Feed-Forward Patterns

### Residual Feed-Forward

```python
from kerasfactory.layers import GatedResidualNetwork

def create_residual_feedforward(input_dim: int, num_classes: int) -> keras.Model:
    """Create a residual feed-forward model.
    
    Constructs a feed-forward network using stacked GatedResidualNetwork layers
    to enable deeper architectures with improved gradient flow through residual connections.
    
    Args:
        input_dim: Dimension of input features.
        num_classes: Number of output classes for classification.
    
    Returns:
        keras.Model: Residual feed-forward model.
        
    Example:
        ```python
        import keras
        model = create_residual_feedforward(input_dim=20, num_classes=3)
        ```
    """
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Residual blocks
    x = GatedResidualNetwork(units=64, dropout_rate=0.1)(inputs)
    x = GatedResidualNetwork(units=64, dropout_rate=0.1)(x)
    x = GatedResidualNetwork(units=64, dropout_rate=0.1)(x)
    
    # Output
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)
```

### Multi-Branch Feed-Forward

```python
def create_multibranch_feedforward(input_dim: int, num_classes: int) -> keras.Model:
    """Create a multi-branch feed-forward model.
    
    Builds a feed-forward network with multiple branches that process input features
    in different ways and combines their outputs for enhanced feature representation.
    
    Args:
        input_dim: Dimension of input features.
        num_classes: Number of output classes for classification.
    
    Returns:
        keras.Model: Multi-branch feed-forward model.
        
    Example:
        ```python
        import keras
        model = create_multibranch_feedforward(input_dim=20, num_classes=3)
        ```
    """
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Branch 1: Feature selection
    branch1 = VariableSelection(hidden_dim=64)(inputs)
    branch1 = keras.layers.Dense(64, activation='relu')(branch1)
    
    # Branch 2: Direct processing
    branch2 = keras.layers.Dense(64, activation='relu')(inputs)
    branch2 = keras.layers.Dense(64, activation='relu')(branch2)
    
    # Combine branches
    x = keras.layers.Concatenate()([branch1, branch2])
    x = keras.layers.Dense(128, activation='relu')(x)
    
    # Output
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)
```

## ⚡ Performance Optimization

### Memory-Efficient Feed-Forward

```python
def create_memory_efficient_feedforward(input_dim: int, num_classes: int) -> keras.Model:
    """Create a memory-efficient feed-forward model.
    
    Constructs a lightweight feed-forward network with reduced dimensionality
    for deployment on memory-constrained devices.
    
    Args:
        input_dim: Dimension of input features.
        num_classes: Number of output classes for classification.
    
    Returns:
        keras.Model: Memory-efficient feed-forward model.
        
    Example:
        ```python
        import keras
        model = create_memory_efficient_feedforward(input_dim=20, num_classes=3)
        ```
    """
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Use smaller dimensions
    x = VariableSelection(hidden_dim=32)(inputs)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    
    # Output
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)
```

### Speed-Optimized Feed-Forward

```python
def create_speed_optimized_feedforward(input_dim: int, num_classes: int) -> keras.Model:
    """Create a speed-optimized feed-forward model.
    
    Builds a minimal feed-forward network designed for fast inference
    with minimal computational overhead.
    
    Args:
        input_dim: Dimension of input features.
        num_classes: Number of output classes for classification.
    
    Returns:
        keras.Model: Speed-optimized feed-forward model.
        
    Example:
        ```python
        import keras
        model = create_speed_optimized_feedforward(input_dim=20, num_classes=3)
        ```
    """
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Minimal layers for speed
    x = VariableSelection(hidden_dim=32)(inputs)
    x = keras.layers.Dense(64, activation='relu')(x)
    
    # Output
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)
```

## 🌍 Real-World Examples

### Financial Risk Assessment

```python
def create_financial_risk_model(input_dim: int, num_classes: int) -> keras.Model:
    """Create a financial risk assessment model.
    
    Constructs a specialized feed-forward model for financial risk assessment
    with preprocessing, feature selection, and multiple classification layers.
    
    Args:
        input_dim: Dimension of input features (financial indicators).
        num_classes: Number of risk classes for classification.
    
    Returns:
        keras.Model: Financial risk assessment model.
        
    Example:
        ```python
        import keras
        model = create_financial_risk_model(input_dim=20, num_classes=3)
        ```
    """
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Preprocessing
    x = DifferentiableTabularPreprocessor()(inputs)
    
    # Feature selection
    x = VariableSelection(hidden_dim=64)(x)
    
    # Risk assessment layers
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    
    # Output
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)
```

### Healthcare Outcome Prediction

```python
def create_healthcare_model(input_dim: int, num_classes: int) -> keras.Model:
    """Create a healthcare outcome prediction model.
    
    Builds a specialized feed-forward network for healthcare applications
    using feature embedding and gated fusion for medical outcome prediction.
    
    Args:
        input_dim: Dimension of input features (patient health indicators).
        num_classes: Number of outcome classes for prediction.
    
    Returns:
        keras.Model: Healthcare outcome prediction model.
        
    Example:
        ```python
        import keras
        model = create_healthcare_model(input_dim=20, num_classes=3)
        ```
    """
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Feature engineering
    x = AdvancedNumericalEmbedding(embedding_dim=64)(inputs)
    x = GatedFeatureFusion(hidden_dim=128)(x)
    
    # Medical processing layers
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    
    # Output
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)
```

## 📊 Training and Evaluation

### Training Configuration

```python
def train_feedforward_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> keras.callbacks.History:
    """Train a feed-forward model with proper configuration.
    
    Trains a feed-forward model using Adam optimizer with callbacks for early stopping
    and learning rate reduction on plateau.
    
    Args:
        model: Compiled Keras model to train.
        X_train: Training feature array of shape (n_samples, n_features).
        y_train: Training target array of shape (n_samples, n_classes).
        X_val: Validation feature array of shape (n_val_samples, n_features).
        y_val: Validation target array of shape (n_val_samples, n_classes).
    
    Returns:
        keras.callbacks.History: Training history object containing loss and metrics per epoch.
        
    Example:
        ```python
        import numpy as np
        import keras
        X_train = np.random.rand(100, 20)
        y_train = np.zeros((100, 3))
        y_train[np.arange(100), np.random.randint(0, 3, 100)] = 1
        X_val = np.random.rand(20, 20)
        y_val = np.zeros((20, 3))
        y_val[np.arange(20), np.random.randint(0, 3, 20)] = 1
        model = create_basic_feedforward(input_dim=20, num_classes=3)
        history = train_feedforward_model(model, X_train, y_train, X_val, y_val)
        ```
    """
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5
        )
    ]
    
    logger.info("Starting model training...")
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("Model training completed.")
    
    return history
```

### Model Evaluation

```python
def evaluate_feedforward_model(model: keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
    """Evaluate a feed-forward model.
    
    Evaluates model performance on test data and generates classification report
    with detailed metrics including precision, recall, and F1-score.
    
    Args:
        model: Trained Keras model to evaluate.
        X_test: Test feature array of shape (n_samples, n_features).
        y_test: One-hot encoded test target array of shape (n_samples, n_classes).
    
    Returns:
        Tuple[float, float]: Tuple containing (test_accuracy, test_loss).
        
    Example:
        ```python
        import numpy as np
        X_test = np.random.rand(20, 20)
        y_test = np.zeros((20, 3))
        y_test[np.arange(20), np.random.randint(0, 3, 20)] = 1
        model = create_basic_feedforward(input_dim=20, num_classes=3)
        test_accuracy, test_loss = evaluate_feedforward_model(model, X_test, y_test)
        ```
    """
    
    # Basic evaluation
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Predictions
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Additional metrics
    from sklearn.metrics import classification_report
    
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"\nClassification Report:\n{classification_report(true_classes, predicted_classes)}")
    
    return test_accuracy, test_loss
```

## 📚 Next Steps

1. **KDP Integration Guide**: Learn about Keras Data Processor integration
2. **Data Analyzer Examples**: Explore data analysis workflows
3. **Rich Docstrings Showcase**: See comprehensive examples
4. **API Reference**: Deep dive into layer parameters

---

**Ready for more examples?** Check out the [KDP Integration Guide](kdp_integration_guide.md) next!