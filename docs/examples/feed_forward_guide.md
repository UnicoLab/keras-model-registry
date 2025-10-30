# 🏗️ BaseFeedForwardModel Guide

Learn how to build feed-forward models using KMR layers. This guide covers the fundamentals of creating efficient feed-forward architectures for tabular data.

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
from kmr.layers import VariableSelection, GatedFeatureFusion

def create_basic_feedforward(input_dim, num_classes):
    """Create a basic feed-forward model with KMR layers."""
    
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
model = create_basic_feedforward(input_dim=20, num_classes=3)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### Feed-Forward with Feature Engineering

```python
from kmr.layers import (
    DifferentiableTabularPreprocessor,
    AdvancedNumericalEmbedding,
    GatedFeatureFusion
)

def create_engineered_feedforward(input_dim, num_classes):
    """Create a feed-forward model with feature engineering."""
    
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
from kmr.layers import GatedResidualNetwork

def create_residual_feedforward(input_dim, num_classes):
    """Create a residual feed-forward model."""
    
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
def create_multibranch_feedforward(input_dim, num_classes):
    """Create a multi-branch feed-forward model."""
    
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
def create_memory_efficient_feedforward(input_dim, num_classes):
    """Create a memory-efficient feed-forward model."""
    
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
def create_speed_optimized_feedforward(input_dim, num_classes):
    """Create a speed-optimized feed-forward model."""
    
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
def create_financial_risk_model(input_dim, num_classes):
    """Create a financial risk assessment model."""
    
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
def create_healthcare_model(input_dim, num_classes):
    """Create a healthcare outcome prediction model."""
    
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
def train_feedforward_model(model, X_train, y_train, X_val, y_val):
    """Train a feed-forward model with proper configuration."""
    
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
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    return history
```

### Model Evaluation

```python
def evaluate_feedforward_model(model, X_test, y_test):
    """Evaluate a feed-forward model."""
    
    # Basic evaluation
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Predictions
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Additional metrics
    from sklearn.metrics import classification_report
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes))
    
    return test_accuracy, test_loss
```

## 📚 Next Steps

1. **KDP Integration Guide**: Learn about Keras Data Processor integration
2. **Data Analyzer Examples**: Explore data analysis workflows
3. **Rich Docstrings Showcase**: See comprehensive examples
4. **API Reference**: Deep dive into layer parameters

---

**Ready for more examples?** Check out the [KDP Integration Guide](kdp_integration_guide.md) next!