# 🏗️ Model Building Tutorial

Learn how to build sophisticated tabular models using KerasFactory layers. This tutorial covers advanced architectures, design patterns, and optimization techniques.

## 📋 Table of Contents

1. [Architecture Patterns](#architecture-patterns)
2. [Attention-Based Models](#attention-based-models)
3. [Residual and Gated Networks](#residual-and-gated-networks)
4. [Ensemble Methods](#ensemble-methods)
5. [Specialized Architectures](#specialized-architectures)
6. [Performance Optimization](#performance-optimization)

## 🏛️ Architecture Patterns

### 1. **Sequential Architecture**

The most straightforward approach - layers applied in sequence:

```python
import keras
from kerasfactory.layers import (
    DifferentiableTabularPreprocessor,
    VariableSelection,
    TabularAttention,
    GatedFeatureFusion
)

def create_sequential_model(input_dim, num_classes):
    """Create a sequential tabular model."""
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Sequential processing
    x = DifferentiableTabularPreprocessor()(inputs)
    x = VariableSelection(hidden_dim=64, dropout=0.1)(x)
    x = TabularAttention(num_heads=8, key_dim=64, dropout=0.1)(x)
    x = GatedFeatureFusion(hidden_dim=128, dropout=0.1)(x)
    
    # Output layer
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

# Usage
model = create_sequential_model(input_dim=20, num_classes=3)
model.summary()
```

### 2. **Parallel Architecture**

Multiple processing branches that are later combined:

```python
def create_parallel_model(input_dim, num_classes):
    """Create a parallel processing model."""
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Branch 1: Attention-based processing
    branch1 = TabularAttention(num_heads=8, key_dim=64)(inputs)
    branch1 = GatedFeatureFusion(hidden_dim=64)(branch1)
    
    # Branch 2: Selection-based processing
    branch2 = VariableSelection(hidden_dim=64)(inputs)
    branch2 = GatedFeatureFusion(hidden_dim=64)(branch2)
    
    # Branch 3: Direct processing
    branch3 = keras.layers.Dense(64, activation='relu')(inputs)
    branch3 = keras.layers.Dense(64, activation='relu')(branch3)
    
    # Combine branches
    combined = keras.layers.Concatenate()([branch1, branch2, branch3])
    x = keras.layers.Dense(128, activation='relu')(combined)
    
    # Output
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

# Usage
parallel_model = create_parallel_model(input_dim=20, num_classes=3)
```

### 3. **Residual Architecture**

Skip connections for improved gradient flow:

```python
from kerasfactory.layers import GatedResidualNetwork

def create_residual_model(input_dim, num_classes):
    """Create a residual model with skip connections."""
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Initial processing
    x = DifferentiableTabularPreprocessor()(inputs)
    
    # Residual blocks
    x = GatedResidualNetwork(units=64, dropout_rate=0.1)(x)
    x = GatedResidualNetwork(units=64, dropout_rate=0.1)(x)
    x = GatedResidualNetwork(units=64, dropout_rate=0.1)(x)
    
    # Skip connection
    x = keras.layers.Add()([inputs, x])
    
    # Output
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

# Usage
residual_model = create_residual_model(input_dim=20, num_classes=3)
```

## 🧠 Attention-Based Models

### 1. **Multi-Head Attention Model**

```python
from kerasfactory.layers import (
    TabularAttention,
    MultiResolutionTabularAttention,
    InterpretableMultiHeadAttention
)

def create_attention_model(input_dim, num_classes):
    """Create a multi-head attention model."""
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Multi-resolution attention
    x = MultiResolutionTabularAttention(
        num_heads=8,
        numerical_heads=4,
        categorical_heads=4,
        dropout=0.1
    )(inputs)
    
    # Interpretable attention
    x = InterpretableMultiHeadAttention(
        num_heads=8,
        key_dim=64,
        dropout=0.1
    )(x)
    
    # Feature fusion
    x = GatedFeatureFusion(hidden_dim=128)(x)
    
    # Output
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

# Usage
attention_model = create_attention_model(input_dim=20, num_classes=3)
```

### 2. **Column and Row Attention Model**

```python
from kerasfactory.layers import ColumnAttention, RowAttention

def create_column_row_attention_model(input_dim, num_classes):
    """Create a model with column and row attention."""
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Column attention (feature-level)
    x = ColumnAttention(hidden_dim=64, dropout=0.1)(inputs)
    
    # Row attention (sample-level)
    x = RowAttention(hidden_dim=64, dropout=0.1)(x)
    
    # Feature fusion
    x = GatedFeatureFusion(hidden_dim=128)(x)
    
    # Output
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

# Usage
column_row_model = create_column_row_attention_model(input_dim=20, num_classes=3)
```

## 🔄 Residual and Gated Networks

### 1. **Gated Residual Network**

```python
from kerasfactory.layers import GatedResidualNetwork, GatedLinearUnit

def create_gated_residual_model(input_dim, num_classes):
    """Create a gated residual network model."""
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Gated residual blocks
    x = GatedResidualNetwork(units=64, dropout_rate=0.1)(inputs)
    x = GatedResidualNetwork(units=64, dropout_rate=0.1)(x)
    x = GatedResidualNetwork(units=64, dropout_rate=0.1)(x)
    
    # Gated linear unit
    x = GatedLinearUnit(units=64)(x)
    
    # Output
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

# Usage
gated_residual_model = create_gated_residual_model(input_dim=20, num_classes=3)
```

### 2. **Transformer Block Model**

```python
from kerasfactory.layers import TransformerBlock

def create_transformer_model(input_dim, num_classes):
    """Create a transformer-based model."""
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Transformer blocks
    x = TransformerBlock(
        dim_model=64,
        num_heads=4,
        ff_units=128,
        dropout_rate=0.1
    )(inputs)
    
    x = TransformerBlock(
        dim_model=64,
        num_heads=4,
        ff_units=128,
        dropout_rate=0.1
    )(x)
    
    # Output
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

# Usage
transformer_model = create_transformer_model(input_dim=20, num_classes=3)
```

## 🎯 Ensemble Methods

### 1. **Mixture of Experts**

```python
from kerasfactory.layers import TabularMoELayer

def create_moe_model(input_dim, num_classes):
    """Create a mixture of experts model."""
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Mixture of experts
    x = TabularMoELayer(
        num_experts=4,
        expert_units=16
    )(inputs)
    
    # Additional processing
    x = GatedFeatureFusion(hidden_dim=128)(x)
    
    # Output
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

# Usage
moe_model = create_moe_model(input_dim=20, num_classes=3)
```

### 2. **Boosting Ensemble**

```python
from kerasfactory.layers import BoostingEnsembleLayer

def create_boosting_model(input_dim, num_classes):
    """Create a boosting ensemble model."""
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Boosting ensemble
    x = BoostingEnsembleLayer(
        num_learners=3,
        learner_units=64,
        hidden_activation='relu'
    )(inputs)
    
    # Output
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

# Usage
boosting_model = create_boosting_model(input_dim=20, num_classes=3)
```

## 🚀 Specialized Architectures

### 1. **Graph-Based Model**

```python
from kerasfactory.layers import (
    AdvancedGraphFeature,
    GraphFeatureAggregation,
    MultiHeadGraphFeaturePreprocessor
)

def create_graph_model(input_dim, num_classes):
    """Create a graph-based model."""
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Graph feature processing
    x = AdvancedGraphFeature(
        hidden_dim=64,
        num_heads=4,
        dropout=0.1
    )(inputs)
    
    # Graph aggregation
    x = GraphFeatureAggregation(
        aggregation_method='mean',
        hidden_dim=64
    )(x)
    
    # Multi-head graph preprocessing
    x = MultiHeadGraphFeaturePreprocessor(
        num_heads=4,
        hidden_dim=64
    )(x)
    
    # Output
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

# Usage
graph_model = create_graph_model(input_dim=20, num_classes=3)
```

### 2. **Anomaly Detection Model**

```python
from kerasfactory.layers import (
    NumericalAnomalyDetection,
    CategoricalAnomalyDetectionLayer
)

def create_anomaly_detection_model(input_dim, num_classes):
    """Create a model with anomaly detection."""
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Anomaly detection
    numerical_anomalies = NumericalAnomalyDetection()(inputs)
    categorical_anomalies = CategoricalAnomalyDetectionLayer()(inputs)
    
    # Main processing
    x = VariableSelection(hidden_dim=64)(inputs)
    x = TabularAttention(num_heads=8)(x)
    x = GatedFeatureFusion(hidden_dim=128)(x)
    
    # Output
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, [outputs, numerical_anomalies, categorical_anomalies])

# Usage
anomaly_model = create_anomaly_detection_model(input_dim=20, num_classes=3)
```

### 3. **Business Rules Integration**

```python
from kerasfactory.layers import BusinessRulesLayer

def create_business_rules_model(input_dim, num_classes, rules):
    """Create a model with business rules integration."""
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Business rules layer
    x = BusinessRulesLayer(
        rules=rules,
        feature_type='numerical',
        trainable_weights=True
    )(inputs)
    
    # Additional processing
    x = VariableSelection(hidden_dim=64)(x)
    x = TabularAttention(num_heads=8)(x)
    
    # Output
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

# Usage
rules = [
    {'feature': 'age', 'operator': '>', 'value': 18, 'weight': 1.0},
    {'feature': 'income', 'operator': '>', 'value': 50000, 'weight': 0.8}
]
business_model = create_business_rules_model(input_dim=20, num_classes=3, rules=rules)
```

## ⚡ Performance Optimization

### 1. **Memory-Efficient Model**

```python
def create_memory_efficient_model(input_dim, num_classes):
    """Create a memory-efficient model."""
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Use smaller dimensions
    x = VariableSelection(hidden_dim=32)(inputs)
    x = TabularAttention(num_heads=4, key_dim=32)(x)
    x = GatedFeatureFusion(hidden_dim=64)(x)
    
    # Output
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

# Usage
memory_efficient_model = create_memory_efficient_model(input_dim=20, num_classes=3)
```

### 2. **Speed-Optimized Model**

```python
def create_speed_optimized_model(input_dim, num_classes):
    """Create a speed-optimized model."""
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Use fewer layers and smaller dimensions
    x = VariableSelection(hidden_dim=32)(inputs)
    x = TabularAttention(num_heads=4, key_dim=32)(x)
    
    # Output
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

# Usage
speed_optimized_model = create_speed_optimized_model(input_dim=20, num_classes=3)
```

### 3. **Mixed Precision Training**

```python
# Enable mixed precision
keras.mixed_precision.set_global_policy('mixed_float16')

def create_mixed_precision_model(input_dim, num_classes):
    """Create a mixed precision model."""
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Use mixed precision layers
    x = VariableSelection(hidden_dim=64)(inputs)
    x = TabularAttention(num_heads=8, key_dim=64)(x)
    x = GatedFeatureFusion(hidden_dim=128)(x)
    
    # Output (use float32 for final layer)
    outputs = keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    return keras.Model(inputs, outputs)

# Usage
mixed_precision_model = create_mixed_precision_model(input_dim=20, num_classes=3)
```

## 🔧 Model Compilation and Training

### 1. **Advanced Compilation**

```python
def compile_model(model, learning_rate=0.001):
    """Compile model with advanced settings."""
    
    # Learning rate scheduling
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=0.9
    )
    
    # Compile with advanced optimizer
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=lr_schedule,
            clipnorm=1.0
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Usage
model = create_sequential_model(input_dim=20, num_classes=3)
model = compile_model(model, learning_rate=0.001)
```

### 2. **Advanced Training**

```python
def train_model(model, X_train, y_train, X_val, y_val):
    """Train model with advanced callbacks."""
    
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
            patience=5,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True
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

# Usage
history = train_model(model, X_train, y_train, X_val, y_val)
```

## 📊 Model Evaluation and Analysis

### 1. **Comprehensive Evaluation**

```python
def evaluate_model(model, X_test, y_test):
    """Comprehensive model evaluation."""
    
    # Basic evaluation
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Predictions
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Additional metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes))
    
    return test_accuracy, test_loss

# Usage
accuracy, loss = evaluate_model(model, X_test, y_test)
```

### 2. **Model Interpretation**

```python
def interpret_model(model, X_test, layer_name='tabular_attention'):
    """Interpret model using attention weights."""
    
    # Get attention weights
    attention_model = keras.Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output
    )
    
    attention_weights = attention_model.predict(X_test)
    
    # Analyze attention patterns
    mean_attention = np.mean(attention_weights, axis=0)
    print("Mean attention weights:", mean_attention)
    
    return attention_weights

# Usage
attention_weights = interpret_model(model, X_test)
```

## 📚 Next Steps

1. **Examples**: See real-world model building applications
2. **API Reference**: Deep dive into layer parameters
3. **Performance**: Optimize your models for production
4. **Advanced Topics**: Explore cutting-edge techniques

---

**Ready to see real examples?** Check out the [Examples](../examples/) section!
