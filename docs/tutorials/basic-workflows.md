# 🔄 Basic Workflows

Learn the fundamental workflows for building tabular models with KMR layers. This tutorial covers the most common patterns and best practices.

## 📋 Table of Contents

1. [Data Preparation](#data-preparation)
2. [Model Building](#model-building)
3. [Training and Evaluation](#training-and-evaluation)
4. [Common Patterns](#common-patterns)
5. [Troubleshooting](#troubleshooting)

## 📊 Data Preparation

### Loading and Preprocessing

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from kmr.layers import DifferentiableTabularPreprocessor

# Load your dataset
df = pd.read_csv('your_dataset.csv')

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to numpy arrays
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

print(f"Training shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
```

### Handling Missing Values

```python
from kmr.layers import DifferentiableTabularPreprocessor

# Create preprocessing layer
preprocessor = DifferentiableTabularPreprocessor(
    imputation_strategy='learnable',
    normalization='learnable'
)

# Fit on training data
preprocessor.adapt(X_train)

# Transform data
X_train_processed = preprocessor(X_train)
X_test_processed = preprocessor(X_test)
```

## 🏗️ Model Building

### Basic Sequential Model

```python
import keras
from kmr.layers import (
    VariableSelection,
    TabularAttention,
    GatedFeatureFusion
)

def create_basic_model(input_dim, num_classes):
    """Create a basic tabular model with KMR layers."""
    
    # Input layer
    inputs = keras.Input(shape=(input_dim,))
    
    # Preprocessing
    x = DifferentiableTabularPreprocessor()(inputs)
    
    # Feature selection
    x = VariableSelection(hidden_dim=64, dropout=0.1)(x)
    
    # Attention mechanism
    x = TabularAttention(num_heads=8, key_dim=64, dropout=0.1)(x)
    
    # Feature fusion
    x = GatedFeatureFusion(hidden_dim=128, dropout=0.1)(x)
    
    # Output layer
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

# Create model
model = create_basic_model(input_dim=X_train.shape[1], num_classes=3)
model.summary()
```

### Advanced Model with Residual Connections

```python
from kmr.layers import GatedResidualNetwork, TransformerBlock

def create_advanced_model(input_dim, num_classes):
    """Create an advanced model with residual connections."""
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Preprocessing
    x = DifferentiableTabularPreprocessor()(inputs)
    
    # Residual blocks
    x = GatedResidualNetwork(units=64, dropout_rate=0.1)(x)
    x = GatedResidualNetwork(units=64, dropout_rate=0.1)(x)
    
    # Transformer block
    x = TransformerBlock(dim_model=64, num_heads=4, ff_units=128)(x)
    
    # Feature selection
    x = VariableSelection(hidden_dim=64)(x)
    
    # Output
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

# Create advanced model
advanced_model = create_advanced_model(input_dim=X_train.shape[1], num_classes=3)
```

## 🎯 Training and Evaluation

### Model Compilation

```python
# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# For regression tasks
# model.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=0.001),
#     loss='mse',
#     metrics=['mae']
# )
```

### Training with Callbacks

```python
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Define callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )
]

# Train model
history = model.fit(
    X_train_processed,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)
```

### Evaluation

```python
# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test_processed, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Make predictions
predictions = model.predict(X_test_processed)
predicted_classes = np.argmax(predictions, axis=1)
```

## 🔄 Common Patterns

### 1. **Feature Engineering Pipeline**

```python
from kmr.layers import (
    AdvancedNumericalEmbedding,
    DistributionAwareEncoder,
    SparseAttentionWeighting
)

def feature_engineering_pipeline(inputs):
    """Advanced feature engineering pipeline."""
    
    # Numerical embedding
    x = AdvancedNumericalEmbedding(embedding_dim=64)(inputs)
    
    # Distribution-aware encoding
    x = DistributionAwareEncoder(encoding_dim=64)(x)
    
    # Sparse attention weighting
    x = SparseAttentionWeighting(temperature=1.0)(x)
    
    return x
```

### 2. **Multi-Head Processing**

```python
from kmr.layers import MultiResolutionTabularAttention

def multi_head_model(inputs):
    """Model with multi-resolution attention."""
    
    # Multi-resolution attention
    x = MultiResolutionTabularAttention(
        num_heads=8,
        numerical_heads=4,
        categorical_heads=4
    )(inputs)
    
    # Feature fusion
    x = GatedFeatureFusion(hidden_dim=128)(x)
    
    return x
```

### 3. **Ensemble Approach**

```python
from kmr.layers import BoostingEnsembleLayer

def ensemble_model(inputs):
    """Model with boosting ensemble."""
    
    # Boosting ensemble
    x = BoostingEnsembleLayer(
        num_learners=3,
        learner_units=64
    )(inputs)
    
    # Final processing
    x = GatedResidualNetwork(units=64)(x)
    
    return x
```

### 4. **Anomaly Detection**

```python
from kmr.layers import NumericalAnomalyDetection

def anomaly_detection_model(inputs):
    """Model with anomaly detection."""
    
    # Anomaly detection
    anomaly_output = NumericalAnomalyDetection()(inputs)
    
    # Main processing
    x = VariableSelection(hidden_dim=64)(inputs)
    x = TabularAttention(num_heads=8)(x)
    
    return x, anomaly_output
```

## 🐛 Troubleshooting

### Common Issues

#### **Memory Issues**
```python
# Reduce model size
layer = TabularAttention(
    num_heads=4,      # Reduce from 8
    key_dim=32,       # Reduce from 64
    dropout=0.1
)

# Use smaller batch size
model.fit(X_train, y_train, batch_size=16)  # Instead of 32
```

#### **Training Instability**
```python
# Add gradient clipping
model.compile(
    optimizer=keras.optimizers.Adam(clipnorm=1.0),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Use learning rate scheduling
def lr_schedule(epoch):
    return 0.001 * (0.1 ** (epoch // 20))

callbacks.append(keras.callbacks.LearningRateScheduler(lr_schedule))
```

#### **Overfitting**
```python
# Increase regularization
layer = VariableSelection(
    hidden_dim=64,
    dropout=0.3  # Increase dropout
)

# Add early stopping
callbacks.append(
    EarlyStopping(monitor='val_loss', patience=5)
)
```

### Performance Optimization

#### **Speed Optimization**
```python
# Use fewer attention heads
layer = TabularAttention(num_heads=4, key_dim=32)

# Reduce hidden dimensions
layer = VariableSelection(hidden_dim=32)

# Use mixed precision
keras.mixed_precision.set_global_policy('mixed_float16')
```

#### **Memory Optimization**
```python
# Use gradient checkpointing
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    run_eagerly=False  # Use graph mode
)

# Process data in smaller chunks
def process_in_chunks(data, chunk_size=1000):
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        result = model.predict(chunk)
        results.append(result)
    return np.concatenate(results)
```

## 📚 Next Steps

1. **Feature Engineering**: Learn advanced feature engineering techniques
2. **Model Building**: Explore specialized architectures
3. **Examples**: See real-world applications
4. **API Reference**: Deep dive into layer parameters

---

**Ready for more advanced topics?** Check out [Feature Engineering](feature-engineering.md) next!
