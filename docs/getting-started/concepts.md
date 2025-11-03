# 🧠 Core Concepts

Understand the fundamental concepts behind KMR and how to effectively use its layers for modeling.

## 🎯 What is KMR?

KMR (Keras Model Registry) is a comprehensive collection of specialized layers designed exclusively for tabular data (but not only !!!). Unlike traditional neural network layers that were designed for images or sequences, KMR layers understand the unique characteristics of tabular data.

### Key Principles

1. **Tabular-First Design**: Every layer is optimized for tabular data characteristics
2. **Production Ready**: Battle-tested layers used in real-world applications
3. **Keras 3 Native**: Built specifically for Keras 3 with modern best practices
4. **No TensorFlow Dependencies**: Pure Keras implementation for maximum compatibility

## 📊 Understanding Tabular Data

### Characteristics of Tabular Data

```python
# Example tabular dataset
import pandas as pd
import numpy as np

# Sample tabular data
data = {
    'age': [25, 30, 35, 40, 45],
    'income': [50000, 75000, 90000, 110000, 130000],
    'education': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master'],
    'city': ['NYC', 'SF', 'LA', 'Chicago', 'Boston']
}

df = pd.DataFrame(data)
print(df)
```

**Key Characteristics:**
- **Mixed Data Types**: Numerical and categorical features
- **No Spatial Structure**: Unlike images, features don't have spatial relationships
- **Variable Importance**: Some features are more important than others
- **Missing Values**: Common in real-world datasets
- **Feature Interactions**: Complex relationships between features

## 🏗️ Layer Architecture

### Layer Categories

#### 1. **🧠 Attention Layers**
Focus on important features and relationships:

```python
from kmr.layers import TabularAttention, ColumnAttention, RowAttention

# Tabular attention for feature relationships
attention = TabularAttention(num_heads=8, key_dim=64)

# Column attention for feature importance
col_attention = ColumnAttention(hidden_dim=64)

# Row attention for sample relationships
row_attention = RowAttention(hidden_dim=64)
```

#### 2. **⚙️ Preprocessing Layers**
Handle data preparation and missing values:

```python
from kmr.layers import (
    DifferentiableTabularPreprocessor,
    DateParsingLayer,
    DateEncodingLayer
)

# End-to-end preprocessing
preprocessor = DifferentiableTabularPreprocessor(
    imputation_strategy='learnable',
    normalization='learnable'
)

# Date handling
date_parser = DateParsingLayer()
date_encoder = DateEncodingLayer()
```

#### 3. **🔧 Feature Engineering Layers**
Transform and select features intelligently:

```python
from kmr.layers import (
    VariableSelection,
    GatedFeatureFusion,
    AdvancedNumericalEmbedding
)

# Intelligent feature selection
var_sel = VariableSelection(hidden_dim=64)

# Feature fusion
fusion = GatedFeatureFusion(hidden_dim=128)

# Advanced numerical embedding
embedding = AdvancedNumericalEmbedding(embedding_dim=64)
```

#### 4. **🏗️ Specialized Layers**
Advanced architectures for specific use cases:

```python
from kmr.layers import (
    GatedResidualNetwork,
    TransformerBlock,
    TabularMoELayer
)

# Gated residual network
grn = GatedResidualNetwork(units=64, dropout_rate=0.2)

# Transformer block
transformer = TransformerBlock(dim_model=64, num_heads=4)

# Mixture of experts
moe = TabularMoELayer(num_experts=4, expert_units=16)
```

#### 5. **🛠️ Utility Layers**
Essential tools for data processing:

```python
from kmr.layers import (
    CastToFloat32Layer,
    NumericalAnomalyDetection,
    FeatureCutout
)

# Type casting
cast_layer = CastToFloat32Layer()

# Anomaly detection
anomaly_detector = NumericalAnomalyDetection()

# Data augmentation
cutout = FeatureCutout(cutout_prob=0.1)
```

## 🔄 Layer Composition Patterns

### 1. **Sequential Composition**
Layers applied in sequence:

```python
def create_sequential_model(input_dim):
    inputs = keras.Input(shape=(input_dim,))
    
    # Sequential processing
    x = DifferentiableTabularPreprocessor()(inputs)
    x = VariableSelection(hidden_dim=64)(x)
    x = TabularAttention(num_heads=8, key_dim=64)(x)
    x = GatedFeatureFusion(hidden_dim=128)(x)
    
    return keras.Model(inputs, x)
```

### 2. **Parallel Composition**
Multiple processing branches:

```python
def create_parallel_model(input_dim):
    inputs = keras.Input(shape=(input_dim,))
    
    # Parallel processing branches
    branch1 = VariableSelection(hidden_dim=64)(inputs)
    branch2 = TabularAttention(num_heads=8, key_dim=64)(inputs)
    
    # Fusion
    x = GatedFeatureFusion(hidden_dim=128)([branch1, branch2])
    
    return keras.Model(inputs, x)
```

### 3. **Residual Composition**
Skip connections for gradient flow:

```python
def create_residual_model(input_dim):
    inputs = keras.Input(shape=(input_dim,))
    
    # Residual block
    x = GatedResidualNetwork(units=64)(inputs)
    x = GatedResidualNetwork(units=64)(x)
    
    # Skip connection
    x = keras.layers.Add()([inputs, x])
    
    return keras.Model(inputs, x)
```

## 🎛️ Layer Parameters

### Common Parameters

#### **Hidden Dimensions**
```python
# Control model capacity
layer = VariableSelection(hidden_dim=64)  # Small model
layer = VariableSelection(hidden_dim=256) # Large model
```

#### **Dropout Rates**
```python
# Regularization
layer = TabularAttention(dropout=0.1)  # Light regularization
layer = TabularAttention(dropout=0.3)  # Heavy regularization
```

#### **Attention Heads**
```python
# Multi-head attention
layer = TabularAttention(num_heads=4)  # Few heads
layer = TabularAttention(num_heads=16) # Many heads
```

### Performance Considerations

#### **Memory Usage**
```python
# Memory-efficient configuration
layer = TabularAttention(
    num_heads=4,      # Fewer heads
    key_dim=32,       # Smaller key dimension
    dropout=0.1
)
```

#### **Computational Speed**
```python
# Fast configuration
layer = VariableSelection(
    hidden_dim=32,    # Smaller hidden dimension
    dropout=0.1       # Light dropout
)
```

## 🔍 Best Practices

### 1. **Start Simple**
Begin with basic layers and gradually add complexity:

```python
# Start with preprocessing
x = DifferentiableTabularPreprocessor()(inputs)

# Add feature selection
x = VariableSelection(hidden_dim=64)(x)

# Add attention
x = TabularAttention(num_heads=8, key_dim=64)(x)
```

### 2. **Monitor Performance**
Track training metrics and adjust accordingly:

```python
# Monitor during training
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Use callbacks for monitoring
callbacks = [
    keras.callbacks.EarlyStopping(patience=10),
    keras.callbacks.ReduceLROnPlateau(factor=0.5)
]
```

### 3. **Experiment with Architectures**
Try different layer combinations:

```python
# Architecture 1: Attention-focused
def attention_model(inputs):
    x = TabularAttention(num_heads=8)(inputs)
    x = GatedFeatureFusion(hidden_dim=128)(x)
    return x

# Architecture 2: Selection-focused
def selection_model(inputs):
    x = VariableSelection(hidden_dim=64)(inputs)
    x = GatedResidualNetwork(units=64)(x)
    return x
```

## 📚 Next Steps

1. **Explore Layers**: Check out the [Layer Explorer](../layers-explorer.md)
2. **Read Documentation**: Dive into specific layer documentation
3. **Try Examples**: Run through practical examples
4. **Build Models**: Start creating your own tabular models

---

**Ready to dive deeper?** Explore the [Layer Explorer](../layers-explorer.md) to see all available layers!
