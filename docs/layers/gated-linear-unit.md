---
title: GatedLinearUnit - KMR
description: Gated linear unit that applies gated linear transformation to control information flow in neural networks
keywords: [gated linear unit, GLU, gating mechanism, information flow, keras, neural networks, feature transformation]
---

# 🚪 GatedLinearUnit

<div class="layer-hero">
  <div class="layer-hero-content">
    <h1>🚪 GatedLinearUnit</h1>
    <div class="layer-badges">
      <span class="badge badge-intermediate">🟡 Intermediate</span>
      <span class="badge badge-stable">✅ Stable</span>
      <span class="badge badge-popular">🔥 Popular</span>
    </div>
  </div>
</div>

## 🎯 Overview

The `GatedLinearUnit` applies a gated linear transformation to input tensors, controlling information flow in neural networks. It multiplies the output of a dense linear transformation with the output of a dense sigmoid transformation, creating a gating mechanism that filters information based on learned weights and biases.

This layer is particularly powerful for controlling information flow, implementing attention-like mechanisms, and creating sophisticated feature transformations in neural networks.

## 🔍 How It Works

The GatedLinearUnit processes data through a gated transformation:

1. **Linear Transformation**: Applies dense linear transformation to input
2. **Sigmoid Transformation**: Applies dense sigmoid transformation to input
3. **Gating Mechanism**: Multiplies linear output with sigmoid output
4. **Information Filtering**: The sigmoid output acts as a gate controlling information flow
5. **Output Generation**: Produces gated and filtered features

```mermaid
graph TD
    A[Input Features] --> B[Linear Dense Layer]
    A --> C[Sigmoid Dense Layer]
    B --> D[Linear Output]
    C --> E[Sigmoid Output (Gate)]
    D --> F[Element-wise Multiplication]
    E --> F
    F --> G[Gated Output]
    
    style A fill:#e6f3ff,stroke:#4a86e8
    style G fill:#e8f5e9,stroke:#66bb6a
    style B fill:#fff9e6,stroke:#ffb74d
    style C fill:#f3e5f5,stroke:#9c27b0
    style F fill:#e1f5fe,stroke:#03a9f4
```

## 💡 Why Use This Layer?

| Challenge | Traditional Approach | GatedLinearUnit's Solution |
|-----------|---------------------|---------------------------|
| **Information Flow** | No control over information flow | 🎯 **Gated control** of information flow |
| **Feature Filtering** | All features treated equally | ⚡ **Selective filtering** based on learned gates |
| **Attention Mechanisms** | Separate attention layers | 🧠 **Built-in gating** for attention-like behavior |
| **Feature Transformation** | Simple linear transformations | 🔗 **Sophisticated gated** transformations |

## 📊 Use Cases

- **Information Flow Control**: Controlling how information flows through networks
- **Feature Filtering**: Filtering features based on learned importance
- **Attention Mechanisms**: Implementing attention-like behavior
- **Feature Transformation**: Sophisticated feature processing
- **Ensemble Learning**: As a component in ensemble architectures

## 🚀 Quick Start

### Basic Usage

```python
import keras
from kmr.layers import GatedLinearUnit

# Create sample input data
batch_size, input_dim = 32, 16
x = keras.random.normal((batch_size, input_dim))

# Apply gated linear unit
glu = GatedLinearUnit(units=8)
output = glu(x)

print(f"Input shape: {x.shape}")           # (32, 16)
print(f"Output shape: {output.shape}")     # (32, 8)
```

### In a Sequential Model

```python
import keras
from kmr.layers import GatedLinearUnit

model = keras.Sequential([
    keras.layers.Dense(32, activation='relu'),
    GatedLinearUnit(units=16),
    keras.layers.Dense(8, activation='relu'),
    GatedLinearUnit(units=4),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### In a Functional Model

```python
import keras
from kmr.layers import GatedLinearUnit

# Define inputs
inputs = keras.Input(shape=(20,))  # 20 features

# Apply gated linear unit
x = GatedLinearUnit(units=16)(inputs)

# Continue processing
x = keras.layers.Dense(32, activation='relu')(x)
x = GatedLinearUnit(units=16)(x)
x = keras.layers.Dense(8, activation='relu')(x)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, outputs)
```

### Advanced Configuration

```python
# Advanced configuration with multiple GLU layers
def create_gated_network():
    inputs = keras.Input(shape=(30,))
    
    # Multiple GLU layers with different configurations
    x = GatedLinearUnit(units=32)(inputs)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = GatedLinearUnit(units=32)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = GatedLinearUnit(units=16)(x)
    
    # Final processing
    x = keras.layers.Dense(8, activation='relu')(x)
    
    # Multi-task output
    classification = keras.layers.Dense(3, activation='softmax', name='classification')(x)
    regression = keras.layers.Dense(1, name='regression')(x)
    
    return keras.Model(inputs, [classification, regression])

model = create_gated_network()
model.compile(
    optimizer='adam',
    loss={'classification': 'categorical_crossentropy', 'regression': 'mse'},
    loss_weights={'classification': 1.0, 'regression': 0.5}
)
```

## 📖 API Reference

::: kmr.layers.GatedLinearUnit

## 🔧 Parameters Deep Dive

### `units` (int)
- **Purpose**: Dimensionality of the output space
- **Range**: 1 to 1000+ (typically 8-128)
- **Impact**: Determines the size of the gated output
- **Recommendation**: Start with 16-32, scale based on data complexity

## 📈 Performance Characteristics

- **Speed**: ⚡⚡⚡⚡ Very fast - simple mathematical operations
- **Memory**: 💾💾 Low memory usage - minimal additional parameters
- **Accuracy**: 🎯🎯🎯🎯 Excellent for information flow control
- **Best For**: Networks requiring sophisticated information flow control

## 🎨 Examples

### Example 1: Information Flow Control

```python
import keras
import numpy as np
from kmr.layers import GatedLinearUnit

# Create a network with controlled information flow
def create_controlled_flow_network():
    inputs = keras.Input(shape=(25,))
    
    # Initial processing
    x = keras.layers.Dense(64, activation='relu')(inputs)
    x = keras.layers.BatchNormalization()(x)
    
    # Gated processing stages
    x = GatedLinearUnit(units=32)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = GatedLinearUnit(units=16)(x)
    x = keras.layers.Dense(16, activation='relu')(x)
    x = GatedLinearUnit(units=8)(x)
    
    # Final output
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    
    return keras.Model(inputs, outputs)

model = create_controlled_flow_network()
model.compile(optimizer='adam', loss='binary_crossentropy')

# Test with sample data
sample_data = keras.random.normal((100, 25))
predictions = model(sample_data)
print(f"Controlled flow predictions shape: {predictions.shape}")
```

### Example 2: Feature Filtering Analysis

```python
# Analyze how GLU filters features
def analyze_feature_filtering():
    # Create model with GLU
    inputs = keras.Input(shape=(10,))
    x = GatedLinearUnit(units=5)(inputs)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Test with different input patterns
    test_inputs = [
        keras.ops.convert_to_tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),  # First feature only
        keras.ops.convert_to_tensor([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]),  # Second feature only
        keras.ops.convert_to_tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]),  # First half
        keras.ops.convert_to_tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]),  # Second half
    ]
    
    print("Feature Filtering Analysis:")
    print("=" * 50)
    
    for i, test_input in enumerate(test_inputs):
        prediction = model(test_input)
        print(f"Test {i+1}: Prediction = {prediction.numpy()[0][0]:.4f}")
    
    return model

# Analyze feature filtering
# model = analyze_feature_filtering()
```

### Example 3: Attention-like Behavior

```python
# Create attention-like behavior with GLU
def create_attention_like_network():
    inputs = keras.Input(shape=(20,))
    
    # Create attention-like gates
    attention_gate = GatedLinearUnit(units=20)(inputs)
    
    # Apply attention to features
    attended_features = inputs * attention_gate
    
    # Process attended features
    x = keras.layers.Dense(32, activation='relu')(attended_features)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(16, activation='relu')(x)
    
    # Output
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    
    return keras.Model(inputs, outputs)

model = create_attention_like_network()
model.compile(optimizer='adam', loss='binary_crossentropy')

# Test attention-like behavior
sample_data = keras.random.normal((50, 20))
predictions = model(sample_data)
print(f"Attention-like predictions shape: {predictions.shape}")
```

## 💡 Tips & Best Practices

- **Units**: Start with 16-32 units, scale based on data complexity
- **Information Flow**: Use GLU to control how information flows through networks
- **Feature Filtering**: GLU can act as a learned feature filter
- **Attention**: GLU can implement attention-like mechanisms
- **Combination**: Works well with other Keras layers
- **Regularization**: Consider adding dropout after GLU layers

## ⚠️ Common Pitfalls

- **Units**: Must be positive integer
- **Output Size**: Output size is determined by units parameter
- **Gradient Flow**: GLU can affect gradient flow - monitor training
- **Overfitting**: Can overfit on small datasets - use regularization
- **Memory Usage**: Scales with units parameter

## 🔗 Related Layers

- [GatedResidualNetwork](gated-residual-network.md) - GRN using GLU
- [VariableSelection](variable-selection.md) - Variable selection with gating
- [SparseAttentionWeighting](sparse-attention-weighting.md) - Sparse attention weighting
- [TabularAttention](tabular-attention.md) - Attention mechanisms

## 📚 Further Reading

- [Gated Linear Units](https://arxiv.org/abs/1612.08083) - Original GLU paper
- [Information Flow in Neural Networks](https://en.wikipedia.org/wiki/Information_flow) - Information flow concepts
- [Attention Mechanisms](https://distill.pub/2016/augmented-rnns/) - Attention mechanism concepts
- [KMR Layer Explorer](../layers_overview.md) - Browse all available layers
- [Feature Engineering Tutorial](../tutorials/feature-engineering.md) - Complete guide to feature engineering
