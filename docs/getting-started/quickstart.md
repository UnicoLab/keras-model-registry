# 🚀 Quick Start Guide

Get up and running with KMR in minutes! This guide will walk you through installing KMR and building your first tabular model.

## 📦 Installation

```bash
pip install kmr
```

## 🎯 Your First Model

Here's a complete example that demonstrates the power of KMR layers:

```python
import keras
from kmr.layers import (
    TabularAttention, 
    VariableSelection, 
    GatedFeatureFusion,
    DifferentiableTabularPreprocessor
)

# Create a simple tabular model
def create_tabular_model(input_dim, num_classes):
    inputs = keras.Input(shape=(input_dim,))
    
    # Preprocessing layer
    x = DifferentiableTabularPreprocessor()(inputs)
    
    # Variable selection
    x = VariableSelection(hidden_dim=64)(x)
    
    # Attention mechanism
    x = TabularAttention(num_heads=8, key_dim=64)(x)
    
    # Feature fusion
    x = GatedFeatureFusion(hidden_dim=128)(x)
    
    # Output layer
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

# Build and compile model
model = create_tabular_model(input_dim=20, num_classes=3)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model created successfully!")
print(f"Total parameters: {model.count_params():,}")
```

## 🔧 Key Concepts

### 1. **Layer Categories**
- **🧠 Attention**: Focus on important features and relationships
- **⚙️ Preprocessing**: Handle missing values and data preparation
- **🔧 Feature Engineering**: Transform and select features intelligently
- **🏗️ Specialized**: Advanced architectures for specific use cases
- **🛠️ Utility**: Essential tools for data processing

### 2. **Layer Composition**
KMR layers are designed to work together seamlessly:

```python
# Example: Building a feature engineering pipeline
from kmr.layers import (
    AdvancedNumericalEmbedding,
    DistributionAwareEncoder,
    SparseAttentionWeighting
)

# Create feature processing pipeline
def feature_pipeline(inputs):
    # Embed numerical features
    x = AdvancedNumericalEmbedding(embedding_dim=64)(inputs)
    
    # Encode with distribution awareness
    x = DistributionAwareEncoder(encoding_dim=64)(x)
    
    # Apply sparse attention weighting
    x = SparseAttentionWeighting(temperature=1.0)(x)
    
    return x
```

### 3. **Performance Optimization**
KMR layers are optimized for production use:

```python
# Example: Memory-efficient model
def create_efficient_model(input_dim):
    inputs = keras.Input(shape=(input_dim,))
    
    # Use memory-efficient layers
    x = DifferentiableTabularPreprocessor()(inputs)
    x = VariableSelection(hidden_dim=32)(x)  # Smaller hidden dim
    x = TabularAttention(num_heads=4, key_dim=32)(x)  # Fewer heads
    
    return keras.Model(inputs, x)
```

## 📚 Next Steps

1. **Explore Layers**: Check out the [Layer Explorer](../layers_overview.md) to see all available layers
2. **Read Documentation**: Dive deep into specific layers in the [Layers section](../layers/)
3. **Try Examples**: Run through the [Examples](../examples/README.md) to see real-world applications
4. **API Reference**: Consult the [API Reference](../api/layers.md) for detailed parameter information

## 🆘 Need Help?

- **Documentation**: Browse the comprehensive layer documentation
- **Examples**: Check out the examples directory for practical implementations
- **GitHub**: Report issues or contribute to the project

---

**Ready to build amazing tabular models?** Start with the [Layer Explorer](layers_overview.md) to discover all available layers!
