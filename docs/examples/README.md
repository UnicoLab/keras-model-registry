# 📖 KMR Examples

This directory contains comprehensive examples demonstrating how to use KMR layers and models effectively. Each example showcases the rich documentation and best practices built into KMR.

## 🎯 Available Examples

### ✨ Rich Docstrings Showcase
- **File**: `rich_docstrings_showcase.md`
- **Description**: Demonstrates the comprehensive documentation available in KMR layers
- **Highlights**: Parameter documentation, usage examples, best practices, and performance considerations

## 📚 Example Categories

### 1. 🚀 Basic Usage Examples
Simple, clear examples showing how to use individual layers:

```python
import keras
from kmr.layers import AdvancedGraphFeatureLayer

# Create sample data
x = keras.random.normal((32, 10))

# Create the layer with comprehensive parameter documentation
layer = AdvancedGraphFeatureLayer(
    embed_dim=16,      # Dimensionality of feature embeddings
    num_heads=4,       # Number of attention heads
    dropout_rate=0.1,  # Dropout for regularization
    hierarchical=True, # Enable hierarchical aggregation
    num_groups=4       # Number of feature groups
)

# Apply the layer
y = layer(x, training=True)
print("Output shape:", y.shape)  # (32, 10)
```

### 2. 🎯 Advanced Integration Examples
Complex scenarios showing layer combinations:

```python
from kmr.layers import TabularAttention, AdvancedNumericalEmbedding
from kmr.models import BaseFeedForwardModel

# Create a comprehensive tabular model
inputs = keras.Input(shape=(100, 20))  # 100 samples, 20 features

# Apply advanced numerical embedding
embedded = AdvancedNumericalEmbedding(
    embedding_dim=32,
    mlp_hidden_units=64,
    num_bins=20
)(inputs)

# Apply tabular attention
attended = TabularAttention(
    num_heads=8,
    d_model=64,
    dropout_rate=0.1
)(embedded)

# Create final model
model = keras.Model(inputs, attended)
```

### 3. ⚡ Best Practices Examples
Demonstrating optimal usage patterns:

```python
# Performance-optimized configuration
layer = AdvancedGraphFeatureLayer(
    embed_dim=32,      # Start with moderate embedding size
    num_heads=8,       # Use 8 heads for good performance
    dropout_rate=0.1,  # Standard dropout rate
    hierarchical=True, # Enable for large feature sets
    num_groups=8       # Group features for efficiency
)
```

## Documentation Features

### Comprehensive Parameter Documentation
Each layer includes:
- **Type annotations**: Complete type hints for all parameters
- **Default values**: Sensible defaults with explanations
- **Validation rules**: Parameter constraints and error messages
- **Usage guidance**: When and how to use each parameter

### Rich Usage Examples
Multiple examples for different scenarios:
- **Basic usage**: Simple, clear examples
- **Advanced usage**: Complex scenarios with explanations
- **Integration examples**: How to combine with other layers
- **Performance examples**: Optimized configurations

### Best Practices and Performance Notes
Guidance on:
- **When to use**: Specific scenarios where layers excel
- **Performance considerations**: Memory usage and scalability
- **Common pitfalls**: Mistakes to avoid
- **Optimization tips**: How to get the best performance

### Implementation Details
Technical information for developers:
- **Architecture overview**: How the layer works internally
- **Input/output specifications**: Shape requirements and transformations
- **Keras 3 compatibility**: Backend-agnostic implementation
- **Serialization support**: Save/load functionality

## Getting Started

1. **Browse the API Reference**: Start with the [Layers API](../api/layers.md) to see all available layers
2. **Read the Examples**: Check out the [Rich Docstrings Showcase](rich_docstrings_showcase.md)
3. **Follow Best Practices**: Use the guidance in each layer's documentation
4. **Experiment**: Try the examples and modify them for your use case

## Contributing Examples

If you have interesting examples or use cases, please contribute them! See the [Contributing Guide](../contributing.md) for details on how to add new examples.
