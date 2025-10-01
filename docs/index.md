# 🚀 Welcome to KMR - Keras Model Registry

<div align="center">
  <img src="kmr_logo.png" width="350" alt="KMR Logo"/>
  
  <p><strong>🧩 Reusable Model Architecture Bricks in Keras 3</strong></p>
  
  <p><strong>🏢 Provided and maintained by <a href="https://unicolab.ai" target="_blank">UnicoLab</a></strong></p>
</div>

---

## 🎯 What is KMR?

KMR (Keras Model Registry) is a comprehensive collection of **production-ready layers and models** designed specifically for **tabular data processing** with Keras 3. Our library provides:

- 🧠 **Advanced Attention Mechanisms** for tabular data
- 🔧 **Feature Engineering Layers** for data preprocessing  
- 🏗️ **Pre-built Models** for common ML tasks
- 📊 **Data Analysis Tools** for intelligent layer recommendations
- ⚡ **Keras 3 Native** - No TensorFlow dependencies in production code

!!! tip "Why KMR?"
    KMR eliminates the need to build complex tabular models from scratch. Our layers are battle-tested, well-documented, and designed to work seamlessly together.

## 🚀 Quick Start

### Installation

```bash
pip install kmr
```

### Basic Usage

```python
import keras
from kmr.layers import AdvancedGraphFeatureLayer, TabularAttention

# Create sample tabular data
x = keras.random.normal((32, 10))  # 32 samples, 10 features

# Apply advanced graph-based feature processing
graph_layer = AdvancedGraphFeatureLayer(
    embed_dim=16,
    num_heads=4,
    hierarchical=True,
    num_groups=4
)

# Process with tabular attention
attention_layer = TabularAttention(
    num_heads=8,
    d_model=64,
    dropout_rate=0.1
)

# Build your model
output = attention_layer(graph_layer(x))
print(f"Output shape: {output.shape}")  # (32, 100, 64)
```

!!! success "That's it!"
    In just a few lines, you've created a sophisticated tabular model with graph-based feature processing and multi-head attention!

## 🧩 Core Components

### 🎯 Attention Layers
- **TabularAttention**: Dual attention for features and samples
- **AdvancedGraphFeature**: Graph-based feature relationships
- **ColumnAttention** & **RowAttention**: Specialized attention mechanisms
- **InterpretableMultiHeadAttention**: Attention with interpretability

### 🔧 Feature Engineering
- **AdvancedNumericalEmbedding**: Dual-branch numerical feature processing
- **DateEncodingLayer**: Comprehensive date/time feature extraction
- **DistributionTransformLayer**: Automatic distribution transformation
- **VariableSelection**: Intelligent feature selection

### 🏗️ Pre-built Models
- **BaseFeedForwardModel**: Flexible feed-forward architecture
- **SFNEBlock**: Sparse Feature Network Ensemble
- **TerminatorModel**: Comprehensive tabular model

### 🔍 Smart Tools
- **DataAnalyzer**: Intelligent layer recommendations
- **CLI Tools**: Command-line data analysis

## 📚 Documentation Highlights

Our documentation is designed to be **developer-friendly** with:

- ✨ **Rich Docstrings**: Every layer includes comprehensive examples, best practices, and performance notes
- 🎯 **Usage Examples**: Multiple scenarios from basic to advanced
- ⚡ **Performance Tips**: Memory usage, scalability, and optimization guidance
- 🔗 **Cross-references**: Easy navigation between related components

!!! example "Try the Interactive Examples"
    Check out our [Rich Docstrings Showcase](examples/rich_docstrings_showcase.md) to see the comprehensive documentation in action!

## 🎨 Key Features

=== "🧠 Advanced Architecture"
    - **Graph-based Processing**: Learn feature relationships dynamically
    - **Multi-head Attention**: Capture complex feature interactions  
    - **Hierarchical Aggregation**: Efficient processing of large feature sets
    - **Residual Connections**: Stable training and better gradients

=== "⚡ Performance Optimized"
    - **Keras 3 Native**: Latest Keras features and optimizations
    - **Memory Efficient**: Optimized for large-scale tabular data
    - **GPU Ready**: Full GPU acceleration support
    - **Serializable**: Save and load models seamlessly

=== "🔧 Developer Friendly"
    - **Type Annotations**: Complete type hints for better IDE support
    - **Comprehensive Testing**: Extensive test coverage
    - **Clear Documentation**: Rich docstrings with examples
    - **Modular Design**: Mix and match layers as needed

## 🚀 Why Choose KMR?

!!! success "Production Ready"
    All layers are battle-tested, well-documented, and designed for production use with comprehensive error handling and validation.

!!! tip "Keras 3 Exclusive"
    Built exclusively for Keras 3 with no TensorFlow dependencies in production code, ensuring compatibility and performance.

!!! example "Rich Documentation"
    Every layer includes comprehensive examples, best practices, performance notes, and usage guidance.

!!! note "Modular Design"
    Mix and match layers to build custom architectures that fit your specific use case.

## 🎯 Perfect For

<div class="feature-grid">

<div class="feature-card">

### 🏢 Enterprise ML Teams
- **Scalable Architecture**: Handle large-scale tabular datasets
- **Production Ready**: Battle-tested layers with comprehensive testing
- **Team Collaboration**: Clear documentation and consistent APIs

</div>

<div class="feature-card">

### 🔬 Research & Development
- **Cutting-edge Techniques**: Latest attention mechanisms and graph processing
- **Experimentation**: Easy to combine and modify layers
- **Reproducibility**: Well-documented with examples

</div>

<div class="feature-card">

### 🎓 Learning & Education
- **Rich Examples**: Comprehensive documentation with real-world examples
- **Best Practices**: Learn from production-ready implementations
- **Interactive**: Try examples and modify them for learning

</div>

</div>

## 🤝 Contributing

We welcome contributions! Whether you're:

- 🐛 **Reporting bugs** or suggesting improvements
- 🧩 **Adding new layers** or models
- 📝 **Improving documentation** or examples
- 🔍 **Enhancing data analysis** tools

Check out our [Contributing Guide](contributing.md) to get started!

## 📖 Next Steps

1. **📋 Browse Examples**: Start with our [Examples Overview](examples/README.md)
2. **🧩 Explore Layers**: Check out the [Layers API](api/layers.md)
3. **🏗️ Build Models**: See available [Models](api/models.md)
4. **🔍 Analyze Data**: Try our [Data Analyzer](data_analyzer.md)

---

<p align="center">
  <strong>Ready to build amazing tabular models? Let's get started! 🚀</strong>
</p>
