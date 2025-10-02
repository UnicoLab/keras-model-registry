# 🚀 Welcome to KMR - Keras Model Registry

<div align="center">
  <img src="kmr_logo.png" width="350" alt="KMR Logo"/>
  
  <p><strong>🧩 Reusable Model Architecture Bricks in Keras 3</strong></p>
  
  <p><strong>🏢 Provided and maintained by <a href="https://unicolab.ai" target="_blank">UnicoLab</a></strong></p>
</div>

!!! success "🎯 Production-Ready Tabular AI"
    Build sophisticated tabular models with **38+ specialized layers**, **smart preprocessing**, and **intelligent feature engineering** - all designed exclusively for Keras 3.

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
# Install from PyPI (recommended)
pip install kmr

# Or install from source using Poetry
git clone https://github.com/UnicoLab/keras-model-registry
cd keras-model-registry
poetry install
```

### Basic Usage

```python
import keras
from kmr.layers import DistributionTransformLayer, GatedFeatureFusion

# Create sample tabular data
inputs = keras.Input(shape=(10,))  # 10 features

# Smart data preprocessing
transformed = DistributionTransformLayer(transform_type='auto')(inputs)

# Create two feature representations
linear_features = keras.layers.Dense(16, activation='relu')(transformed)
nonlinear_features = keras.layers.Dense(16, activation='tanh')(transformed)

# Intelligently combine features
fused = GatedFeatureFusion()([linear_features, nonlinear_features])

# Final prediction
outputs = keras.layers.Dense(1, activation='sigmoid')(fused)

model = keras.Model(inputs=inputs, outputs=outputs)
print("✅ Model ready! Smart preprocessing + intelligent feature fusion.")
```

!!! success "That's it!"
    In just a few lines, you've created a sophisticated tabular model with automatic data transformation and intelligent feature fusion!

## 🧩 What's Inside KMR?

<div class="grid cards" markdown>

-   :material-brain:{ .lg .middle } **38+ Production Layers**

    ---

    Advanced attention mechanisms, feature processing, and specialized architectures ready for production use.

    [:octicons-arrow-right-24: Explore Layers](api/layers.md)

-   :material-cog:{ .lg .middle } **Smart Preprocessing**

    ---

    Automatic data transformation, date encoding, and intelligent feature engineering layers.

    [:octicons-arrow-right-24: See Examples](examples/README.md)

-   :material-rocket-launch:{ .lg .middle } **Pre-built Models**

    ---

    Ready-to-use models like BaseFeedForwardModel and SFNEBlock for common ML tasks.

    [:octicons-arrow-right-24: View Models](api/models.md)

-   :material-chart-line:{ .lg .middle } **Data Analyzer**

    ---

    Intelligent CSV analysis tool that recommends the best layers for your specific data.

    [:octicons-arrow-right-24: Try Analyzer](data_analyzer.md)

</div>

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
