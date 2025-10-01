# 🌟 Keras Model Registry (KMR) - Reusable Model Architecture Bricks in Keras 🌟

<div align="center">
  <img src="docs/kmr_logo.png" width="350" alt="KMR Logo"/>
  
  <p><strong>Provided and maintained by <a href="https://unicolab.ai">UnicoLab</a></strong></p>
</div>

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Keras 3.8+](https://img.shields.io/badge/keras-3.8+-red.svg)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![UnicoLab](https://img.shields.io/badge/UnicoLab-Enterprise%20AI-blue.svg)](https://unicolab.ai)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://unicolab.github.io/keras-model-registry/)

**KMR** is a comprehensive collection of reusable Keras layers and models specifically designed for tabular data processing, feature engineering, and advanced neural network architectures. Built with Keras 3 and developed by [UnicoLab](https://unicolab.ai), it provides a clean, efficient, and extensible foundation for building sophisticated machine learning models for enterprise AI applications.

## ✨ Key Features

- **🎯 38+ Production-Ready Layers**: Attention mechanisms, feature processing, preprocessing, and specialized architectures
- **🧠 Advanced Models**: SFNE blocks, Terminator models, and more coming soon
- **📊 Data Analyzer**: Intelligent CSV analysis tool that recommends appropriate layers
- **🔬 Experimental Modules**: 20+ cutting-edge layers and models for research
- **⚡ Keras 3 Only**: Pure Keras 3 implementation with no TensorFlow dependencies
- **🧪 Comprehensive Testing**: Full test coverage with 38+ test suites
- **📚 Rich Documentation**: Detailed guides, examples, and API documentation

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI (coming soon)
pip install kmr

# Or install from source
git clone https://github.com/UnicoLab/keras-model-registry
cd keras-model-registry
pip install -e .
```

### 🚀 Quick Start Examples

#### Example 1: Simple Tabular Model with Attention

```python
import keras
from kmr.layers import TabularAttention, AdvancedNumericalEmbedding
from kmr.models import BaseFeedForwardModel

# Create a simple model for tabular data
inputs = keras.Input(shape=(20,))  # 20 features

# Apply advanced numerical embedding
embedded = AdvancedNumericalEmbedding(embed_dim=16, num_heads=4)(inputs)

# Add tabular attention
attention = TabularAttention(num_heads=4, d_model=16, dropout_rate=0.1)
# Reshape for attention (add sequence dimension)
reshaped = keras.ops.expand_dims(embedded, axis=1)
attended = attention(reshaped)
flattened = keras.ops.reshape(attended, (-1, 16))

# Final prediction
outputs = keras.layers.Dense(1, activation='sigmoid')(flattened)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Model created with", model.count_params(), "parameters")
```

#### Example 2: Feature Engineering Pipeline

```python
import keras
from kmr.layers import (
    DateEncodingLayer, 
    DistributionTransformLayer,
    GatedFeatureFusion,
    VariableSelection
)

# Create a feature engineering pipeline
inputs = keras.Input(shape=(10,))  # 10 features

# Apply distribution transformation
transformed = DistributionTransformLayer(transform_type='auto')(inputs)

# Create two feature representations
feat1 = keras.layers.Dense(16, activation='relu')(transformed)
feat2 = keras.layers.Dense(16, activation='tanh')(transformed)

# Fuse features using gated fusion
fused = GatedFeatureFusion()([feat1, feat2])

# Apply variable selection
# Reshape for variable selection (needs 3D input)
reshaped = keras.ops.expand_dims(fused, axis=1)
selected, weights = VariableSelection(nr_features=1, units=16, use_context=False)(reshaped)

# Final output
outputs = keras.layers.Dense(1)(selected)

model = keras.Model(inputs=inputs, outputs=outputs)
print("Feature engineering model ready!")
```

#### Example 3: Using Pre-built Models

```python
import keras
from kmr.models import BaseFeedForwardModel

# Create a feed-forward model with individual feature inputs
feature_names = ['age', 'income', 'education', 'experience']
model = BaseFeedForwardModel(
    feature_names=feature_names,
    hidden_units=[64, 32, 16],
    output_units=1,
    dropout_rate=0.2
)

# Prepare data (each feature as separate input)
age = keras.random.normal((100, 1))
income = keras.random.normal((100, 1))
education = keras.random.normal((100, 1))
experience = keras.random.normal((100, 1))

# Train the model
model.compile(optimizer='adam', loss='mse')
model.fit([age, income, education, experience], 
          keras.random.normal((100, 1)), 
          epochs=10, verbose=0)

print("Model trained successfully!")
```

### Data Analyzer

```python
from kmr.utils import analyze_data

# Analyze your CSV data and get layer recommendations
results = analyze_data("path/to/your/data.csv")
recommendations = results["recommendations"]

print("Recommended layers:")
for layer in recommendations:
    print(f"- {layer['layer_name']}: {layer['description']}")
```

## 🏗️ Architecture Overview

### Core Components

#### **Layers** (`kmr.layers`)
- **Attention Mechanisms**: `TabularAttention`, `MultiResolutionTabularAttention`, `ColumnAttention`, `RowAttention`
- **Feature Processing**: `AdvancedNumericalEmbedding`, `GatedFeatureFusion`, `VariableSelection`
- **Preprocessing**: `DateEncodingLayer`, `DateParsingLayer`, `DifferentiableTabularPreprocessor`
- **Advanced Architectures**: `TransformerBlock`, `GatedResidualNetwork`, `BoostingBlock`
- **Specialized Layers**: `BusinessRulesLayer`, `StochasticDepth`, `FeatureCutout`

#### **Models** (`kmr.models`)
- **SFNEBlock**: Advanced feature processing block
- **TerminatorModel**: Multi-block hierarchical processing model

#### **Utilities** (`kmr.utils`)
- **Data Analyzer**: Intelligent CSV analysis and layer recommendation system
- **CLI Tools**: Command-line interface for data analysis

#### **Experimental** (`experimental/`)
- **Time Series**: 12+ specialized time series preprocessing layers
- **Advanced Models**: Neural Additive Models, Temporal Fusion Transformers, and more
- **Research Components**: Cutting-edge architectures for experimentation
- **Note**: Experimental components are not included in the PyPI package

## 📖 Documentation

- **[Online Documentation](https://unicolab.github.io/keras-model-registry/)**: Full API reference with automatic docstring generation
- **[API Reference](https://unicolab.github.io/keras-model-registry/api/)**: Complete documentation for all layers, models, and utilities
- **[Layer Implementation Guide](docs/layers_implementation_guide.md)**: Comprehensive guide for implementing new layers
- **[Data Analyzer Documentation](docs/data_analyzer.md)**: Complete guide to the data analysis tools
- **[Contributing Guide](docs/contributing.md)**: How to contribute to the project

## 🎯 Use Cases

### Tabular Data Processing
```python
from kmr.layers import DifferentiableTabularPreprocessor, TabularAttention

# Preprocess tabular data
preprocessor = DifferentiableTabularPreprocessor(
    numerical_features=['age', 'income'],
    categorical_features=['category', 'region']
)

# Apply attention mechanism
attention = TabularAttention(num_heads=8, d_model=64)
```

### Feature Engineering
```python
from kmr.layers import AdvancedNumericalEmbedding, GatedFeatureFusion

# Advanced numerical embedding
embedding = AdvancedNumericalEmbedding(embed_dim=32, num_heads=4)

# Feature fusion
fusion = GatedFeatureFusion(units=64, dropout_rate=0.1)
```

### Date and Time Processing
```python
from kmr.layers import DateParsingLayer, DateEncodingLayer, SeasonLayer

# Parse dates
date_parser = DateParsingLayer(date_format="YYYY-MM-DD")

# Encode dates
date_encoder = DateEncodingLayer(embed_dim=16)

# Extract seasonal features
season_extractor = SeasonLayer()
```

## 🧪 Testing

```bash
# Run all tests
make all_tests

# Run specific test categories
make unittests
make data_analyzer_tests

# Generate coverage report
make coverage
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/contributing.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/UnicoLab/keras-model-registry.git
cd keras-model-registry

# Install development dependencies
poetry install

# Install pre-commit hooks
pre-commit install

# Run tests
make all_tests
```

### Commit Convention

We use semantic commit messages:
- `feat(KMR): add new layer for feature processing`
- `fix(KMR): resolve serialization issue`
- `docs(KMR): update installation guide`

## 📊 Performance

KMR is optimized for performance with:
- **Keras 3 Backend**: Leverages the latest Keras optimizations
- **Efficient Operations**: Uses only Keras operations for maximum compatibility
- **Memory Optimization**: Careful memory management in complex layers
- **Batch Processing**: Optimized for batch operations

## 🔮 Roadmap

- [ ] **v0.3.0**: Additional model architectures and pre-trained models
- [ ] **v0.4.0**: Integration with popular ML frameworks
- [ ] **v0.5.0**: Model zoo with pre-trained weights
- [ ] **v1.0.0**: Production-ready with comprehensive benchmarks

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Keras 3](https://keras.io/)
- Inspired by modern deep learning research
- Community-driven development

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/UnicoLab/keras-model-registry/issues)
- **Discussions**: [GitHub Discussions](https://github.com/UnicoLab/keras-model-registry/discussions)
- **Documentation**: [Online Docs](https://unicolab.github.io/keras-model-registry/)

---

<p align="center">
  <strong>Built with ❤️ for the Keras community</strong>
</p>