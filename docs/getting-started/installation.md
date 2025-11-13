# 📦 Installation Guide

Install KerasFactory and get your development environment ready for tabular modeling with Keras 3.

## 🎯 Quick Install

```bash
pip install kerasfactory
```

## 🔧 Requirements

### Python Version
- **Python 3.8+** (recommended: Python 3.10+)

### Core Dependencies
- **Keras 3.0+** (TensorFlow backend recommended for testing)
- **NumPy 1.21+**
- **Pandas 1.3+** (for data handling or tf.DataSet if you have tensorflow)

### Optional Dependencies
- **Matplotlib** (for visualization)
- **Seaborn** (for statistical plots)
- **Scikit-learn** (for preprocessing utilities)

## 🚀 Installation Methods

### 1. **Pip Install (Recommended)**
```bash
# Latest stable release
pip install kerasfactory

# With optional dependencies
pip install kerasfactory[full]

# Specific version
pip install kerasfactory==1.0.0
```

### 2. **Development Install**
```bash
# Clone the repository
git clone https://github.com/UnicoLab/KerasFactory.git
cd KerasFactory

# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

### 3. **Conda Install**
```bash
# Create a new environment
conda create -n kerasfactory python=3.10
conda activate kerasfactory

# Install KerasFactory
pip install kerasfactory
```

## 🔍 Verify Installation

Test your installation with this simple script:

```python
import keras
from kerasfactory.layers import TabularAttention

# Test basic import
print("✅ KerasFactory imported successfully!")

# Test layer creation
layer = TabularAttention(num_heads=8, key_dim=64)
print("✅ TabularAttention layer created!")

# Test with sample data
import numpy as np
x = np.random.random((32, 20))
output = layer(x)
print(f"✅ Layer output shape: {output.shape}")
```

## 🐛 Troubleshooting

### Common Issues

#### **ImportError: No module named 'keras'**
```bash
# Install Keras 3
pip install keras>=3.0.0
```

#### **TensorFlow Backend Issues**
```bash
# Install TensorFlow
pip install tensorflow>=2.13.0

# Or use JAX backend
pip install jax jaxlib
```

#### **Memory Issues**
```python
# Set memory growth for TensorFlow
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

### Backend Configuration

KerasFactory works with multiple Keras backends:

```python
# TensorFlow backend (default)
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

# JAX backend
os.environ['KERAS_BACKEND'] = 'jax'

# PyTorch backend
os.environ['KERAS_BACKEND'] = 'torch'
```

## 📋 System Requirements

### Minimum Requirements
- **RAM**: 4GB
- **Storage**: 1GB free space
- **CPU**: 2 cores

### Recommended Requirements
- **RAM**: 8GB+
- **Storage**: 5GB+ free space
- **CPU**: 4+ cores
- **GPU**: NVIDIA GPU with CUDA support (optional)

## 🔄 Updating KerasFactory

```bash
# Update to latest version
pip install --upgrade kerasfactory

# Check current version
python -c "import kerasfactory; print(kerasfactory.__version__)"
```

## 🧪 Testing Installation

Run the test suite to ensure everything works:

```bash
# Run basic tests
python -c "
import kerasfactory
from kerasfactory.layers import *
print('All layers imported successfully!')
"

# Run comprehensive tests (if available)
pytest tests/
```

## 📚 Next Steps

1. **Quick Start**: Follow the [Quick Start Guide](quickstart.md)
2. **Explore Layers**: Check out the [Layer Explorer](../layers_overview.md)
3. **Read Documentation**: Browse the [Layers section](../layers_overview.md)
4. **Try Examples**: Run through the [Examples](../examples/README.md)

---

**Installation complete!** Ready to start building with KerasFactory? Head to the [Quick Start Guide](quickstart.md)!
