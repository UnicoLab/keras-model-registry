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

---

## 💡 See It In Action - Build Real Models Now!

### ✨ Example 1: Pre-built Model (Fastest Way to Start)

Get a production-ready model in 3 lines of code:

```python
import keras
from kmr.models import SFNEBlock

# Create a state-of-the-art tabular model
model = SFNEBlock(
    input_dim=25,           # 25 input features
    hidden_dim=128,         # Hidden layer size
    num_blocks=3,           # Number of processing blocks
    output_dim=10           # 10-class classification
)

# Compile and train immediately
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=50, validation_split=0.2)
print("✅ State-of-the-art model ready to train!")
```

**Why use this?** 🎯 When you want maximum performance with zero architecture design effort.

---

### 🎨 Example 2: Mix & Match Layers (Custom Models Made Easy)

Build advanced models by combining specialized layers:

```python
import keras
from kmr.layers import (
    DistributionTransformLayer,    # Smart preprocessing
    VariableSelection,              # Feature importance learning
    TabularAttention,               # Feature relationship modeling
    GatedFeatureFusion,             # Intelligent feature combination
    GatedLinearUnit                 # Non-linear transformations
)

# Create your custom pipeline
inputs = keras.Input(shape=(15,))

# Build your model step by step
x = DistributionTransformLayer()(inputs)           # Preprocess intelligently
x = VariableSelection(num_features=15)(x)          # Learn which features matter
x = TabularAttention(num_heads=4, head_dim=16)(x)  # Model feature relationships
x = GatedFeatureFusion()([
    keras.layers.Dense(32, activation='relu')(x),
    keras.layers.Dense(32, activation='tanh')(x)
])  # Combine different representations
x = GatedLinearUnit(units=16)(x)                   # Final non-linear processing
outputs = keras.layers.Dense(1, activation='sigmoid')(x)

# That's it - sophisticated model ready!
model = keras.Model(inputs=inputs, outputs=outputs)
print("✅ Advanced custom model built with KMR layers!")
```

**Why use this?** 🎯 When you need fine-grained control and want to reuse battle-tested components.

---

### 🚀 Example 3: Quick Classification (Most Common Use Case)

Build a complete classification model with all bells and whistles:

```python
import keras
from kmr.models import BaseFeedForwardModel

# Create your model with advanced features built-in
model = BaseFeedForwardModel(
    input_dim=20,
    output_dim=1,
    hidden_layers=[256, 128, 64],
    activation='relu',
    dropout_rate=0.2
)

# Compile with production-ready metrics
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

print("✅ Production-ready classification model!")
# Now train: model.fit(X_train, y_train, epochs=50)
```

**Why use this?** 🎯 When you want a proven architecture that just works for classification tasks.

---

## 🧩 What's Inside KMR?

<div class="grid cards" markdown>

- **38+ Production Layers**

    Advanced attention mechanisms, feature processing, and specialized architectures ready for production use.

    [Explore All Layers →](api/layers.md){ .md-button .md-button--primary }

- **Smart Preprocessing**

    Automatic data transformation, date encoding, and intelligent feature engineering layers.

    [See Layer Examples →](examples/README.md){ .md-button .md-button--primary }

- **Pre-built Models**

    Ready-to-use models like BaseFeedForwardModel and SFNEBlock for common ML tasks.

    [View Models →](api/models.md){ .md-button .md-button--primary }

- **Data Analyzer**

    Intelligent CSV analysis tool that recommends the best layers for your specific data.

    [Try Analyzer →](data_analyzer.md){ .md-button .md-button--primary }

</div>

---

## 📚 Why Developers Love KMR

Our documentation is designed to be **developer-friendly** with:

- ✨ **Rich Docstrings** - Every layer includes comprehensive examples, best practices, and performance notes
- 🎯 **Usage Examples** - Multiple scenarios from basic to advanced use cases
- ⚡ **Performance Tips** - Memory usage, scalability, and optimization guidance  
- 🔗 **Cross-references** - Easy navigation between related components

!!! example "Interactive Learning"
    Check out our [Rich Docstrings Showcase](examples/rich_docstrings_showcase.md) to see comprehensive documentation in action!

---

## 🎨 Key Technical Features

### 🧠 Advanced Architecture
- **Graph-based Processing** - Learn feature relationships dynamically
- **Multi-head Attention** - Capture complex feature interactions  
- **Hierarchical Aggregation** - Efficient processing of large feature sets
- **Residual Connections** - Stable training and better gradients

### ⚡ Performance Optimized
- **Keras 3 Native** - Latest Keras features and optimizations
- **Memory Efficient** - Optimized for large-scale tabular data
- **GPU Ready** - Full GPU acceleration support
- **Serializable** - Save and load models seamlessly

### 🔧 Developer Friendly
- **Type Annotations** - Complete type hints for better IDE support
- **Comprehensive Testing** - Extensive test coverage with 461+ passing tests
- **Clear Documentation** - Rich docstrings with real-world examples
- **Modular Design** - Mix and match layers as needed

---

## 🚀 Perfect For

### 🏢 Enterprise ML Teams

**Build Production-Ready Systems**

✓ Scalable architecture for large-scale tabular datasets  
✓ Battle-tested layers with comprehensive testing  
✓ Clear APIs and consistent interfaces for team collaboration  
✓ Detailed logging and monitoring support  
✓ 461+ passing tests ensuring reliability

[Start Building →](getting-started/installation.md){ .md-button }

---

### 🔬 Research & Development

**Experiment with Cutting-Edge Techniques**

✓ Latest attention mechanisms and graph processing methods  
✓ Easy layer composition and modification for experimentation  
✓ Reproducible results with well-documented implementations  
✓ Access to state-of-the-art architectures  
✓ All layers include detailed docstrings with performance notes

[Explore Layers →](api/layers.md){ .md-button }

---

### 🎓 Learning & Education

**Master Tabular Deep Learning**

✓ Rich examples from basic to advanced  
✓ Learn from production-ready implementations  
✓ Interactive examples and tutorials  
✓ Best practices embedded in the library  
✓ Real-world use cases with working code

[Start Learning →](getting-started/quickstart.md){ .md-button }

---

### ⚙️ Data Engineering

**Streamline Feature Engineering**

✓ Intelligent feature transformation and selection layers  
✓ Automatic preprocessing with smart defaults  
✓ Data quality analysis and recommendations  
✓ Seamless integration with data pipelines  
✓ Built-in data analyzer for layer recommendations

[Try Data Analyzer →](data_analyzer.md){ .md-button }

---

## 📊 Quick Comparison: Time to Production

| Aspect | Without KMR | With KMR |
|--------|-------------|----------|
| **Lines to build attention layer** | 150+ | Use built-in layer |
| **Feature preprocessing** | Manual implementation | 1-line DistributionTransformLayer |
| **Feature selection** | Manual logic | VariableSelection layer |
| **Model training time** | 2-3 weeks | 1-2 hours |
| **Production readiness** | Additional QA needed | Built-in validation & testing |
| **Documentation quality** | Your responsibility | Rich docstrings with examples |
| **Performance optimization** | Trial & error | Best practices included |
| **Maintenance burden** | High | Low - rely on KMR updates |

---

## 🎯 Real-World Use Cases

### Financial Risk Modeling
```python
# Predict credit risk with advanced tabular features
from kmr.models import BaseFeedForwardModel
model = BaseFeedForwardModel(input_dim=50, output_dim=1, hidden_layers=[256, 128, 64])
```

### Healthcare Analytics
```python
# Analyze patient data with intelligent preprocessing
from kmr.layers import DistributionTransformLayer, VariableSelection
# Automatically handles mixed data types and missing values
```

### E-commerce Recommendations
```python
# Build recommendation systems with attention mechanisms
from kmr.layers import TabularAttention, GatedFeatureFusion
# Model complex user-item interactions effortlessly
```

---

## 🤝 Contributing

We welcome contributions! Whether you're:

- 🐛 **Reporting bugs** or suggesting improvements
- 🧩 **Adding new layers** or models
- 📝 **Improving documentation** or examples
- 🔍 **Enhancing data analysis** tools

Check out our [Contributing Guide](contributing.md) to get started!

---

## 📖 Your Journey with KMR

=== "🟢 Beginner"

    **Start here** - Get up and running in 5 minutes
    
    1. [Quick Start Guide](getting-started/quickstart.md)
    2. [Try pre-built models](api/models.md)
    3. [Run basic examples](examples/README.md)

=== "🟡 Intermediate"

    **Build custom models** - Mix and match layers
    
    1. [Explore all layers](api/layers.md)
    2. [Study layer combinations](examples/README.md)
    3. [Build your first custom model](tutorials/model-building.md)

=== "🔴 Advanced"

    **Push boundaries** - Extend and optimize
    
    1. [Study implementation details](layers_implementation_guide.md)
    2. [Analyze data with our tools](data_analyzer.md)
    3. [Contribute new layers](contributing.md)

---

<p align="center">
  <strong>Ready to build amazing tabular models?</strong>
  
  **Choose your path:**
  
  [⚡ Quick Start (5 min)](getting-started/quickstart.md){ .md-button .md-button--primary .md-button--large }
  
  [🧩 Explore Layers](api/layers.md){ .md-button .md-button--large }
  
  [🏗️ View Models](api/models.md){ .md-button .md-button--large }
</p>

---

<p align="center">
  <em>Join thousands of ML engineers building production-ready tabular models with KMR 🚀</em>
</p>
