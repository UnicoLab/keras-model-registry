# 🚀 Welcome to KMR - Keras Model Registry

<div align="center">
  <img src="kmr_logo.png" width="350" alt="KMR Logo"/>
  
  <p><strong>🧩 Reusable Model Architecture Bricks in Keras 3</strong></p>
  
  <p><strong>🏢 Provided and maintained by <a href="https://unicolab.ai" target="_blank">UnicoLab</a></strong></p>
</div>

!!! success "🎯 Production-Ready Tabular AI"
    Build sophisticated tabular models with **50+ specialized layers**, **smart preprocessing**, **intelligent feature engineering**, and **recommendation systems** - all designed exclusively for Keras 3.

---

## 🎯 What is KMR?

KMR (Keras Model Registry) is a comprehensive collection of **production-ready layers and models** designed specifically for **tabular data processing** with Keras 3. Our library provides:

- 🧠 **Advanced Attention Mechanisms** for tabular data
- 🔧 **Feature Engineering Layers** for data preprocessing  
- 🏗️ **Pre-built Models** for common ML tasks
- 📊 **Recommendation Systems** with collaborative filtering, content-based, and geospatial models
- 📈 **Data Analysis Tools** for intelligent layer recommendations
- ⚡ **Keras 3 Native** - No TensorFlow dependencies in production code

!!! tip "Why KMR?"
    KMR eliminates the need to build complex tabular models from scratch. Our layers are battle-tested, well-documented, and designed to work seamlessly together.

---


## 🧩 What's Inside KMR?

<div class="grid cards" markdown>

- **50+ Production Layers**

    Advanced attention mechanisms, feature processing, recommendation systems, and specialized architectures ready for production use.

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

## 💡 See It In Action - Build Real Models Now!

=== "⚡ Pre-built Model (Fastest)"

    Get a production-ready model in 3 lines of code:
    
    ```python
    import keras
    from kmr.models import SFNEBlock
    
    # Create a state-of-the-art tabular model
    model = SFNEBlock(
        input_dim=25,              # Number of input features
        hidden_dim=128,            # Hidden representation size
        num_blocks=3,              # Number of processing blocks
        output_dim=10              # Number of output classes
    )
    
    # Compile and train
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print("✅ State-of-the-art model ready!")
    ```
    
    **When to use:** Maximum performance, zero architecture design effort.

=== "🎨 Custom Layers (Full Control)"

    Build advanced models by combining specialized layers:
    
    ```python
    import keras
    from kmr.layers import (
        DistributionTransformLayer,    # Intelligent preprocessing
        VariableSelection,              # Learn feature importance
        TabularAttention,               # Model feature relationships
        GatedFeatureFusion              # Combine representations
    )
    
    inputs = keras.Input(shape=(15,))
    
    # Build processing pipeline
    x = DistributionTransformLayer()(inputs)
    x = VariableSelection(num_features=15)(x)
    x = TabularAttention(num_heads=4, head_dim=16)(x)
    
    # Combine representations
    linear = keras.layers.Dense(32, activation='relu')(x)
    nonlinear = keras.layers.Dense(32, activation='tanh')(x)
    x = GatedFeatureFusion()([linear, nonlinear])
    
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    ```
    
    **When to use:** Fine-grained control, reuse battle-tested components.

=== "🚀 Classification (Common Use Case)"

    Production-ready classification with all best practices:
    
    ```python
    import keras
    from kmr.models import BaseFeedForwardModel
    
    # Create robust classification model
    model = BaseFeedForwardModel(
        input_dim=20,
        output_dim=1,
        hidden_layers=[256, 128, 64],
        activation='relu',
        dropout_rate=0.2
    )
    
    # Compile with production metrics
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    print("✅ Production model ready!")
    ```
    
    **When to use:** Proven architecture for classification tasks.

---

## 🎯 Real-World Use Cases

=== "💰 Financial Risk Modeling"

    Predict credit risk with advanced tabular features:
    
    ```python
    from kmr.models import BaseFeedForwardModel
    
    # 50+ financial features → Risk prediction
    model = BaseFeedForwardModel(
        input_dim=50,
        output_dim=1,
        hidden_layers=[256, 128, 64]
    )
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['auc', 'precision', 'recall']
    )
    ```
    
    **Use case:** Loan approval, credit scoring, fraud detection

=== "🏥 Healthcare Analytics"

    Intelligent medical diagnosis with mixed data:
    
    ```python
    from kmr.layers import (
        DistributionTransformLayer,
        VariableSelection,
        TabularAttention
    )
    
    inputs = keras.Input(shape=(30,))
    x = DistributionTransformLayer()(inputs)
    x = VariableSelection(num_features=30)(x)
    x = TabularAttention(num_heads=4, head_dim=16)(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    ```
    
    **Use case:** Disease prediction, patient risk assessment, diagnosis support

=== "🛒 E-commerce Recommendations"

    Build recommendation systems with collaborative filtering and content-based features:
    
    ```python
    from kmr.models import MatrixFactorizationModel, TwoTowerModel
    
    # Option 1: Collaborative Filtering
    model = MatrixFactorizationModel(
        num_users=10000,
        num_items=5000,
        embedding_dim=64,
        top_k=10
    )
    
    # Option 2: Content-Based (Two-Tower)
    model = TwoTowerModel(
        user_feature_dim=20,
        item_feature_dim=15,
        output_dim=64,
        top_k=10
    )
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    ```
    
    **Use case:** Product recommendations, CTR prediction, customer lifetime value, personalized search

---


## 🎨 Key Technical Features

<div class="grid cards" markdown>

- **🧠 Advanced Architecture**

    Graph-based feature relationships • Multi-head attention mechanisms • Hierarchical aggregation • Residual connections for stable training

- **⚡ Performance Optimized**

    Keras 3 native • Memory efficient • GPU ready • Fully serializable for production deployment

- **🔧 Developer Friendly**

    Complete type annotations • 461+ passing tests • Rich docstrings with examples • Modular design for customization

</div>

---

## 🚀 Perfect For

<div class="grid cards" markdown>

- **🏢 Enterprise ML Teams**

    Build production systems that scale. Battle-tested layers, 461+ passing tests, clear APIs for team collaboration, and detailed monitoring support.
    
    [Get Started →](getting-started/installation.md){ .md-button }

- **🔬 Research & Development**

    Experiment with cutting-edge techniques. State-of-the-art architectures, easy composition, reproducible results, and detailed docstrings throughout.
    
    [Explore Layers →](api/layers.md){ .md-button }

- **🎓 Learning & Education**

    Master tabular deep learning. Rich examples from basic to advanced, learn from production code, interactive examples, and best practices embedded in the library.
    
    [Start Learning →](getting-started/quickstart.md){ .md-button }

- **⚙️ Data Engineering**

    Streamline feature engineering. Intelligent feature layers, automatic preprocessing, data quality analysis, and built-in layer recommendations.
    
    [Try Analyzer →](data_analyzer.md){ .md-button }

</div>

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
