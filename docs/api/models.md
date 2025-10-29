# 🏗️ Models API Reference

Welcome to the KMR Models documentation! All models are designed to work exclusively with **Keras 3** and provide high-level abstractions for common tabular data processing tasks.

!!! tip "What You'll Find Here"
    Each model includes detailed documentation with:
    - ✨ **Complete parameter descriptions** with types and defaults
    - 🎯 **Usage examples** showing real-world applications
    - ⚡ **Best practices** and performance considerations
    - 🎨 **When to use** guidance for each model
    - 🔧 **Implementation notes** for developers

!!! success "Ready-to-Use Models"
    These models provide complete architectures that you can use out-of-the-box or customize for your specific needs.

!!! note "Base Classes"
    All models inherit from `BaseModel` ensuring consistent behavior and Keras 3 compatibility.

## 🏗️ Core Models

### 🚀 BaseFeedForwardModel
Flexible feed-forward model architecture for tabular data with customizable layers.

::: kmr.models.feed_forward.BaseFeedForwardModel

## 🎯 Advanced Models

### 🧩 SFNEBlock
Sparse Feature Network Ensemble block for advanced feature processing and ensemble learning.

::: kmr.models.SFNEBlock

### 🎭 TerminatorModel
Comprehensive tabular model that combines multiple SFNE blocks for complex data tasks.

::: kmr.models.TerminatorModel

### 🔍 Autoencoder
Advanced autoencoder model for anomaly detection with optional preprocessing integration and automatic threshold configuration.

::: kmr.models.autoencoder.Autoencoder

## 🔧 Base Classes

### 🏛️ BaseModel
Base class for all KMR models, providing common functionality and Keras 3 compatibility.

::: kmr.models._base.BaseModel
