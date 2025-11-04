# 🤖 Models API Reference

Welcome to the KMR Models documentation! All models are designed to work exclusively with **Keras 3** and provide specialized implementations for advanced machine learning tasks including time series forecasting, tabular data processing, and multimodal learning.

!!! tip "What You'll Find Here"
    Each model includes detailed documentation with:
    - ✨ **Complete parameter descriptions** with types and defaults
    - 🎯 **Usage examples** showing real-world applications
    - ⚡ **Best practices** and performance considerations
    - 🎨 **When to use** guidance for each model
    - 🔧 **Implementation notes** for developers

!!! success "Production-Ready"
    All models are fully tested, documented, and ready for production use.

!!! note "Keras 3 Compatible"
    All models are built on top of Keras base classes and are fully compatible with Keras 3.

## ⏱️ Time Series Forecasting

### 🎛️ TimeMixer
TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting. 

A state-of-the-art time series forecasting model that uses decomposable components and multi-scale mixing to capture both seasonal and trend patterns at different temporal scales.

::: kmr.models.TimeMixer

**Key Features:**
- Trend-seasonal decomposition (moving average or DFT)
- Multi-scale seasonal and trend mixing
- Channel-independent or dependent processing
- Support for temporal features (month, day, hour, etc.)
- Reversible instance normalization for improved training
- Multivariate time series forecasting

**Architecture:**
- Decomposition layer extracts seasonal and trend components
- Multi-scale mixing layers hierarchically combine patterns
- Encoder blocks with past decomposable mixing
- Projection layers for forecast horizon
- Reversible normalization for stable training

**References:**
- Wang, S., et al. (2023). "TimeMixer: Decomposable Multiscale Mixing For Time Series Forecasting"

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
