# 🧩 Layers API Reference

Welcome to the KerasFactory Layers documentation! All layers are designed to work exclusively with **Keras 3** and provide specialized implementations for advanced tabular data processing, feature engineering, attention mechanisms, and time series forecasting.

!!! tip "What You'll Find Here"
    Each layer includes detailed documentation with:
    - ✨ **Complete parameter descriptions** with types and defaults
    - 🎯 **Usage examples** showing real-world applications
    - ⚡ **Best practices** and performance considerations
    - 🎨 **When to use** guidance for each layer
    - 🔧 **Implementation notes** for developers

!!! success "Modular & Composable"
    These layers can be combined together to create complex neural network architectures tailored to your specific needs.

!!! note "Keras 3 Compatible"
    All layers are built on top of Keras base classes and are fully compatible with Keras 3.

## ⏱️ Time Series & Forecasting

### 📍 PositionalEmbedding
Fixed sinusoidal positional encoding for transformers and sequence models.

::: kerasfactory.layers.PositionalEmbedding

### 🔧 FixedEmbedding
Non-trainable sinusoidal embeddings for discrete indices (months, days, hours, etc.).

::: kerasfactory.layers.FixedEmbedding

### 🎫 TokenEmbedding
1D convolution-based embedding layer for time series values.

::: kerasfactory.layers.TokenEmbedding

### ⏰ TemporalEmbedding
Embedding layer for temporal features (month, day, weekday, hour, minute).

::: kerasfactory.layers.TemporalEmbedding

### 🎯 DataEmbeddingWithoutPosition
Combined token and temporal embedding layer for comprehensive feature representation.

::: kerasfactory.layers.DataEmbeddingWithoutPosition

### 🏃 MovingAverage
Trend extraction layer using moving average filtering for time series.

::: kerasfactory.layers.MovingAverage

### 🔀 SeriesDecomposition
Trend-seasonal decomposition using moving average.

::: kerasfactory.layers.SeriesDecomposition

### 📊 DFTSeriesDecomposition
Frequency-based series decomposition using Discrete Fourier Transform.

::: kerasfactory.layers.DFTSeriesDecomposition

### 🔄 ReversibleInstanceNorm
Reversible instance normalization with optional denormalization for time series.

::: kerasfactory.layers.ReversibleInstanceNorm

### 🏗️ ReversibleInstanceNormMultivariate
Multivariate version of reversible instance normalization.

::: kerasfactory.layers.ReversibleInstanceNormMultivariate

### 🌊 MultiScaleSeasonMixing
Bottom-up multi-scale seasonal pattern mixing.

::: kerasfactory.layers.MultiScaleSeasonMixing

### 📈 MultiScaleTrendMixing
Top-down multi-scale trend pattern mixing.

::: kerasfactory.layers.MultiScaleTrendMixing

### 🔀 PastDecomposableMixing
Past decomposable mixing encoder block combining decomposition and multi-scale mixing.

::: kerasfactory.layers.PastDecomposableMixing

### ⏱️ TemporalMixing
MLP-based temporal mixing layer for TSMixer that applies transformations across the time dimension.

::: kerasfactory.layers.TemporalMixing

### 🔀 FeatureMixing
Feed-forward network mixing layer for TSMixer that learns cross-series correlations across feature dimension.

::: kerasfactory.layers.FeatureMixing

### 🔀 MixingLayer
Core mixing block combining TemporalMixing and FeatureMixing for the TSMixer architecture.

::: kerasfactory.layers.MixingLayer

## 🎯 Feature Selection & Gating

### 🔀 VariableSelection
Dynamic feature selection using gated residual networks with optional context conditioning.

::: kerasfactory.layers.VariableSelection

### 🚪 GatedFeatureSelection
Feature selection layer using gating mechanisms for conditional feature routing.

::: kerasfactory.layers.GatedFeatureSelection

### 🌊 GatedFeatureFusion
Combines and fuses features using gated mechanisms for adaptive feature integration.

::: kerasfactory.layers.GatedFeatureFusion

### 📍 GatedLinearUnit
Gated linear transformation for controlling information flow in neural networks.

::: kerasfactory.layers.GatedLinearUnit

### 🔗 GatedResidualNetwork
Gated residual network architecture for feature processing with residual connections.

::: kerasfactory.layers.GatedResidualNetwork

## 👁️ Attention Mechanisms

### 🎯 TabularAttention
Dual attention mechanism for tabular data with inter-feature and inter-sample attention.

::: kerasfactory.layers.TabularAttention

### 📊 MultiResolutionTabularAttention
Multi-resolution attention mechanism for capturing features at different scales.

::: kerasfactory.layers.MultiResolutionTabularAttention

### 🔍 InterpretableMultiHeadAttention
Interpretable multi-head attention layer with explainability features.

::: kerasfactory.layers.InterpretableMultiHeadAttention

### 🧠 TransformerBlock
Complete transformer block combining self-attention and feed-forward networks.

::: kerasfactory.layers.TransformerBlock

### 📌 ColumnAttention
Attention mechanism focused on inter-column (feature) relationships.

::: kerasfactory.layers.ColumnAttention

### 📍 RowAttention
Attention mechanism focused on inter-row (sample) relationships.

::: kerasfactory.layers.RowAttention

## 📊 Data Preprocessing & Transformation

### 🔄 DistributionTransformLayer
Transforms data distributions (log, Box-Cox, Yeo-Johnson, etc.) for improved analysis.

::: kerasfactory.layers.DistributionTransformLayer

### 🎓 DistributionAwareEncoder
Encodes features while accounting for their underlying distributions.

::: kerasfactory.layers.DistributionAwareEncoder

### 📈 AdvancedNumericalEmbedding
Advanced numerical embedding layer for rich feature representations.

::: kerasfactory.layers.AdvancedNumericalEmbedding

### 📅 DateParsingLayer
Parses and processes date/time features.

::: kerasfactory.layers.DateParsingLayer

### 🕐 DateEncodingLayer
Encodes dates into learnable embeddings for temporal features.

::: kerasfactory.layers.DateEncodingLayer

### 🌙 SeasonLayer
Extracts and processes seasonal patterns from temporal data.

::: kerasfactory.layers.SeasonLayer

### 🔀 DifferentialPreprocessingLayer
Applies differential preprocessing transformations to features.

::: kerasfactory.layers.DifferentialPreprocessingLayer

### 🔧 DifferentiableTabularPreprocessor
Differentiable preprocessing layer for tabular data end-to-end training.

::: kerasfactory.layers.DifferentiableTabularPreprocessor

### 🎨 CastToFloat32Layer
Type casting layer for ensuring float32 precision.

::: kerasfactory.layers.CastToFloat32Layer

## 🌐 Graph & Ensemble Methods

### 📊 GraphFeatureAggregation
Aggregates features from graph structures for relational learning.

::: kerasfactory.layers.GraphFeatureAggregation

### 🧬 AdvancedGraphFeatureLayer
Advanced graph feature processing with multi-hop aggregation.

::: kerasfactory.layers.AdvancedGraphFeatureLayer

### 👥 MultiHeadGraphFeaturePreprocessor
Multi-head preprocessing for graph features with parallel aggregation.

::: kerasfactory.layers.MultiHeadGraphFeaturePreprocessor

### 📈 BoostingBlock
Boosting ensemble block for combining weak learners.

::: kerasfactory.layers.BoostingBlock

### 🎯 BoostingEnsembleLayer
Ensemble layer implementing gradient boosting mechanisms.

::: kerasfactory.layers.BoostingEnsembleLayer

### 📊 TabularMoELayer
Mixture of Experts layer optimized for tabular data.

::: kerasfactory.layers.TabularMoELayer

### 🏗️ BusinessRulesLayer
Layer for integrating domain-specific business rules into model.

::: kerasfactory.layers.BusinessRulesLayer

## 🛡️ Regularization & Robustness

### 🎲 StochasticDepth
Stochastic depth regularization for improved generalization.

::: kerasfactory.layers.StochasticDepth

### 🗑️ FeatureCutout
Feature cutout regularization for dropout-like effects on features.

::: kerasfactory.layers.FeatureCutout

### 🎯 SparseAttentionWeighting
Sparse attention weighting for computational efficiency.

::: kerasfactory.layers.SparseAttentionWeighting

## 🔧 Specialized Processing

### 🐢 SlowNetwork
Slow network layer for temporal smoothing and stability.

::: kerasfactory.layers.SlowNetwork

### ⚡ HyperZZWOperator
Specialized hyperparameter operator for advanced transformations.

::: kerasfactory.layers.HyperZZWOperator

## 🚨 Anomaly Detection

### 📉 NumericalAnomalyDetection
Detects anomalies in numerical features using statistical methods.

::: kerasfactory.layers.NumericalAnomalyDetection

### 📊 CategoricalAnomalyDetectionLayer
Detects anomalies in categorical features.

::: kerasfactory.layers.CategoricalAnomalyDetectionLayer
