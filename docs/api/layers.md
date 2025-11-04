# 🧩 Layers API Reference

Welcome to the KMR Layers documentation! All layers are designed to work exclusively with **Keras 3** and provide specialized implementations for advanced tabular data processing, feature engineering, attention mechanisms, and time series forecasting.

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

::: kmr.layers.PositionalEmbedding

### 🔧 FixedEmbedding
Non-trainable sinusoidal embeddings for discrete indices (months, days, hours, etc.).

::: kmr.layers.FixedEmbedding

### 🎫 TokenEmbedding
1D convolution-based embedding layer for time series values.

::: kmr.layers.TokenEmbedding

### ⏰ TemporalEmbedding
Embedding layer for temporal features (month, day, weekday, hour, minute).

::: kmr.layers.TemporalEmbedding

### 🎯 DataEmbeddingWithoutPosition
Combined token and temporal embedding layer for comprehensive feature representation.

::: kmr.layers.DataEmbeddingWithoutPosition

### 🏃 MovingAverage
Trend extraction layer using moving average filtering for time series.

::: kmr.layers.MovingAverage

### 🔀 SeriesDecomposition
Trend-seasonal decomposition using moving average.

::: kmr.layers.SeriesDecomposition

### 📊 DFTSeriesDecomposition
Frequency-based series decomposition using Discrete Fourier Transform.

::: kmr.layers.DFTSeriesDecomposition

### 🔄 ReversibleInstanceNorm
Reversible instance normalization with optional denormalization for time series.

::: kmr.layers.ReversibleInstanceNorm

### 🏗️ ReversibleInstanceNormMultivariate
Multivariate version of reversible instance normalization.

::: kmr.layers.ReversibleInstanceNormMultivariate

### 🌊 MultiScaleSeasonMixing
Bottom-up multi-scale seasonal pattern mixing.

::: kmr.layers.MultiScaleSeasonMixing

### 📈 MultiScaleTrendMixing
Top-down multi-scale trend pattern mixing.

::: kmr.layers.MultiScaleTrendMixing

### 🔀 PastDecomposableMixing
Past decomposable mixing encoder block combining decomposition and multi-scale mixing.

::: kmr.layers.PastDecomposableMixing

## 🎯 Feature Selection & Gating

### 🔀 VariableSelection
Dynamic feature selection using gated residual networks with optional context conditioning.

::: kmr.layers.VariableSelection

### 🚪 GatedFeatureSelection
Feature selection layer using gating mechanisms for conditional feature routing.

::: kmr.layers.GatedFeatureSelection

### 🌊 GatedFeatureFusion
Combines and fuses features using gated mechanisms for adaptive feature integration.

::: kmr.layers.GatedFeatureFusion

### 📍 GatedLinearUnit
Gated linear transformation for controlling information flow in neural networks.

::: kmr.layers.GatedLinearUnit

### 🔗 GatedResidualNetwork
Gated residual network architecture for feature processing with residual connections.

::: kmr.layers.GatedResidualNetwork

## 👁️ Attention Mechanisms

### 🎯 TabularAttention
Dual attention mechanism for tabular data with inter-feature and inter-sample attention.

::: kmr.layers.TabularAttention

### 📊 MultiResolutionTabularAttention
Multi-resolution attention mechanism for capturing features at different scales.

::: kmr.layers.MultiResolutionTabularAttention

### 🔍 InterpretableMultiHeadAttention
Interpretable multi-head attention layer with explainability features.

::: kmr.layers.InterpretableMultiHeadAttention

### 🧠 TransformerBlock
Complete transformer block combining self-attention and feed-forward networks.

::: kmr.layers.TransformerBlock

### 📌 ColumnAttention
Attention mechanism focused on inter-column (feature) relationships.

::: kmr.layers.ColumnAttention

### 📍 RowAttention
Attention mechanism focused on inter-row (sample) relationships.

::: kmr.layers.RowAttention

## 📊 Data Preprocessing & Transformation

### 🔄 DistributionTransformLayer
Transforms data distributions (log, Box-Cox, Yeo-Johnson, etc.) for improved analysis.

::: kmr.layers.DistributionTransformLayer

### 🎓 DistributionAwareEncoder
Encodes features while accounting for their underlying distributions.

::: kmr.layers.DistributionAwareEncoder

### 📈 AdvancedNumericalEmbedding
Advanced numerical embedding layer for rich feature representations.

::: kmr.layers.AdvancedNumericalEmbedding

### 📅 DateParsingLayer
Parses and processes date/time features.

::: kmr.layers.DateParsingLayer

### 🕐 DateEncodingLayer
Encodes dates into learnable embeddings for temporal features.

::: kmr.layers.DateEncodingLayer

### 🌙 SeasonLayer
Extracts and processes seasonal patterns from temporal data.

::: kmr.layers.SeasonLayer

### 🔀 DifferentialPreprocessingLayer
Applies differential preprocessing transformations to features.

::: kmr.layers.DifferentialPreprocessingLayer

### 🔧 DifferentiableTabularPreprocessor
Differentiable preprocessing layer for tabular data end-to-end training.

::: kmr.layers.DifferentiableTabularPreprocessor

### 🎨 CastToFloat32Layer
Type casting layer for ensuring float32 precision.

::: kmr.layers.CastToFloat32Layer

## 🌐 Graph & Ensemble Methods

### 📊 GraphFeatureAggregation
Aggregates features from graph structures for relational learning.

::: kmr.layers.GraphFeatureAggregation

### 🧬 AdvancedGraphFeatureLayer
Advanced graph feature processing with multi-hop aggregation.

::: kmr.layers.AdvancedGraphFeatureLayer

### 👥 MultiHeadGraphFeaturePreprocessor
Multi-head preprocessing for graph features with parallel aggregation.

::: kmr.layers.MultiHeadGraphFeaturePreprocessor

### 📈 BoostingBlock
Boosting ensemble block for combining weak learners.

::: kmr.layers.BoostingBlock

### 🎯 BoostingEnsembleLayer
Ensemble layer implementing gradient boosting mechanisms.

::: kmr.layers.BoostingEnsembleLayer

### 📊 TabularMoELayer
Mixture of Experts layer optimized for tabular data.

::: kmr.layers.TabularMoELayer

### 🏗️ BusinessRulesLayer
Layer for integrating domain-specific business rules into model.

::: kmr.layers.BusinessRulesLayer

## 🛡️ Regularization & Robustness

### 🎲 StochasticDepth
Stochastic depth regularization for improved generalization.

::: kmr.layers.StochasticDepth

### 🗑️ FeatureCutout
Feature cutout regularization for dropout-like effects on features.

::: kmr.layers.FeatureCutout

### 🎯 SparseAttentionWeighting
Sparse attention weighting for computational efficiency.

::: kmr.layers.SparseAttentionWeighting

## 🔧 Specialized Processing

### 🐢 SlowNetwork
Slow network layer for temporal smoothing and stability.

::: kmr.layers.SlowNetwork

### ⚡ HyperZZWOperator
Specialized hyperparameter operator for advanced transformations.

::: kmr.layers.HyperZZWOperator

## 🚨 Anomaly Detection

### 📉 NumericalAnomalyDetection
Detects anomalies in numerical features using statistical methods.

::: kmr.layers.NumericalAnomalyDetection

### 📊 CategoricalAnomalyDetectionLayer
Detects anomalies in categorical features.

::: kmr.layers.CategoricalAnomalyDetectionLayer
