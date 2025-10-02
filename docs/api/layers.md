# 🧩 Layers API Reference

Welcome to the KMR Layers documentation! All layers are designed to work exclusively with **Keras 3** and follow consistent patterns for easy integration.

!!! tip "Quick Navigation"
    - 🎯 **Most Popular**: Start with TabularAttention, DistributionTransformLayer, GatedFeatureFusion
    - 🔧 **Feature Engineering**: DateEncodingLayer, VariableSelection, BusinessRulesLayer  
    - 🧠 **Advanced**: AdvancedNumericalEmbedding, TransformerBlock, StochasticDepth

!!! success "Keras 3 Native"
    All layers are built exclusively for Keras 3 with no TensorFlow dependencies in production code.

## 📚 Layer Categories

=== "🎯 Core Layers"
    Essential layers for building tabular models with attention mechanisms and feature processing.

=== "🔧 Feature Engineering"
    Layers for data preprocessing, transformation, and feature engineering tasks.

=== "🧠 Attention Mechanisms"
    Advanced attention layers for capturing complex feature relationships.

=== "🏗️ Specialized Layers"
    Specialized layers for specific use cases like anomaly detection and boosting.

## 🎯 Core Layers

### 🧠 TabularAttention
Dual attention mechanism for inter-feature and inter-sample relationships in tabular data.

::: kmr.layers.TabularAttention

### 🔢 AdvancedNumericalEmbedding
Advanced numerical feature embedding with dual-branch architecture (continuous + discrete).

::: kmr.layers.AdvancedNumericalEmbedding

### 🔀 GatedFeatureFusion
Gated mechanism for intelligently fusing multiple feature representations.

::: kmr.layers.GatedFeatureFusion

### 🎯 VariableSelection
Intelligent variable selection network for identifying important features.

::: kmr.layers.VariableSelection

### 🔄 TransformerBlock
Standard transformer block with multi-head attention and feed-forward networks.

::: kmr.layers.TransformerBlock

### 🎲 StochasticDepth
Stochastic depth regularization for improved training and generalization.

::: kmr.layers.StochasticDepth

## 🔧 Feature Engineering Layers

### 📊 DistributionTransformLayer
Automatic distribution transformation for numerical features to improve model performance.

::: kmr.layers.DistributionTransformLayer

### 📅 DateEncodingLayer
Comprehensive date and time feature encoding with multiple temporal representations.

::: kmr.layers.DateEncodingLayer

### 🔍 DateParsingLayer
Flexible date parsing and extraction from various date formats and strings.

::: kmr.layers.DateParsingLayer

### 🌸 SeasonLayer
Seasonal feature extraction from date/time data for temporal pattern recognition.

::: kmr.layers.SeasonLayer

## 🧠 Attention Mechanisms

### 📊 ColumnAttention
Column-wise attention for tabular data to capture feature-level relationships.

::: kmr.layers.ColumnAttention

### 📋 RowAttention
Row-wise attention mechanisms for sample-level pattern recognition.

::: kmr.layers.RowAttention

### 🔍 InterpretableMultiHeadAttention
Interpretable multi-head attention with attention weight analysis and visualization.

::: kmr.layers.InterpretableMultiHeadAttention

### 🎯 MultiResolutionTabularAttention
Multi-resolution attention for different feature scales and granularities.

::: kmr.layers.MultiResolutionTabularAttention

## 🔀 Gated Networks

### ⚡ GatedLinearUnit
Gated linear unit for intelligent feature gating and selective information flow.

::: kmr.layers.GatedLinearUnit

### 🔄 GatedResidualNetwork
Gated residual network for complex feature interactions and gradient flow.

::: kmr.layers.GatedResidualNetwork

### 🎯 GatedFeaturesSelection
Gated feature selection mechanism for adaptive feature importance weighting.

::: kmr.layers.GatedFeaturesSelection

## 🚀 Boosting Layers

### 📈 BoostingBlock
Gradient boosting inspired neural network block for sequential learning.

::: kmr.layers.BoostingBlock

### 🎯 BoostingEnsembleLayer
Ensemble of boosting blocks for improved performance and robustness.

::: kmr.layers.BoostingEnsembleLayer

## 🏗️ Specialized Layers

### 📋 BusinessRulesLayer
Integration of business rules and domain knowledge into neural networks.

::: kmr.layers.BusinessRulesLayer

### 🔍 NumericalAnomalyDetection
Anomaly detection for numerical features using statistical and ML methods.

::: kmr.layers.NumericalAnomalyDetection

### 🏷️ CategoricalAnomalyDetectionLayer
Anomaly detection for categorical features with pattern recognition.

::: kmr.layers.CategoricalAnomalyDetectionLayer

### ✂️ FeatureCutout
Feature cutout for data augmentation and regularization in tabular data.

::: kmr.layers.FeatureCutout

### 🎯 SparseAttentionWeighting
Sparse attention weighting mechanisms for efficient computation.

::: kmr.layers.SparseAttentionWeighting

### 🎭 TabularMoELayer
Mixture of Experts for tabular data with adaptive expert selection.

::: kmr.layers.TabularMoELayer

## 🔧 Utility Layers

### 🔢 CastToFloat32Layer
Type casting utility layer for ensuring consistent data types.

::: kmr.layers.CastToFloat32Layer

### ⚙️ DifferentiableTabularPreprocessor
Differentiable preprocessing for tabular data with gradient flow.

::: kmr.layers.DifferentiableTabularPreprocessor

### 🔄 DifferentialPreprocessingLayer
Differential preprocessing operations for advanced data transformations.

::: kmr.layers.DifferentialPreprocessingLayer

### 📊 DistributionAwareEncoder
Distribution-aware feature encoding for optimal representation learning.

::: kmr.layers.DistributionAwareEncoder

### 🎛️ HyperZZWOperator
Hyperparameter-aware operator for adaptive model behavior.

::: kmr.layers.HyperZZWOperator

### 🐌 SlowNetwork
Slow network architecture for careful and deliberate feature processing.

::: kmr.layers.SlowNetwork

### 📝 TextPreprocessingLayer
Text preprocessing utilities for natural language features in tabular data.

::: kmr.layers.TextPreprocessingLayer

## 🕸️ Graph and Advanced Features

### 🧠 AdvancedGraphFeature
Advanced graph-based feature processing with dynamic adjacency learning.

::: kmr.layers.AdvancedGraphFeature

### 🔗 GraphFeatureAggregation
Graph feature aggregation mechanisms for relationship modeling.

::: kmr.layers.GraphFeatureAggregation

### 🎯 MultiHeadGraphFeaturePreprocessor
Multi-head graph feature preprocessing for complex feature interactions.

::: kmr.layers.MultiHeadGraphFeaturePreprocessor
