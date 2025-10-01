# Layers API Reference

This page provides comprehensive documentation for all KMR layers. All layers are designed to work exclusively with Keras 3 and follow consistent patterns for initialization, serialization, and usage.

## Core Layers

### TabularAttention
Multi-head attention mechanism specifically designed for tabular data processing.

### AdvancedNumericalEmbedding
Advanced numerical feature embedding with learnable binning and MLP processing.

### GatedFeatureFusion
Gated mechanism for fusing multiple feature representations.

### VariableSelection
Variable selection network for identifying important features.

### TransformerBlock
Standard transformer block with multi-head attention and feed-forward networks.

### StochasticDepth
Stochastic depth regularization for improved training.

## Feature Engineering Layers

### DistributionTransformLayer
Automatic distribution transformation for numerical features.

### DateEncodingLayer
Comprehensive date and time feature encoding.

### DateParsingLayer
Flexible date parsing and extraction.

### SeasonLayer
Seasonal feature extraction from date/time data.

## Attention Mechanisms

### ColumnAttention
Column-wise attention for tabular data.

### RowAttention
Row-wise attention mechanisms.

### InterpretableMultiHeadAttention
Interpretable multi-head attention with attention weight analysis.

### MultiResolutionTabularAttention
Multi-resolution attention for different feature scales.

## Gated Networks

### GatedLinearUnit
Gated linear unit for feature gating.

### GatedResidualNetwork
Gated residual network for complex feature interactions.

### GatedFeaturesSelection
Gated feature selection mechanism.

## Boosting Layers

### BoostingBlock
Gradient boosting inspired neural network block.

### BoostingEnsembleLayer
Ensemble of boosting blocks for improved performance.

## Specialized Layers

### BusinessRulesLayer
Integration of business rules into neural networks.

### NumericalAnomalyDetection
Anomaly detection for numerical features.

### CategoricalAnomalyDetectionLayer
Anomaly detection for categorical features.

### FeatureCutout
Feature cutout for data augmentation.

### SparseAttentionWeighting
Sparse attention weighting mechanisms.

### TabularMoELayer
Mixture of Experts for tabular data.

## Utility Layers

### CastToFloat32Layer
Type casting utility layer.

### DifferentiableTabularPreprocessor
Differentiable preprocessing for tabular data.

### DifferentialPreprocessingLayer
Differential preprocessing operations.

### DistributionAwareEncoder
Distribution-aware feature encoding.

### HyperZZWOperator
Hyperparameter-aware operator.

### SlowNetwork
Slow network architecture for careful feature processing.

### TextPreprocessingLayer
Text preprocessing utilities.

## Graph and Advanced Features

### AdvancedGraphFeature
Advanced graph feature processing.

### GraphFeatureAggregation
Graph feature aggregation mechanisms.

### MultiHeadGraphFeaturePreprocessor
Multi-head graph feature preprocessing.
