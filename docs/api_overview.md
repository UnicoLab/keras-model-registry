# KMR API Overview

## Available Layers

- **AdvancedGraphFeatureLayer**: Advanced graph-based feature layer for tabular data.

- **AdvancedNumericalEmbedding**: Advanced numerical embedding layer for continuous features.

- **BoostingBlock**: A neural network layer that simulates gradient boosting behavior.

- **BoostingEnsembleLayer**: Ensemble layer of boosting blocks for tabular data.

- **BusinessRulesLayer**: Evaluates business-defined rules for anomaly detection.

- **CastToFloat32Layer**: Layer that casts input tensors to float32 data type.

- **CategoricalAnomalyDetectionLayer**: Backend-agnostic anomaly detection for categorical features.

- **ColumnAttention**: Column attention mechanism to weight features dynamically.

- **DateEncodingLayer**: Layer for encoding date components into cyclical features.

- **DateParsingLayer**: Layer for parsing date strings into numerical components.

- **DifferentiableTabularPreprocessor**: A differentiable preprocessing layer for numeric tabular data.

- **DifferentialPreprocessingLayer**: Differentiable preprocessing layer for numeric tabular data with multiple candidate transformations.

- **DifferentialPreprocssingLayer**: Differentiable preprocessing layer for numeric tabular data with multiple candidate transformations.

- **DistributionAwareEncoder**: Layer that automatically detects and encodes data based on its distribution.

- **DistributionTransformLayer**: Layer for transforming data distributions to improve anomaly detection.

- **FeatureCutout**: Feature cutout regularization layer.

- **GatedFeatureFusion**: Gated feature fusion layer for combining two feature representations.

- **GatedFeatureSelection**: Gated feature selection layer with residual connection.

- **GatedLinearUnit**: GatedLinearUnit is a custom Keras layer that implements a gated linear unit.

- **GatedResidualNetwork**: GatedResidualNetwork is a custom Keras layer that implements a gated residual network.

- **GraphFeatureAggregation**: Graph-based feature aggregation layer with self-attention for tabular data.

- **HyperZZWOperator**: A layer that computes context-dependent weights by multiplying inputs with hyper-kernels.

- **InterpretableMultiHeadAttention**: Interpretable Multi-Head Attention layer.

- **MultiHeadGraphFeaturePreprocessor**: Multi-head graph-based feature preprocessor for tabular data.

- **MultiResolutionTabularAttention**: Custom layer to apply multi-resolution attention for mixed-type tabular data.

- **NumericalAnomalyDetection**: Numerical anomaly detection layer for identifying outliers in numerical features.

- **RowAttention**: Row attention mechanism to weight samples dynamically.

- **SeasonLayer**: Layer for adding seasonal information based on month.

- **SlowNetwork**: A multi-layer network with configurable depth and width.

- **SparseAttentionWeighting**: Sparse attention mechanism with temperature scaling for module outputs combination.

- **StochasticDepth**: Stochastic depth layer for regularization.

- **TabularAttention**: Custom layer to apply inter-feature and inter-sample attention for tabular data.

- **TabularMoELayer**: Mixture-of-Experts layer for tabular data.

- **TransformerBlock**: Transformer block with multi-head attention and feed-forward layers.

- **VariableSelection**: Layer for dynamic feature selection using gated residual networks.


## Available Models

- **SFNEBlock**: Slow-Fast Neural Engine Block for feature processing.

- **TerminatorModel**: Terminator model for advanced feature processing.

- **BaseFeedForwardModel**: Base feed forward neural network model.
