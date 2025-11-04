---
title: DeepFeatureRanking - KMR
description: Deep neural network tower for feature-based ranking in recommendation systems
keywords: [deep ranking, neural ranking, feature ranking, recommendation, learning to rank, keras]
---

# Deep Feature Ranking

<div class="layer-hero">
  <div class="layer-hero-content">
    <h1>Deep Feature Ranking</h1>
    <div class="layer-badges">
      <span class="badge badge-intermediate">Intermediate</span>
      <span class="badge badge-stable">Stable</span>
      <span class="badge badge-recommendation">Recommendation</span>
    </div>
  </div>
</div>

## Overview

The `DeepFeatureRanking` layer implements a deep neural network tower for feature-based ranking. It processes combined user-item features through multiple dense layers with batch normalization and dropout to produce ranking scores.

This layer is essential for learning-to-rank models in recommendation systems, enabling complex non-linear ranking functions based on user-item feature combinations. It learns sophisticated patterns that simpler similarity-based approaches cannot capture.

## How It Works

The layer processes combined features through a deep network:

1. Input Features: Combined user-item features (batch_size, num_items, feature_dim)
2. Dense Layers: Multiple dense layers with configurable activations
3. Batch Normalization: Normalizes activations for training stability
4. Dropout: Regularization to prevent overfitting
5. Output Layer: Final dense layer producing ranking scores
6. Output Scores: (batch_size, num_items, 1) ranking scores

Each hidden layer applies: Dense → BatchNorm → Dropout → Activation.

## Why Use This Layer?

| Challenge | Traditional Approach | DeepFeatureRanking Solution |
|-----------|---------------------|----------------------------|
| Complex Patterns | Linear similarity functions | Non-linear deep learning patterns |
| Feature Combination | Manual feature engineering | Automatic feature learning |
| Ranking Optimization | Pointwise loss functions | Pairwise/listwise ranking optimization |
| Scalability | Hand-crafted rules | End-to-end learnable ranking |
| Flexibility | Fixed scoring functions | Adaptive complex scoring functions |

## Use Cases

- Learning-to-Rank: Deep ranking models for recommendations
- Feature-Based Ranking: Combine user-item features for scoring
- Complex Scoring: Learn non-linear ranking functions
- Ranking Optimization: Optimize for ranking-specific metrics
- Hybrid Recommendations: Combine multiple signals for ranking
- Personalized Ranking: Learn user-specific ranking preferences

## Quick Start

### Basic Usage

```python
import keras
from kmr.layers import DeepFeatureRanking

# Create ranking tower
ranker = DeepFeatureRanking(
    hidden_units=64,
    num_layers=2,
    dropout_rate=0.2,
    activation='relu'
)

# Combined user-item features
features = keras.random.normal((32, 50, 128))  # (batch, items, features)
scores = ranker(features, training=True)

print(f"Input features: {features.shape}")    # (32, 50, 128)
print(f"Ranking scores: {scores.shape}")      # (32, 50, 1)
```

### In a Complete Recommendation Pipeline

```python
import keras
from kmr.layers import DeepFeatureRanking, TopKRecommendationSelector

# Define inputs
user_features_input = keras.Input(shape=(20,), name='user_features')
item_features_input = keras.Input(shape=(50, 15), name='item_features')

# Combine features
batch_size = keras.ops.shape(item_features_input)[0]
num_items = keras.ops.shape(item_features_input)[1]

# Expand and tile user features
user_exp = keras.ops.expand_dims(user_features_input, axis=1)
user_tiled = keras.ops.tile(user_exp, (1, num_items, 1))

# Concatenate user and item features
combined = keras.ops.concatenate([user_tiled, item_features_input], axis=-1)

# Reshape for ranking tower
combined_flat = keras.ops.reshape(combined, (-1, 35))  # 20 + 15 features

# Apply ranking tower
ranker = DeepFeatureRanking(hidden_units=64, num_layers=2)
scores_flat = ranker(combined_flat)

# Reshape back
scores = keras.ops.reshape(scores_flat, (batch_size, num_items, 1))
scores = keras.ops.squeeze(scores, axis=-1)

# Select top-K
selector = TopKRecommendationSelector(k=10)
rec_indices, rec_scores = selector(scores)

# Build model
model = keras.Model(
    inputs=[user_features_input, item_features_input],
    outputs=[rec_indices, rec_scores]
)
```

## API Reference

::: kmr.layers.DeepFeatureRanking

## Parameters Deep Dive

### hidden_units (int)
- Purpose: Number of units in each hidden layer
- Range: 16 to 512 (typically 32-128)
- Impact: Model capacity and expressiveness
- Recommendation: Start with 64, adjust based on data complexity

### num_layers (int)
- Purpose: Number of hidden layers in the ranking tower
- Range: 1 to 10 (typically 2-4)
- Impact: Model depth and capacity
- Recommendation: 2-3 layers for balanced complexity

### dropout_rate (float)
- Purpose: Fraction of inputs to drop during training
- Range: 0.0 to 0.5 (typically 0.2-0.3)
- Default: 0.2
- Recommendation: Increase for overfitting, decrease for underfitting

### activation (str)
- Purpose: Activation function for hidden layers
- Options: 'relu', 'tanh', 'sigmoid', 'elu', 'selu'
- Default: 'relu'
- Recommendation: 'relu' for most cases

### l2_reg (float)
- Purpose: L2 regularization strength
- Range: 0.0 to 0.1
- Default: 1e-4
- Tip: Balance with dropout for regularization

## Performance Characteristics

- Speed: Moderate - depends on number of layers and units
- Memory: Scales with hidden_units × num_layers × feature_dim
- Accuracy: Excellent for complex ranking patterns
- Capacity: Can learn sophisticated non-linear ranking functions
- Training: Requires careful tuning of regularization

## Examples

### Example 1: Basic Feature Ranking

```python
import keras
from kmr.layers import DeepFeatureRanking

# Create ranking tower
ranker = DeepFeatureRanking(
    hidden_units=64,
    num_layers=2,
    dropout_rate=0.2
)

# Combined features
features = keras.random.normal((16, 100, 128))
scores = ranker(features)

print(f"Features: {features.shape}")      # (16, 100, 128)
print(f"Scores: {scores.shape}")         # (16, 100, 1)
```

### Example 2: Different Tower Depths

```python
import keras
from kmr.layers import DeepFeatureRanking

# Create towers with different depths
shallow = DeepFeatureRanking(hidden_units=64, num_layers=1)
medium = DeepFeatureRanking(hidden_units=64, num_layers=2)
deep = DeepFeatureRanking(hidden_units=64, num_layers=4)

# Test data
features = keras.random.normal((32, 50, 128))

# Process
shallow_scores = shallow(features)
medium_scores = medium(features)
deep_scores = deep(features)

print(f"Shallow: {shallow_scores.shape}")
print(f"Medium: {medium_scores.shape}")
print(f"Deep: {deep_scores.shape}")
```

### Example 3: Ranking with User-Item Features

```python
import keras
from kmr.layers import DeepFeatureRanking, TopKRecommendationSelector

# User features (age, income, interests, etc.)
user_features = keras.random.normal((32, 20))

# Item features (category, price, rating, etc.)
item_features = keras.random.normal((32, 100, 15))

# Combine features
user_exp = keras.ops.expand_dims(user_features, axis=1)
user_tiled = keras.ops.tile(user_exp, (1, 100, 1))
combined = keras.ops.concatenate([user_tiled, item_features], axis=-1)

# Reshape for ranking
combined_flat = keras.ops.reshape(combined, (-1, 35))
ranker = DeepFeatureRanking(hidden_units=64, num_layers=2)
scores_flat = ranker(combined_flat)
scores = keras.ops.reshape(scores_flat, (32, 100, 1))
scores = keras.ops.squeeze(scores, axis=-1)

# Select top-K
selector = TopKRecommendationSelector(k=10)
indices, rec_scores = selector(scores)

print(f"Top-10 recommendations: {indices.shape}")
```

## Tips and Best Practices

- Hidden Units: Start with 64, increase if underfitting
- Layers: 2-3 layers usually sufficient; avoid too deep (>5)
- Dropout: Use 0.2-0.3 for regularization
- Feature Engineering: Pre-normalize features for better convergence
- Training Mode: Always set training=True during training
- Regularization: Balance dropout with L2 for best results
- Loss Function: Use ranking-specific losses (pairwise/listwise)

## Common Pitfalls

- Overfitting: Deep networks can easily overfit; use regularization
- Feature Normalization: Unnormalized features can cause training issues
- Training Mode: Forgetting training=True causes incorrect dropout behavior
- Too Deep: Very deep networks (>5 layers) can be hard to train
- Memory: Large hidden_units can consume significant memory
- Feature Dimension: Ensure combined feature dimension matches input

## Related Layers

- DeepFeatureTower - Process user/item features separately
- TopKRecommendationSelector - Select top-K based on scores
- NormalizedDotProductSimilarity - Alternative similarity-based ranking
- LearnableWeightedCombination - Combine multiple ranking signals

## Further Reading

- Learning to Rank - Overview of ranking approaches
- Deep Learning for Ranking - Neural ranking methods
- Feature Engineering - Feature combination techniques
- Ranking Metrics - NDCG, MAP, MRR evaluation
