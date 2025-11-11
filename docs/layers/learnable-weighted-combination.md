---
title: LearnableWeightedCombination - KMR
description: Combine multiple recommendation scores with learnable softmax-normalized weights for hybrid recommendations
keywords: [score combination, learnable weights, ensemble, recommendation, keras, hybrid, weighted combination]
---

# Learnable Weighted Combination

<div class="layer-hero">
  <div class="layer-hero-content">
    <h1>Learnable Weighted Combination</h1>
    <div class="layer-badges">
      <span class="badge badge-advanced">Advanced</span>
      <span class="badge badge-stable">Stable</span>
      <span class="badge badge-recommendation">Recommendation</span>
    </div>
  </div>
</div>

## Overview

The `LearnableWeightedCombination` layer combines multiple recommendation scores using learnable, softmax-normalized weights. It learns optimal weights for blending different recommendation signals during training, enabling intelligent hybrid recommendation systems.

This layer is crucial for hybrid recommendation systems where multiple recommendation approaches (collaborative filtering, content-based, deep learning, etc.) need to be combined intelligently. The learnable weights adapt to the data, finding the best combination automatically.

## How It Works

The layer combines scores with learned weights:

1. Input Scores: Multiple score tensors (list of (batch_size, num_items))
2. Weight Learning: Learnable weights for each score component
3. Softmax Normalization: Normalize weights to sum to 1
4. Weighted Combination: Combine scores using learned weights
5. Output Combined Score: (batch_size, num_items) combined score

The weights are learned during training, automatically finding the optimal combination of different recommendation signals.

## Why Use This Layer?

| Challenge | Traditional Approach | LearnableWeightedCombination Solution |
|-----------|---------------------|--------------------------------------|
| Score Combination | Fixed weights | Learnable optimal weights |
| Hybrid Systems | Manual weight tuning | Automatic weight learning |
| Signal Fusion | Simple averaging | Intelligent weighted fusion |
| Adaptation | Static combinations | Data-driven adaptation |
| Optimization | Manual optimization | End-to-end learning |

## Use Cases

- Hybrid Recommendations: Blending CF, CB, and other approaches
- Multi-Signal Fusion: Combining multiple scoring signals
- Ensemble Learning: Ensemble of different recommendation models
- Adaptive Ranking: Learn to adapt weights dynamically
- Multi-Modal Recommendations: Combine different data modalities
- A/B Testing: Evaluate different combination strategies

## Quick Start

### Basic Usage

```python
import keras
from kmr.layers import LearnableWeightedCombination

# Create combination layer
combiner = LearnableWeightedCombination(num_scores=3)

# Combine three scores
score1 = keras.random.normal((32, 100))  # CF score
score2 = keras.random.normal((32, 100))  # CB score
score3 = keras.random.normal((32, 100))  # Deep score

combined = combiner([score1, score2, score3])
print(f"Combined score: {combined.shape}")  # (32, 100)
```

### In Hybrid Recommendation Pipeline

```python
import keras
from kmr.layers import (
    CollaborativeUserItemEmbedding,
    DeepFeatureTower,
    NormalizedDotProductSimilarity,
    LearnableWeightedCombination,
    TopKRecommendationSelector
)

# Define inputs
user_id_input = keras.Input(shape=(1,), dtype='int32')
user_features_input = keras.Input(shape=(20,), dtype='float32')
item_id_input = keras.Input(shape=(100,), dtype='int32')
item_features_input = keras.Input(shape=(100, 15), dtype='float32')

# Collaborative Filtering score
embedding = CollaborativeUserItemEmbedding(1000, 5000, 32)
user_emb, item_emb = embedding([user_id_input, item_id_input])
cf_similarity = NormalizedDotProductSimilarity()
cf_score = cf_similarity([keras.ops.expand_dims(user_emb, 1), item_emb])

# Content-Based score
user_tower = DeepFeatureTower(units=32, hidden_layers=2)
item_tower = DeepFeatureTower(units=32, hidden_layers=2)
user_repr = user_tower(user_features_input)
item_flat = keras.ops.reshape(item_features_input, (-1, 15))
item_repr_flat = item_tower(item_flat)
item_repr = keras.ops.reshape(item_repr_flat, (keras.ops.shape(item_features_input)[0], 100, 32))
cb_similarity = NormalizedDotProductSimilarity()
cb_score = cb_similarity([keras.ops.expand_dims(user_repr, 1), item_repr])

# Combine scores
combiner = LearnableWeightedCombination(num_scores=2)
combined_score = combiner([cf_score, cb_score])

# Select top-K
selector = TopKRecommendationSelector(k=10)
rec_indices, rec_scores = selector(combined_score)

# Build hybrid model
model = keras.Model(
    inputs=[user_id_input, user_features_input, item_id_input, item_features_input],
    outputs=[rec_indices, rec_scores]
)
```

## API Reference

::: kmr.layers.LearnableWeightedCombination

## Parameters Deep Dive

### num_scores (int)
- Purpose: Number of score components to combine
- Range: 2 to 10 (typically 2-5)
- Impact: Determines number of learnable weights
- Recommendation: Start with 2-3 scores, add more if needed

## Performance Characteristics

- Speed: Fast - O(batch × items × num_scores)
- Memory: Minimal - only stores weight parameters
- Accuracy: Excellent - learns optimal combination
- Scalability: Good for multiple score components
- Learning: Adapts weights during training

## Examples

### Example 1: Basic Score Combination

```python
import keras
from kmr.layers import LearnableWeightedCombination

combiner = LearnableWeightedCombination(num_scores=3)

# Three different scoring signals
cf_score = keras.random.normal((16, 50))      # Collaborative filtering
cb_score = keras.random.normal((16, 50))      # Content-based
deep_score = keras.random.normal((16, 50))    # Deep learning

combined = combiner([cf_score, cb_score, deep_score])
print(f"Combined scores: {combined.shape}")
```

### Example 2: Weight Analysis

```python
import keras
from kmr.layers import LearnableWeightedCombination

combiner = LearnableWeightedCombination(num_scores=3)

# Test scores
scores = [keras.random.normal((8, 100)) for _ in range(3)]
combined = combiner(scores)

# Check learned weights (after training)
weights = combiner.combination_weights
normalized = keras.ops.softmax(weights)
print(f"Learned weights: {normalized.numpy()}")
```

### Example 3: Hybrid System

```python
import keras
from kmr.layers import (
    LearnableWeightedCombination,
    TopKRecommendationSelector
)

combiner = LearnableWeightedCombination(num_scores=3)

# Simulate three recommendation approaches
cf_scores = keras.random.normal((32, 1000))    # Collaborative filtering
cb_scores = keras.random.normal((32, 1000))    # Content-based
geo_scores = keras.random.normal((32, 1000))   # Geospatial

# Combine
combined = combiner([cf_scores, cb_scores, geo_scores])

# Select top-K
selector = TopKRecommendationSelector(k=10)
indices, final_scores = selector(combined)

print(f"Hybrid recommendations: {indices.shape}")
```

## Tips and Best Practices

- Score Normalization: Normalize input scores to similar ranges
- Number of Scores: Start with 2-3, add more if beneficial
- Weight Initialization: Layer uses reasonable default initialization
- Training: Let weights learn during training
- Evaluation: Monitor individual score contributions
- Regularization: Consider L2 regularization on weights if overfitting

## Common Pitfalls

- Score Range Mismatch: Different score ranges can bias weights
- Too Many Scores: Too many components can be hard to learn
- Weight Interpretation: Weights show relative importance
- Cold Start: New score types may need weight adjustment
- Overfitting: Monitor for overfitting with many scores
- Batch Size: Ensure consistent batch sizes across scores

## Related Layers

- NormalizedDotProductSimilarity - Generate similarity scores
- DeepFeatureRanking - Generate deep ranking scores
- TopKRecommendationSelector - Select final recommendations
- CollaborativeUserItemEmbedding - CF score component
- DeepFeatureTower - CB score component

## Further Reading

- Ensemble Learning - Ensemble methods overview
- Hybrid Recommendations - Hybrid RS approaches
- Weighted Combination - Score fusion techniques
- Multi-Modal Learning - Combining multiple signals
