---
title: FeedbackAdjustmentLayer - KMR
description: Adjust recommendation scores based on user feedback signals for adaptive recommendations
keywords: [feedback, adjustment, user feedback, recommendation, keras, adaptation, interactive learning]
---

# Feedback Adjustment Layer

<div class="layer-hero">
  <div class="layer-hero-content">
    <h1>Feedback Adjustment Layer</h1>
    <div class="layer-badges">
      <span class="badge badge-advanced">Advanced</span>
      <span class="badge badge-stable">Stable</span>
      <span class="badge badge-recommendation">Recommendation</span>
    </div>
  </div>
</div>

## Overview

The `FeedbackAdjustmentLayer` adjusts recommendation scores based on user feedback signals. It incorporates user feedback to adapt recommendation scores dynamically, enabling interactive and adaptive recommendation systems.

This layer is crucial for adaptive recommendation systems where user feedback shapes future recommendations, creating personalized experiences that improve over time based on user interactions.

## How It Works

The layer processes feedback signals:

1. Input Scores: Initial recommendation scores (batch_size, num_items)
2. Feedback Signals: User feedback values (batch_size, 1)
3. Feedback Processing: Process feedback through dense transformation
4. Score Adjustment: Combine original scores with feedback-adjusted values
5. Output Adjusted Scores: (batch_size, num_items) adjusted scores

## Why Use This Layer?

| Challenge | Traditional Approach | FeedbackAdjustmentLayer Solution |
|-----------|---------------------|--------------------------------|
| User Feedback | Ignore feedback signals | Incorporate feedback directly |
| Adaptive Learning | Static recommendations | Dynamic adaptation to feedback |
| Personalization | Generic recommendations | Personalized based on feedback |
| Interactive Systems | One-way recommendations | Two-way feedback loop |
| User Satisfaction | Fixed ranking | Feedback-aware ranking |

## Use Cases

- Feedback-Aware Recommendations: Incorporate user feedback into scores
- Adaptive Ranking: Adjust rankings based on user signals
- Interactive Recommendations: Adapt to explicit user feedback
- Personalized Ranking: Personalize based on feedback history
- Online Learning: Continuous improvement from user feedback
- A/B Testing: Evaluate feedback mechanisms

## Quick Start

### Basic Usage

```python
import keras
from kmr.layers import FeedbackAdjustmentLayer

# Create adjustment layer
adjuster = FeedbackAdjustmentLayer()

# Adjust scores with feedback
scores = keras.random.normal((32, 100))
feedback = keras.random.uniform((32, 1), 0, 1)

adjusted_scores = adjuster([scores, feedback])
print(f"Original scores: {scores.shape}")
print(f"Adjusted scores: {adjusted_scores.shape}")  # (32, 100)
```

### In Adaptive Recommendation Pipeline

```python
import keras
from kmr.layers import (
    CollaborativeUserItemEmbedding,
    NormalizedDotProductSimilarity,
    FeedbackAdjustmentLayer,
    TopKRecommendationSelector
)

# Define inputs
user_id_input = keras.Input(shape=(1,), dtype='int32')
item_id_input = keras.Input(shape=(100,), dtype='int32')
feedback_input = keras.Input(shape=(1,), dtype='float32')

# Compute initial scores
embedding = CollaborativeUserItemEmbedding(1000, 5000, 32)
user_emb, item_emb = embedding([user_id_input, item_id_input])

similarity = NormalizedDotProductSimilarity()
scores = similarity([keras.ops.expand_dims(user_emb, 1), item_emb])

# Adjust with feedback
adjuster = FeedbackAdjustmentLayer()
adjusted_scores = adjuster([scores, feedback_input])

# Select top-K
selector = TopKRecommendationSelector(k=10)
rec_indices, rec_scores = selector(adjusted_scores)

# Build adaptive model
model = keras.Model(
    inputs=[user_id_input, item_id_input, feedback_input],
    outputs=[rec_indices, rec_scores]
)
```

## API Reference

::: kmr.layers.FeedbackAdjustmentLayer

## Parameters

This layer processes feedback automatically without explicit parameters.

### Automatic Behavior
- Feedback Processing: Transforms feedback through dense layers
- Score Combination: Combines original scores with feedback-adjusted values
- Adaptive Learning: Learns optimal feedback weights during training

## Performance Characteristics

- Speed: Fast - O(batch × items) adjustment operation
- Memory: Minimal - no additional buffers
- Accuracy: Excellent for adaptive recommendations
- Scalability: Good for large-scale systems
- Learning: Adapts weights during training

## Examples

### Example 1: Basic Feedback Adjustment

```python
import keras
from kmr.layers import FeedbackAdjustmentLayer

adjuster = FeedbackAdjustmentLayer()

# Initial scores
scores = keras.random.normal((16, 50))

# User feedback (e.g., click-through rates)
feedback = keras.random.uniform((16, 1), 0, 1)

# Adjust scores
adjusted = adjuster([scores, feedback])

print(f"Original scores range: [{scores.min():.3f}, {scores.max():.3f}]")
print(f"Adjusted scores range: [{adjusted.min():.3f}, {adjusted.max():.3f}]")
```

### Example 2: Feedback Impact Analysis

```python
import keras
from kmr.layers import FeedbackAdjustmentLayer

adjuster = FeedbackAdjustmentLayer()
scores = keras.random.normal((8, 100))

# Test different feedback values
for feedback_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
    feedback = keras.constant([[feedback_val]] * 8)
    adjusted = adjuster([scores, feedback])
    print(f"Feedback {feedback_val:.2f}: mean score = {adjusted.mean():.4f}")
```

### Example 3: Real-time Feedback Integration

```python
import keras
from kmr.layers import FeedbackAdjustmentLayer, TopKRecommendationSelector

# Initial recommendations
scores = keras.random.normal((32, 1000))

# User feedback from interactions
feedback = keras.constant([[0.8], [0.3], [0.9]] * 10 + [[0.5]] * 2)

# Adjust scores
adjuster = FeedbackAdjustmentLayer()
adjusted_scores = adjuster([scores, feedback])

# Re-rank with adjusted scores
selector = TopKRecommendationSelector(k=10)
rec_indices, rec_scores = selector(adjusted_scores)

print(f"Re-ranked recommendations: {rec_indices.shape}")
```

## Tips and Best Practices

- Feedback Normalization: Normalize feedback values to [0, 1] range
- Feedback Collection: Collect diverse feedback signals (clicks, ratings, time)
- Weight Learning: Let the layer learn optimal feedback weights
- Feedback Freshness: Use recent feedback for better adaptation
- Integration: Combine with explainable layers for transparency
- Evaluation: Monitor feedback impact on recommendation quality

## Common Pitfalls

- Feedback Bias: Biased feedback can skew recommendations
- Over-adaptation: Too much weight on feedback can cause instability
- Cold Start: New users have no feedback history
- Feedback Quality: Low-quality feedback reduces effectiveness
- Real-time Processing: Feedback processing must be fast

## Related Layers

- CosineSimilarityExplainer - Explain recommendations before feedback
- NormalizedDotProductSimilarity - Generate initial scores
- TopKRecommendationSelector - Select final recommendations
- LearnableWeightedCombination - Combine multiple feedback signals

## Further Reading

- Interactive Learning - Feedback-based learning systems
- Adaptive Recommendations - Dynamic recommendation adaptation
- User Feedback - Feedback collection and processing
- Online Learning - Continuous learning from feedback
