---
title: CosineSimilarityExplainer - KMR
description: Compute and explain cosine similarity between embeddings for interpretable recommendations
keywords: [similarity, explanation, interpretability, cosine similarity, recommendation, keras, explainability]
---

# Cosine Similarity Explainer

<div class="layer-hero">
  <div class="layer-hero-content">
    <h1>Cosine Similarity Explainer</h1>
    <div class="layer-badges">
      <span class="badge badge-advanced">Advanced</span>
      <span class="badge badge-stable">Stable</span>
      <span class="badge badge-recommendation">Recommendation</span>
    </div>
  </div>
</div>

## Overview

The `CosineSimilarityExplainer` computes cosine similarity between user and item embeddings while providing interpretable similarity scores. It analyzes user-item similarity with explainability in mind, making it essential for transparent recommendation systems.

This layer is crucial for explainable recommendation systems where understanding why items are recommended is important for user trust and satisfaction.

## How It Works

The layer computes normalized cosine similarity:

1. User Embedding Input: (batch_size, 1, embedding_dim)
2. Item Embeddings Input: (batch_size, num_items, embedding_dim)
3. Normalize User Embeddings: Divide by L2 norm for unit vectors
4. Normalize Item Embeddings: Divide by L2 norm for unit vectors
5. Compute Dot Product: Matrix multiplication of normalized vectors
6. Output Similarities: (batch_size, num_items) with interpretable scores

## Why Use This Layer?

| Challenge | Traditional Approach | CosineSimilarityExplainer Solution |
|-----------|---------------------|-----------------------------------|
| Interpretability | Black-box similarity | Transparent cosine similarity |
| Explainability | Hard to explain scores | Easy-to-understand normalized scores |
| Normalization | Manual normalization | Built-in normalization |
| Explanation Format | Raw scores difficult to interpret | Bounded [-1,1] interpretable range |
| Integration | Separate similarity and explanation | Unified similarity with explanation |

## Use Cases

- Explainable Recommendations: Provide reasons for recommendations
- Similarity Analysis: Analyze user-item similarity patterns
- Recommendation Transparency: Explain similarity-based rankings
- Interpretable Rankings: Trace recommendations back to similarities
- User Trust: Build trust through transparency in recommendations
- Debugging Recommendations: Understand recommendation decisions

## Quick Start

### Basic Usage

```python
import keras
from kmr.layers import CosineSimilarityExplainer

# Create explainer layer
explainer = CosineSimilarityExplainer()

# Compute and explain similarities
user_emb = keras.random.normal((32, 1, 64))
item_emb = keras.random.normal((32, 100, 64))

similarities = explainer([user_emb, item_emb])
print(f"Similarities shape: {similarities.shape}")  # (32, 100)
print(f"Similarity range: [{similarities.min():.3f}, {similarities.max():.3f}]")
```

### In Explainable Recommendation Pipeline

```python
import keras
from kmr.layers import (
    CollaborativeUserItemEmbedding,
    CosineSimilarityExplainer,
    TopKRecommendationSelector
)

# Define inputs
user_id_input = keras.Input(shape=(1,), dtype='int32')
item_id_input = keras.Input(shape=(50,), dtype='int32')

# Get embeddings
embedding_layer = CollaborativeUserItemEmbedding(1000, 5000, 32)
user_emb, item_emb = embedding_layer([user_id_input, item_id_input])

# Explain similarities
explainer = CosineSimilarityExplainer()
user_exp = keras.ops.expand_dims(user_emb, axis=1)
similarities = explainer([user_exp, item_emb])

# Select top-K with explanation
selector = TopKRecommendationSelector(k=10)
rec_indices, rec_scores = selector(similarities)

# Build model for transparent recommendations
model = keras.Model(
    inputs=[user_id_input, item_id_input],
    outputs=[rec_indices, rec_scores, similarities]  # Include similarities for explanation
)
```

## API Reference

::: kmr.layers.CosineSimilarityExplainer

## Parameters

### input_shape
- Purpose: Shape of input embeddings
- Handled Automatically: Layer infers from inputs
- Note: User embedding (batch, 1, dim), Item embedding (batch, items, dim)

## Performance Characteristics

- Speed: Very fast - O(batch × items × dim)
- Memory: Minimal - output same size as input similarities
- Accuracy: Mathematically precise cosine similarity
- Interpretability: Excellent - bounded [-1,1] range
- Scalability: Excellent for large item catalogs

## Examples

### Example 1: Basic Similarity Explanation

```python
import keras
from kmr.layers import CosineSimilarityExplainer

# Create explainer
explainer = CosineSimilarityExplainer()

# Random embeddings
user_emb = keras.random.normal((8, 1, 32))
item_emb = keras.random.normal((8, 100, 32))

# Compute similarities
similarities = explainer([user_emb, item_emb])

# Analyze results
print(f"Similarities shape: {similarities.shape}")
print(f"Min similarity: {similarities.min():.4f}")
print(f"Max similarity: {similarities.max():.4f}")
print(f"Mean similarity: {similarities.mean():.4f}")
```

### Example 2: Analyzing Similarity Patterns

```python
import keras
import numpy as np
from kmr.layers import CosineSimilarityExplainer

explainer = CosineSimilarityExplainer()

# Create realistic embeddings
user_emb = keras.random.normal((16, 1, 64))
item_emb = keras.random.normal((16, 100, 64))

similarities = explainer([user_emb, item_emb])

# Analyze distribution
for user_idx in range(3):
    user_sims = similarities[user_idx]
    print(f"User {user_idx}:")
    print(f"  Top 3 items: {keras.ops.argsort(user_sims)[-3:]}")
    print(f"  Top 3 scores: {keras.ops.sort(user_sims)[-3:]}")
```

### Example 3: Interpretable Explanations

```python
import keras
from kmr.layers import (
    CollaborativeUserItemEmbedding,
    CosineSimilarityExplainer,
    TopKRecommendationSelector
)

# Setup
user_id = keras.constant([1])
item_ids = keras.constant([[10, 20, 30, 40, 50]])

embedding_layer = CollaborativeUserItemEmbedding(100, 100, 16)
user_emb, item_emb = embedding_layer([user_id, item_ids])

# Get explanations
explainer = CosineSimilarityExplainer()
similarities = explainer([keras.ops.expand_dims(user_emb, 1), item_emb])

# Present explanation
print(f"Why were these items recommended?")
for idx, item_id in enumerate(item_ids[0].numpy()):
    similarity_score = similarities[0, idx].numpy()
    print(f"  Item {item_id}: Similarity {similarity_score:.4f} (cosine)")
    print(f"    Interpretation: {interpretation_from_score(similarity_score)}")

def interpretation_from_score(score):
    if score > 0.8:
        return "Very similar - highly relevant"
    elif score > 0.6:
        return "Similar - likely to be interesting"
    elif score > 0.4:
        return "Moderately similar - may be relevant"
    else:
        return "Low similarity - different preferences"
```

## Tips and Best Practices

- Embedding Quality: High-quality embeddings produce more meaningful explanations
- Normalization: Cosine similarity is scale-invariant, good for explanation
- Interpretation: Score range [-1,1] is intuitive and explainable
- User Communication: Explain scores as similarity percentages to users
- Integration: Use with feedback layers for adaptive explanations
- Visualization: Visualize similarity matrices for pattern analysis

## Common Pitfalls

- Zero Vectors: May produce NaN if embeddings have zero norm
- Dimension Mismatch: User and item embedding dims must match
- Interpretation: Don't confuse cosine similarity with other distance metrics
- Normalization: Result is always normalized; don't double normalize
- Performance: Very large embedding dimensions may slow computation

## Related Layers

- CollaborativeUserItemEmbedding - Get embeddings for similarity
- NormalizedDotProductSimilarity - Alternative similarity computation
- FeedbackAdjustmentLayer - Adjust scores based on feedback
- TopKRecommendationSelector - Select top recommendations

## Further Reading

- Cosine Similarity - Mathematical foundation
- Explainable AI - Interpretability principles
- Recommendation Systems - RS overview
- Vector Normalization - Mathematical concepts
