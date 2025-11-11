---
title: TopKRecommendationSelector - KMR
description: Select top-K recommendation items based on scores
keywords: [top-k, ranking, selection, recommendation, scoring, keras, heap]
---

# 🏆 TopKRecommendationSelector

<div class="layer-hero">
  <div class="layer-hero-content">
    <h1>🏆 TopKRecommendationSelector</h1>
    <div class="layer-badges">
      <span class="badge badge-beginner">🟢 Beginner</span>
      <span class="badge badge-stable">✅ Stable</span>
      <span class="badge badge-recommendation">📊 Recommendation</span>
    </div>
  </div>
</div>

## 🎯 Overview

The `TopKRecommendationSelector` selects the top-K recommendation items based on their scores. It retrieves the indices and scores of the highest-scoring items, forming the basis for returning final recommendations to users.

This layer is the final step in recommendation pipelines, converting continuous scores into actionable top-K recommendations efficiently using heap-based selection.

## 🔍 How It Works

The layer performs efficient top-K selection:

1. **Input Scores**: (batch_size, num_items)
2. **Sort Scores**: Find top-K items by score
3. **Extract Indices**: Get item indices of top-K
4. **Extract Scores**: Get score values of top-K
5. **Output**: (batch_size, k) indices and scores

## 💡 Why Use This Layer?

| Challenge | Traditional Approach | Solution |
|-----------|---------------------|----------|
| **Top-K Selection** | Manual sorting and indexing | 🎯 **Automatic** top-K |
| **Efficiency** | Sorting all items O(n log n) | ⚡ **Heap O(n log k)** |
| **Ease of Use** | Complex index management | 🧠 **Simple API** |
| **Scalability** | Slow for large catalogs | 🔗 **Fast millions** |

## 📊 Use Cases

- **Final Recommendation Ranking**: Converting scores to top-K recommendations
- **Retrieval Stages**: Reducing candidate set in multi-stage systems
- **Batch Inference**: Processing multiple users simultaneously
- **A/B Testing**: Generating recommendation lists for evaluation

## 🚀 Quick Start

```python
import keras
from kmr.layers import TopKRecommendationSelector

# Create selector
selector = TopKRecommendationSelector(k=10)

# Create sample scores
batch_size, num_items = 32, 1000
scores = keras.random.normal((batch_size, num_items))

# Select top-K
indices, top_scores = selector(scores)

print(f"Recommendation indices: {indices.shape}")  # (32, 10)
print(f"Recommendation scores: {top_scores.shape}")   # (32, 10)
```

### In a Complete Pipeline

```python
import keras
from kmr.layers import (
    CollaborativeUserItemEmbedding,
    NormalizedDotProductSimilarity,
    TopKRecommendationSelector
)

# Model inputs
user_id_input = keras.Input(shape=(1,), dtype='int32', name='user_id')
item_id_input = keras.Input(shape=(100,), dtype='int32', name='item_id')

# Embedding + Similarity + Selection
embedding = CollaborativeUserItemEmbedding(1000, 10000, 32)
user_emb, item_emb = embedding([user_id_input, item_id_input])

similarity = NormalizedDotProductSimilarity()
scores = similarity([keras.ops.expand_dims(user_emb, 1), item_emb])

selector = TopKRecommendationSelector(k=10)
rec_indices, rec_scores = selector(scores)

model = keras.Model([user_id_input, item_id_input], [rec_indices, rec_scores])
```

## 📖 API Reference

::: kmr.layers.TopKRecommendationSelector

## 🔧 Parameters

### `k` (int)
- **Purpose**: Number of top recommendations to select
- **Range**: 1 to num_items
- **Typical**: 5-100
- **Impact**: Determines recommendation list size

## 📈 Performance Characteristics

- **Speed**: ⚡⚡⚡⚡ O(n log k) heap-based selection
- **Memory**: 💾 Minimal - only stores top-K
- **Accuracy**: 🎯🎯🎯🎯 Perfect ranking preservation
- **Scalability**: Excellent for large catalogs (millions of items)

## 🎨 Examples

### Example 1: Different K Values

```python
import keras
from kmr.layers import TopKRecommendationSelector

# Create scores
scores = keras.random.normal((16, 100))

# Try different K values
for k in [5, 10, 20]:
    selector = TopKRecommendationSelector(k=k)
    indices, top_scores = selector(scores)
    print(f"k={k}: {indices.shape}")  # (16, k)
```

### Example 2: Score Analysis

```python
import keras
from kmr.layers import TopKRecommendationSelector

# Create realistic scores (exponential distribution)
scores = keras.random.exponential(0.5, shape=(32, 1000))

selector = TopKRecommendationSelector(k=10)
indices, top_scores = selector(scores)

print(f"Top score: {top_scores.max():.3f}")
print(f"10th score: {top_scores.min():.3f}")
print(f"Average top-10 score: {top_scores.mean():.3f}")
```

## 💡 Tips & Best Practices

- **K Value**: Adjust based on acceptable recommendation list length
- **Batch Processing**: Efficiently handles multiple users
- **Score Range**: Works with any score range (negative, positive, normalized)
- **Integration**: Final layer in most recommendation pipelines

## ⚠️ Common Pitfalls

- **K too large**: Reduced diversity in recommendations
- **K too small**: Limited options for users
- **Invalid K**: Must be positive and ≤ num_items
- **Performance**: Very large K (> 1000) reduces efficiency

## 🔗 Related Layers

- [NormalizedDotProductSimilarity](normalized-dot-product-similarity.md) - Score computation
- [CollaborativeUserItemEmbedding](collaborative-user-item-embedding.md) - Embeddings
- [DeepFeatureRanking](deep-feature-ranking.md) - Deep ranking

## 📚 Further Reading

- [Learning to Rank](https://en.wikipedia.org/wiki/Learning_to_rank)
- [Selection Algorithms](https://en.wikipedia.org/wiki/Selection_algorithm)
- [Recommendation Systems](https://arxiv.org/abs/1707.07435)
