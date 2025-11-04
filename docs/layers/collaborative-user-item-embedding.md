---
title: CollaborativeUserItemEmbedding - KMR
description: Dual embedding lookup layer for collaborative filtering in recommendation systems
keywords: [collaborative filtering, embeddings, user embeddings, item embeddings, recommendation, matrix factorization, keras]
---

# Collaborative User Item Embedding

<div class="layer-hero">
  <div class="layer-hero-content">
    <h1>Collaborative User Item Embedding</h1>
    <div class="layer-badges">
      <span class="badge badge-intermediate">Intermediate</span>
      <span class="badge badge-stable">Stable</span>
      <span class="badge badge-recommendation">Recommendation</span>
    </div>
  </div>
</div>

## Overview

The `CollaborativeUserItemEmbedding` layer provides dual embedding lookups for users and items in collaborative filtering recommendation systems. It maintains separate embedding tables for users and items with optional L2 regularization to prevent overfitting.

This layer is essential for matrix factorization-based recommendation systems, capturing latent user and item representations for similarity computation. By learning low-dimensional dense representations of users and items, it enables efficient similarity calculations and recommendations.

## How It Works

The layer processes user and item IDs through separate embedding tables:

1. **User ID Input**: Receives user identifiers (batch_size,)
2. **Item ID Input**: Receives item identifiers (batch_size, num_items)
3. **User Embedding Lookup**: Maps user IDs to user embeddings (batch_size, embedding_dim)
4. **Item Embedding Lookup**: Maps item IDs to item embeddings (batch_size, num_items, embedding_dim)
5. **L2 Regularization**: Optional regularization on embedding weights to prevent overfitting

## Why Use This Layer?

| Challenge | Traditional Approach | CollaborativeUserItemEmbedding Solution |
|-----------|---------------------|----------------------------------------|
| Embedding Lookup | Manual embedding management | Integrated embedding tables |
| Regularization | Manual weight regularization | Built-in L2 regularization |
| Dual Embeddings | Separate layers for users/items | Combined user-item embeddings |
| Scalability | Memory-intensive for large catalogs | Efficient embedding indexing |
| Simplicity | Complex embedding setup | Simple single-layer solution |

## Use Cases

- Collaborative Filtering: User-item similarity-based recommendations
- Matrix Factorization: Learning latent representations of users and items
- Embedding-based Ranking: Converting IDs to embedding vectors
- Cold Start Handling: Initializing new users/items with embeddings
- Similarity-based Retrieval: Finding similar users or items

## Quick Start

### Basic Usage

```python
import keras
from kmr.layers import CollaborativeUserItemEmbedding

# Create sample data
batch_size, num_users, num_items = 32, 1000, 5000
user_ids = keras.random.randint((batch_size,), 0, num_users)
item_ids = keras.random.randint((batch_size, 100), 0, num_items)

# Create embedding layer
embedding_layer = CollaborativeUserItemEmbedding(
    num_users=num_users,
    num_items=num_items,
    embedding_dim=32
)

# Get embeddings
user_emb, item_emb = embedding_layer([user_ids, item_ids])

print(f"User embeddings shape: {user_emb.shape}")    # (32, 32)
print(f"Item embeddings shape: {item_emb.shape}")    # (32, 100, 32)
```

### In a Complete Recommendation Model

```python
import keras
from kmr.layers import (
    CollaborativeUserItemEmbedding,
    NormalizedDotProductSimilarity,
    TopKRecommendationSelector
)

# Define inputs
user_id_input = keras.Input(shape=(1,), dtype='int32', name='user_id')
item_id_input = keras.Input(shape=(100,), dtype='int32', name='item_id')

# Embedding lookup
embedding_layer = CollaborativeUserItemEmbedding(
    num_users=1000,
    num_items=5000,
    embedding_dim=32,
    l2_reg=1e-4
)
user_emb, item_emb = embedding_layer([user_id_input, item_id_input])

# Compute similarities
similarity_layer = NormalizedDotProductSimilarity()
similarities = similarity_layer([
    keras.ops.expand_dims(user_emb, axis=1),
    item_emb
])

# Select top-10
selector = TopKRecommendationSelector(k=10)
rec_indices, rec_scores = selector(similarities)

# Build model
model = keras.Model(
    inputs=[user_id_input, item_id_input],
    outputs=[rec_indices, rec_scores]
)
```

## API Reference

::: kmr.layers.CollaborativeUserItemEmbedding

## Parameters Deep Dive

### num_users (int)
- Purpose: Total number of unique users in the catalog
- Range: 10 to 1,000,000+
- Impact: Determines user embedding table size
- Memory Impact: (num_users × embedding_dim × 4 bytes) approximately

### num_items (int)
- Purpose: Total number of unique items in the catalog
- Range: 10 to 10,000,000+
- Impact: Determines item embedding table size
- Memory Impact: (num_items × embedding_dim × 4 bytes) approximately

### embedding_dim (int)
- Purpose: Dimensionality of embedding vectors
- Range: 8 to 512 (typically 16-128)
- Recommendation: Start with 32-64, increase for larger catalogs
- Trade-off: Higher dimensions capture more information but increase memory/computation

### l2_reg (float)
- Purpose: L2 regularization strength on embedding weights
- Range: 0.0 to 0.1
- Default: 1e-4
- Tip: Increase for overfitting, decrease for underfitting
- Impact: Prevents embeddings from growing too large during training

## Performance Characteristics

- Speed: Very fast - simple embedding lookups with O(1) access time
- Memory: Linear with catalog size: O(num_users * embedding_dim + num_items * embedding_dim)
- Accuracy: Excellent for collaborative filtering
- Scalability: Good for millions of users/items (typical production systems)
- Training Speed: Efficient gradient updates

## Examples

### Example 1: Basic Collaborative Filtering

```python
import keras
from kmr.layers import (
    CollaborativeUserItemEmbedding,
    NormalizedDotProductSimilarity,
    TopKRecommendationSelector
)

# Setup
num_users, num_items, embedding_dim = 100, 500, 32

# Create data
user_ids = keras.random.randint((16,), 0, num_users)
item_ids = keras.random.randint((16, 20), 0, num_items)

# Create layers
embedding = CollaborativeUserItemEmbedding(num_users, num_items, embedding_dim)
similarity = NormalizedDotProductSimilarity()
selector = TopKRecommendationSelector(k=5)

# Forward pass
user_emb, item_emb = embedding([user_ids, item_ids])
scores = similarity([keras.ops.expand_dims(user_emb, 1), item_emb])
indices, rec_scores = selector(scores)

print(f"Top-5 recommendations shape: {indices.shape}")
print(f"Recommendation scores shape: {rec_scores.shape}")
```

### Example 2: Regularization Effects

```python
import keras
from kmr.layers import CollaborativeUserItemEmbedding

# Compare different L2 regularization strengths
regularization_strengths = [0.0, 1e-4, 1e-3, 1e-2]
layers = {}

for l2_strength in regularization_strengths:
    layer = CollaborativeUserItemEmbedding(
        num_users=1000,
        num_items=5000,
        embedding_dim=32,
        l2_reg=l2_strength
    )
    layers[l2_strength] = layer

# Test with data
user_ids = keras.random.randint((8,), 0, 1000)
item_ids = keras.random.randint((8, 10), 0, 5000)

for l2_strength, layer in layers.items():
    user_emb, item_emb = layer([user_ids, item_ids])
    print(f"L2={l2_strength}: user_emb norm = {keras.ops.norm(user_emb):.4f}")
```

### Example 3: Large-Scale Catalog

```python
import keras
from kmr.layers import CollaborativeUserItemEmbedding

# Production-scale settings
embedding = CollaborativeUserItemEmbedding(
    num_users=1_000_000,      # 1 million users
    num_items=10_000_000,     # 10 million items
    embedding_dim=64,          # balanced dimension
    l2_reg=1e-3
)

# Process batch
batch_size = 512
user_ids = keras.random.randint((batch_size,), 0, 1_000_000)
item_ids = keras.random.randint((batch_size, 100), 0, 10_000_000)

user_emb, item_emb = embedding([user_ids, item_ids])
print(f"Processed {batch_size} users with {100} items")
```

## Tips and Best Practices

- Embedding Dimension: Start with 32, increase incrementally for better quality
- L2 Regularization: Use 1e-4 to 1e-3 for typical use cases
- Batch Size: Use larger batches (256+) for better GPU utilization
- User/Item Catalogs: Re-train if new users/items are added regularly
- Cold Start: Use pre-trained embeddings for new items
- Normalization: Consider normalizing embeddings for similarity computation

## Common Pitfalls

- ID Ranges: Ensure IDs are within [0, num_users) and [0, num_items)
- Out of Range IDs: Double-check data preprocessing to avoid invalid indices
- Memory Usage: Large catalogs (100M+) require significant memory
- Overfitting: Use L2 regularization and dropout with small datasets
- Sparse Data: Recommendation systems have sparse interactions

## Related Layers

- NormalizedDotProductSimilarity - Compute user-item similarity
- TopKRecommendationSelector - Select top-K recommendations
- DeepFeatureTower - Content-based feature processing
- CosineSimilarityExplainer - Explain similarity scores

## Further Reading

- Collaborative Filtering - Foundational overview
- Matrix Factorization - Netflix Prize approach
- Embedding Techniques - Comprehensive embedding survey
- Recommendation Systems Survey - Comprehensive RS overview
