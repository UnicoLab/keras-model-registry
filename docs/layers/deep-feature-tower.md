---
title: DeepFeatureTower - KMR
description: Dense neural network tower for processing user or item features in recommendation systems
keywords: [deep neural network, feature tower, recommendation, two-tower architecture, keras, representation learning]
---

# 🏢 DeepFeatureTower

<div class="layer-hero">
  <div class="layer-hero-content">
    <h1>🏢 DeepFeatureTower</h1>
    <div class="layer-badges">
      <span class="badge badge-intermediate">🟡 Intermediate</span>
      <span class="badge badge-stable">✅ Stable</span>
      <span class="badge badge-recommendation">📊 Recommendation</span>
    </div>
  </div>
</div>

## 🎯 Overview

The `DeepFeatureTower` is a stack of dense layers with batch normalization and dropout for processing user or item features in two-tower recommendation architectures. It transforms raw features into rich representations for similarity-based recommendations.

This layer is fundamental to modern content-based and hybrid recommendation systems, enabling effective feature learning through deep neural networks while maintaining training stability through batch normalization.

## 🔍 How It Works

The DeepFeatureTower processes features through multiple stacked layers:

1. **Input Features**: Raw user or item features (batch_size, input_dim)
2. **Dense Layers**: Multiple dense layers with configurable activations
3. **Batch Normalization**: Normalizes activations between layers for training stability
4. **Dropout**: Regularization to prevent overfitting during training
5. **Output Representation**: Learned feature representation (batch_size, units)

Each layer applies: Dense → BatchNorm → Dropout sequentially.

## 💡 Why Use This Layer?

| Challenge | Traditional Approach | DeepFeatureTower Solution |
|-----------|---------------------|--------------------------|
| **Feature Learning** | Manual feature engineering | 🎯 **Automatic** deep learning |
| **Normalization** | Separate layers | ⚡ **Integrated** batch normalization |
| **Regularization** | Manual dropout | 🧠 **Configurable** dropout rates |
| **Architecture Consistency** | Multiple layers to manage | 🔗 **Unified** tower definition |
| **Training Stability** | Manual tuning | ⚡ **Automatic** stability via BatchNorm |

## 📊 Use Cases

- **Two-Tower Models**: User and item feature processing in parallel towers
- **Content-Based Filtering**: Processing rich features for recommendations
- **Hybrid Approaches**: Combining collaborative and content-based signals
- **Feature Transformation**: Converting sparse to dense representations
- **Deep Learning Pipelines**: General feature learning tasks

## 🚀 Quick Start

```python
import keras
from kmr.layers import DeepFeatureTower

# Create feature tower for user features
user_features = keras.random.normal((32, 20))
tower = DeepFeatureTower(
    units=32,
    hidden_layers=2,
    dropout_rate=0.2,
    activation='relu'
)

# Process features
user_repr = tower(user_features, training=True)
print(f"Input: {user_features.shape} -> Output: {user_repr.shape}")  # (32, 20) -> (32, 32)
```

### In a Two-Tower Model

```python
import keras
from kmr.layers import DeepFeatureTower, NormalizedDotProductSimilarity

# Create model inputs
user_features_input = keras.Input(shape=(15,), name='user_features')
item_features_input = keras.Input(shape=(50, 12), name='item_features')

# Create towers
user_tower = DeepFeatureTower(units=32, hidden_layers=2, dropout_rate=0.2)
item_tower = DeepFeatureTower(units=32, hidden_layers=2, dropout_rate=0.2)

# User tower
user_repr = user_tower(user_features_input)  # (batch, 32)

# Item tower - reshape for batch processing
batch_size = keras.ops.shape(item_features_input)[0]
num_items = keras.ops.shape(item_features_input)[1]
item_flat = keras.ops.reshape(item_features_input, (-1, 12))
item_repr_flat = item_tower(item_flat)
item_repr = keras.ops.reshape(item_repr_flat, (batch_size, num_items, 32))

# Compute similarities
similarity = NormalizedDotProductSimilarity()([
    keras.ops.expand_dims(user_repr, axis=1),
    item_repr
])

model = keras.Model(
    inputs=[user_features_input, item_features_input],
    outputs=similarity
)
```

## 📖 API Reference

::: kmr.layers.DeepFeatureTower

## 🔧 Parameters Deep Dive

### `units` (int)
- **Purpose**: Output dimension of the tower
- **Range**: 8 to 512 (typically 16-128)
- **Impact**: Size of learned representation
- **Recommendation**: Start with 32, scale based on data complexity

### `hidden_layers` (int)
- **Purpose**: Number of dense layers in the tower
- **Range**: 1 to 10 (typically 2-4)
- **Impact**: Model capacity and depth
- **Recommendation**: 2-3 for balanced complexity; more layers = higher capacity but harder to train

### `dropout_rate` (float)
- **Purpose**: Fraction of inputs to drop during training
- **Range**: 0.0 to 0.5 (typically 0.2-0.3)
- **Default**: 0.2
- **Recommendation**: Increase for overfitting, decrease for underfitting

### `l2_reg` (float)
- **Purpose**: L2 regularization strength on weights
- **Range**: 0.0 to 0.1
- **Default**: 1e-4
- **Tip**: Balance with dropout for regularization

### `activation` (str)
- **Purpose**: Activation function for dense layers
- **Options**: 'relu', 'tanh', 'sigmoid', 'elu', 'selu'
- **Default**: 'relu'
- **Recommendation**: 'relu' for most cases; 'elu' or 'selu' for deeper networks

## 📈 Performance Characteristics

- **Speed**: ⚡⚡⚡ Fast - linear transformations
- **Memory**: 💾💾💾 Scales with layer sizes (units × hidden_layers × input_dim)
- **Accuracy**: 🎯🎯🎯🎯 Excellent for feature learning
- **Capacity**: Can learn complex non-linear transformations

## 🎨 Examples

### Example 1: User Feature Processing

```python
import keras
from kmr.layers import DeepFeatureTower

# Simulate user features (age, income, interests, etc.)
num_users, feature_dim = 100, 15
user_features = keras.random.normal((num_users, feature_dim))

# Create tower for user representation
user_tower = DeepFeatureTower(
    units=32,
    hidden_layers=3,
    dropout_rate=0.2,
    activation='relu'
)

# Get user representations
user_repr = user_tower(user_features)
print(f"User representations: {user_repr.shape}")  # (100, 32)
```

### Example 2: Comparing Tower Depths

```python
import keras
from kmr.layers import DeepFeatureTower

# Create towers with different depths
shallow = DeepFeatureTower(units=32, hidden_layers=1)
medium = DeepFeatureTower(units=32, hidden_layers=2)
deep = DeepFeatureTower(units=32, hidden_layers=4)

# Test data
features = keras.random.normal((64, 20))

# Process
shallow_out = shallow(features)
medium_out = medium(features)
deep_out = deep(features)

print(f"Shallow: {shallow_out.shape}")  # (64, 32)
print(f"Medium: {medium_out.shape}")   # (64, 32)
print(f"Deep: {deep_out.shape}")       # (64, 32)
```

## 💡 Tips & Best Practices

- **Units**: Start with 32, increase for more capacity if needed
- **Layers**: 2-3 layers usually sufficient; avoid too deep towers (>5) without careful tuning
- **Dropout**: Use 0.2-0.3 for regularization; increase if overfitting
- **Activation**: 'relu' works best for most cases
- **Training Mode**: Always set training=True during training for proper dropout and BatchNorm
- **Feature Normalization**: Pre-normalize features for better convergence

## ⚠️ Common Pitfalls

- **Input Shape**: Ensure inputs match feature dimensions
- **Output Size**: Always (batch_size, units)
- **Training Mode**: Dropout behaves differently in inference - incorrect mode causes problems
- **Deep Networks**: Very deep towers (>5 layers) can be hard to train without residual connections
- **Regularization**: Balance dropout with L2 for best results

## 🔗 Related Layers

- [CollaborativeUserItemEmbedding](collaborative-user-item-embedding.md) - User/item embeddings
- [NormalizedDotProductSimilarity](normalized-dot-product-similarity.md) - Similarity computation
- [DeepFeatureRanking](deep-feature-ranking.md) - Ranking with deep features

## 📚 Further Reading

- [Deep Learning for Recommendation Systems](https://arxiv.org/abs/1801.02688)
- [YouTube's Two-Tower Model](https://arxiv.org/abs/1902.07046)
- [Batch Normalization](https://arxiv.org/abs/1502.03167)
- [Dropout Regularization](https://jmlr.org/papers/v15/srivastava14a.html)
