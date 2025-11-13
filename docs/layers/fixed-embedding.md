---
title: FixedEmbedding - KerasFactory
description: Non-trainable sinusoidal embeddings for discrete temporal indices
keywords: [embedding, sinusoidal, fixed embeddings, temporal features, keras, time series]
---

# 📍 FixedEmbedding

<div class="layer-hero">
  <div class="layer-hero-content">
    <h1>📍 FixedEmbedding</h1>
    <div class="layer-badges">
      <span class="badge badge-beginner">🟢 Beginner</span>
      <span class="badge badge-stable">✅ Stable</span>
      <span class="badge badge-timeseries">⏱️ Time Series</span>
    </div>
  </div>
</div>

## 🎯 Overview

The `FixedEmbedding` layer generates non-trainable sinusoidal embeddings for discrete indices (0-indexed categorical values). Perfect for embedding discrete features like:
- Month of year (0-11)
- Day of month (0-30)
- Day of week (0-6)
- Hour of day (0-23)
- Minute of hour (0-59)

These fixed embeddings provide interpretable, frequency-based representations that capture periodicity without trainable parameters.

## 🔍 How It Works

```
Input Index: value in [0, vocab_size)
        |
        V
Sinusoidal Embedding:
- Even dims: sin(value / 10000^(2i/d_model))
- Odd dims: cos(value / 10000^(2i/d_model))
        |
        V
Output: (batch, d_model)
```

The sinusoidal pattern ensures:
- **Periodicity**: Captures cyclical nature (weeks, hours, etc.)
- **Interpretability**: Same index always gets same embedding
- **No Training**: Fixed patterns learned from scratch by model
- **Scalability**: Works for any vocab size

## 💡 Why Use This Layer?

| Advantage | Benefit |
|-----------|---------|
| **Fixed Patterns** | Deterministic, reproducible embeddings |
| **No Parameters** | Lightweight, no training overhead |
| **Interpretable** | Understand what embeddings represent |
| **Periodic** | Perfect for cyclical temporal features |
| **Fast** | Simple computation, O(1) lookup |

## 📊 Use Cases

- **Temporal Features**: Month, day, hour, minute embeddings
- **Categorical Encoding**: Any discrete feature with natural ordering
- **Frequency Analysis**: Capture patterns in discrete sequences
- **Cyclical Features**: Day-of-week, season, hour-of-day patterns
- **Lightweight Models**: Reduce parameters when not training embeddings

## 🚀 Quick Start

```python
import keras
from kerasfactory.layers import FixedEmbedding

# Create fixed embedding for hours (0-23)
hour_embed = FixedEmbedding(vocab_size=24, d_model=64)

# Input: hour indices
hours = keras.ops.cast(
    keras.random.uniform((32, 96), minval=0, maxval=24),
    'int32'
)

# Get embeddings
output = hour_embed(hours)
print(output.shape)  # (32, 96, 64)
```

## 🔧 API Reference

```python
kerasfactory.layers.FixedEmbedding(
    vocab_size: int,
    d_model: int,
    name: str | None = None,
    **kwargs: Any
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `vocab_size` | `int` | Number of possible indices |
| `d_model` | `int` | Embedding dimension |
| `name` | `str \| None` | Optional layer name |

## 📚 References

- Vaswani, A., et al. (2017). "Attention Is All You Need"
- Positional encoding with sinusoidal functions

## 🔗 Related Layers

- [`TemporalEmbedding`](temporal-embedding.md) - Uses FixedEmbedding for temporal features
- [`PositionalEmbedding`](positional-embedding.md) - Similar sinusoidal approach
- [`DataEmbeddingWithoutPosition`](data-embedding-without-position.md) - Combined embeddings

---

**Last Updated**: 2025-11-04 | **Keras**: 3.0+ | **Status**: ✅ Production Ready
