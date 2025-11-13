---
title: PositionalEmbedding - KerasFactory
description: Fixed sinusoidal positional encoding layer for transformer-based time series models
keywords: [positional encoding, transformers, time series, keras, embeddings, attention mechanisms]
---

# 📍 PositionalEmbedding

<div class="layer-hero">
  <div class="layer-hero-content">
    <h1>📍 PositionalEmbedding</h1>
    <div class="layer-badges">
      <span class="badge badge-beginner">🟢 Beginner</span>
      <span class="badge badge-stable">✅ Stable</span>
      <span class="badge badge-timeseries">⏱️ Time Series</span>
    </div>
  </div>
</div>

## 🎯 Overview

The `PositionalEmbedding` layer generates fixed sinusoidal positional encodings for time series and sequence data. Unlike learnable positional embeddings, this layer uses mathematically defined sinusoidal patterns that encode absolute position information, allowing transformer-based models to understand temporal relationships without training positional parameters.

Positional embeddings are essential for transformer architectures as they provide the model with information about the order and position of elements in sequences.

## 🔍 How It Works

The PositionalEmbedding generates sinusoidal encodings based on the mathematical formula:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where:
- `pos` is the position in the sequence
- `i` is the dimension index
- `d_model` is the model dimension

## 💡 Why Use This Layer?

| Challenge | Traditional Approach | PositionalEmbedding's Solution |
|-----------|---------------------|-------------------------------|
| **Position Information** | No position awareness | 🎯 **Fixed sinusoidal** encodings |
| **Scalability** | Learnable embeddings limited | ∞ **Extrapolates** to any length |
| **Interpretability** | Black-box embeddings | 🔍 **Mathematically interpretable** patterns |
| **Computational Cost** | Learnable parameters | ⚡ **Zero-cost** fixed computation |
| **Generalization** | Poor on unseen lengths | 🌍 **Works on any sequence length** |

## 📊 Use Cases

- **Transformer Models**: Providing position information to attention mechanisms
- **Time Series Forecasting**: Encoding temporal positions
- **Language Models**: Position awareness in NLP tasks
- **Sequence-to-Sequence Models**: Maintaining order information
- **Any Sequential Model**: When you need fixed, interpretable positional information

## 🚀 Quick Start

### Basic Usage

```python
import keras
from kerasfactory.layers import PositionalEmbedding

# Create sample sequence
batch_size, seq_len, d_model = 32, 100, 64
x = keras.random.normal((batch_size, seq_len, d_model))

# Apply positional embedding
pos_emb = PositionalEmbedding(max_len=100, d_model=d_model)
pe = pos_emb(x)

print(f"Input shape: {x.shape}")      # (32, 100, 64)
print(f"Embedding shape: {pe.shape}") # (32, 100, 64)
```

### In a Sequential Model

```python
import keras
from kerasfactory.layers import PositionalEmbedding, TokenEmbedding

model = keras.Sequential([
    keras.layers.Input(shape=(100, 1)),
    TokenEmbedding(c_in=1, d_model=64),  # Embed raw values
    PositionalEmbedding(max_len=100, d_model=64),  # Add positional info
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1),
])

model.compile(optimizer='adam', loss='mse')
```

## 🔧 API Reference

### PositionalEmbedding

```python
kerasfactory.layers.PositionalEmbedding(
    max_len: int = 5000,
    d_model: int = 512,
    name: str | None = None,
    **kwargs
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_len` | `int` | 5000 | Maximum sequence length to support |
| `d_model` | `int` | 512 | Model dimension (embedding dimension) |
| `name` | `str \| None` | None | Optional layer name |

#### Input Shape
- `(batch_size, seq_len, ...)`

#### Output Shape
- `(1, seq_len, d_model)`

#### Returns
- Fixed positional encodings for the sequence

## 📈 Performance Characteristics

- **Time Complexity**: O(seq_len × d_model) for generation (one-time during build)
- **Space Complexity**: O(seq_len × d_model) for storage
- **Computational Cost**: Minimal (no learnable parameters)
- **Training Efficiency**: No gradient computation needed

## 🎨 Advanced Usage

### With Different Sequence Lengths

```python
from kerasfactory.layers import PositionalEmbedding

# Create layer for max length 512
pos_emb = PositionalEmbedding(max_len=512, d_model=64)

# Can handle any length up to max_len
x_short = keras.random.normal((32, 100, 64))
x_medium = keras.random.normal((32, 256, 64))
x_long = keras.random.normal((32, 512, 64))

pe_short = pos_emb(x_short)    # Works fine
pe_medium = pos_emb(x_medium)  # Works fine
pe_long = pos_emb(x_long)      # Works fine
```

### Combining with Multiple Embeddings

```python
from kerasfactory.layers import PositionalEmbedding, TokenEmbedding

# Create embeddings
token_emb = TokenEmbedding(c_in=1, d_model=64)
pos_emb = PositionalEmbedding(max_len=100, d_model=64)

# Process sequence
x = keras.random.normal((32, 100, 1))
x_embedded = token_emb(x)           # (32, 100, 64)
x_pos = pos_emb(x_embedded)         # (32, 100, 64)

# Combine embeddings
output = x_embedded + x_pos         # Element-wise addition

print(output.shape)  # (32, 100, 64)
```

## 🔍 Visual Representation

```
┌─────────────────────────────────────────┐
│        Input Sequence (seq_len)         │
│  Shape: (batch, seq_len, d_model)      │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│    Generate Positional Encodings        │
│  - For each position: 0 to seq_len-1    │
│  - Apply sin/cos patterns               │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  Positional Embeddings (fixed)           │
│  Shape: (1, seq_len, d_model)           │
│  - Extrapolates to any length           │
│  - No learnable parameters              │
└─────────────────────────────────────────┘
```

## 💡 Best Practices

1. **Choose Appropriate max_len**: Set it to the maximum sequence length you expect
2. **Use Same d_model**: Ensure d_model matches your embedding dimension
3. **Add to Embeddings**: Typically added to token/value embeddings via addition
4. **Placement**: Usually placed after initial embeddings, before attention layers
5. **Multiple Scales**: The layer naturally captures patterns at multiple frequency scales

## ⚠️ Common Pitfalls

- ❌ **max_len too small**: Sequence lengths beyond max_len won't be handled correctly
- ❌ **d_model mismatch**: Using different d_model than embeddings causes shape errors
- ❌ **Treating as learnable**: These are fixed; don't expect them to train
- ❌ **Using alone**: Usually combined with token embeddings, not used standalone

## 📚 References

- Vaswani et al. (2017). "Attention Is All You Need" - Original transformer paper
- Sinusoidal positional encoding patterns from the original attention paper
- IEEE/ACM standards for positional encoding implementations

## 🔗 Related Layers

- [`TokenEmbedding`](token-embedding.md) - Embed raw time series values
- [`TemporalEmbedding`](temporal-embedding.md) - Embed temporal features
- [`DataEmbeddingWithoutPosition`](data-embedding-without-position.md) - Combined embedding layer

## ✅ Serialization

```python
# Get configuration
config = pos_emb.get_config()

# Recreate layer
pos_emb_new = PositionalEmbedding.from_config(config)
```

---

**Last Updated**: 2025-11-04  
**Version**: 1.0  
**Keras**: 3.0+
