---
title: DataEmbeddingWithoutPosition - KerasFactory
description: Combined token and temporal embedding layer for time series with integrated dropout
keywords: [embedding, token embedding, temporal embedding, time series, multimodal input, keras]
---

# 🎯 DataEmbeddingWithoutPosition

<div class="layer-hero">
  <div class="layer-hero-content">
    <h1>🎯 DataEmbeddingWithoutPosition</h1>
    <div class="layer-badges">
      <span class="badge badge-beginner">🟢 Beginner</span>
      <span class="badge badge-stable">✅ Stable</span>
      <span class="badge badge-timeseries">⏱️ Time Series</span>
    </div>
  </div>
</div>

## 🎯 Overview

The `DataEmbeddingWithoutPosition` layer combines two complementary embedding modalities for time series data:

1. **Token Embedding**: Learns value representations from raw time series
2. **Temporal Embedding**: Encodes calendar/temporal features (month, day, weekday, hour, minute)

It's designed to fuse both value and temporal information in a single, learnable representation with integrated regularization through dropout.

Perfect for time series models that leverage both:
- **Historical values** (price, temperature, traffic volume)
- **Temporal context** (day-of-week effects, seasonal patterns, hourly patterns)

## 🔍 How It Works

```
Input: Raw Time Series Values        Input: Temporal Features
x: (batch, time, channels)           x_mark: (batch, time, 5)
        │                                    │
        ▼                                    ▼
┌──────────────────┐            ┌────────────────────┐
│ TokenEmbedding   │            │TemporalEmbedding   │
│ Conv1D learning  │            │ Month/Day/Hour...  │
└────────┬─────────┘            └─────────┬──────────┘
         │                               │
         ▼                               ▼
    (batch, time, d_model)      (batch, time, d_model)
         │                               │
         └───────────────┬───────────────┘
                         ▼
                   Element-wise Addition
                         │
                         ▼
                    ┌─────────────┐
                    │  Dropout    │
                    │  (optional) │
                    └──────┬──────┘
                           │
                           ▼
          Output: (batch, time, d_model)
```

## 💡 Why Use This Layer?

| Scenario | Challenge | Solution |
|----------|-----------|----------|
| **Multi-Modal Data** | Combining values + temporal | ✨ **Unified embedding** |
| **Calendar Effects** | Day-of-week, seasonal patterns | 🗓️ **Temporal feature support** |
| **Overfit Prevention** | Training instability | 🎯 **Integrated dropout** |
| **Flexible Input** | Optional temporal features | ⚡ **Handles both cases** |
| **End-to-End Learning** | Manual feature engineering | 🧠 **Learned representations** |

## 📊 Use Cases

- **Load Forecasting**: Energy demand with hourly/seasonal patterns
- **Stock Price Prediction**: Historical prices + trading hour patterns
- **Weather Forecasting**: Temperature trends + time-of-day cycles
- **Traffic Prediction**: Volume trends + weekday/rush hour effects
- **Retail Sales**: Historical sales + holiday/weekend features
- **Healthcare**: Patient metrics + time-of-day variations
- **IoT Sensors**: Sensor readings + temporal context

## 🚀 Quick Start

### Basic Usage

```python
import keras
from kerasfactory.layers import DataEmbeddingWithoutPosition

# Create the embedding layer
data_emb = DataEmbeddingWithoutPosition(
    c_in=7,              # Number of input features
    d_model=64,          # Output embedding dimension
    dropout=0.1          # Regularization
)

# Input data
x = keras.random.normal((32, 96, 7))      # 96 timesteps, 7 features
x_mark = keras.random.uniform(
    (32, 96, 5), 
    minval=0, 
    maxval=24, 
    dtype='int32'
)  # Temporal features

# Combine embeddings
output = data_emb([x, x_mark])

print(f"Input shape: {x.shape}")           # (32, 96, 7)
print(f"Temporal shape: {x_mark.shape}")   # (32, 96, 5)
print(f"Output shape: {output.shape}")     # (32, 96, 64)
```

### Without Temporal Features

```python
# Layer automatically handles missing temporal info
output = data_emb(x)  # Skips temporal embedding
print(output.shape)   # (32, 96, 64)
```

### In a Complete Forecasting Pipeline

```python
from kerasfactory.layers import DataEmbeddingWithoutPosition, PositionalEmbedding

def create_time_series_model():
    # Inputs
    x_input = keras.Input(shape=(96, 7))  # Values
    x_mark_input = keras.Input(shape=(96, 5))  # Temporal features
    
    # Embed values + temporal features
    x_emb = DataEmbeddingWithoutPosition(
        c_in=7, d_model=64, dropout=0.1
    )([x_input, x_mark_input])
    
    # Add positional encoding
    pos_emb = PositionalEmbedding(max_len=96, d_model=64)(x_emb)
    x = x_emb + pos_emb
    
    # Transformer blocks
    x = keras.layers.MultiHeadAttention(
        num_heads=8, key_dim=8
    )(x, x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.LayerNormalization()(x)
    
    # Forecast layer
    outputs = keras.layers.Dense(7)(x)  # Predict next values
    
    return keras.Model([x_input, x_mark_input], outputs)

model = create_time_series_model()
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

### With Custom Dropout Rates

```python
# High dropout for noisy data
data_emb_high_dropout = DataEmbeddingWithoutPosition(
    c_in=8, d_model=96, dropout=0.3
)

# Low dropout for clean data
data_emb_low_dropout = DataEmbeddingWithoutPosition(
    c_in=8, d_model=96, dropout=0.05
)

output1 = data_emb_high_dropout([x, x_mark])
output2 = data_emb_low_dropout([x, x_mark])
```

### Multi-Scale Embedding Ensemble

```python
class MultiScaleEmbedding(keras.layers.Layer):
    def __init__(self, c_in, d_model=64):
        super().__init__()
        self.embed1 = DataEmbeddingWithoutPosition(c_in, d_model, dropout=0.1)
        self.embed2 = DataEmbeddingWithoutPosition(c_in, d_model//2, dropout=0.1)
        self.embed3 = DataEmbeddingWithoutPosition(c_in, d_model//4, dropout=0.1)
    
    def call(self, inputs):
        if isinstance(inputs, list):
            x, x_mark = inputs
        else:
            x = inputs
            x_mark = None
        
        e1 = self.embed1([x, x_mark] if x_mark is not None else x)
        e2 = self.embed2([x, x_mark] if x_mark is not None else x)
        e3 = self.embed3([x, x_mark] if x_mark is not None else x)
        
        # Concatenate or combine embeddings
        return keras.layers.concatenate([e1, e2, e3], axis=-1)
```

## 🔧 API Reference

### DataEmbeddingWithoutPosition

```python
kerasfactory.layers.DataEmbeddingWithoutPosition(
    c_in: int,
    d_model: int,
    dropout: float = 0.0,
    embed_type: str = 'fixed',
    freq: str = 'h',
    name: str | None = None,
    **kwargs: Any
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `c_in` | `int` | — | Number of input channels (features) |
| `d_model` | `int` | — | Output embedding dimension |
| `dropout` | `float` | 0.0 | Dropout rate for regularization |
| `embed_type` | `str` | 'fixed' | Embedding type: 'fixed' or 'learned' |
| `freq` | `str` | 'h' | Frequency for temporal features: 'h'(hourly), 't'(minute), 'd'(daily), etc. |
| `name` | `str \| None` | None | Optional layer name |

#### Input Shape
- **Option 1** (with temporal): List of 2 tensors
  - `x`: `(batch_size, time_steps, c_in)` - Raw time series values
  - `x_mark`: `(batch_size, time_steps, 5)` - Temporal features [month, day, weekday, hour, minute]
- **Option 2** (values only): Single tensor
  - `x`: `(batch_size, time_steps, c_in)` - Raw time series values

#### Output Shape
- `(batch_size, time_steps, d_model)` - Combined embeddings

#### Returns
- Fused value and temporal embeddings with dropout applied

## 📈 Performance Characteristics

- **Time Complexity**: O(time_steps × c_in × d_model + time_steps × d_model)
- **Space Complexity**: O(c_in × d_model + temporal_vocab_sizes)
- **Trainable Parameters**: ~c_in × d_model × kernel_size + embedding_params
- **Inference Speed**: Fast, both embeddings computed in parallel
- **Memory Efficient**: Shared d_model dimension for both inputs

## 🎨 Advanced Usage

### Dynamic Dropout During Training

```python
class AdaptiveDataEmbedding(keras.layers.Layer):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.data_emb = DataEmbeddingWithoutPosition(c_in, d_model)
        self.adaptive_dropout = keras.layers.Dropout(0.2)
    
    def call(self, inputs, training=None):
        embeddings = self.data_emb(inputs)
        return self.adaptive_dropout(embeddings, training=training)
```

### With Layer Normalization

```python
def create_normalized_embedding():
    inputs_x = keras.Input(shape=(96, 7))
    inputs_mark = keras.Input(shape=(96, 5))
    
    # Embed
    x = DataEmbeddingWithoutPosition(
        c_in=7, d_model=64, dropout=0.1
    )([inputs_x, inputs_mark])
    
    # Normalize
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    return keras.Model([inputs_x, inputs_mark], x)
```

### Conditional Temporal Features

```python
def embed_with_optional_temporal(x, x_mark=None):
    data_emb = DataEmbeddingWithoutPosition(c_in=7, d_model=64)
    
    if x_mark is not None:
        return data_emb([x, x_mark])
    else:
        # Use only value embeddings
        return data_emb(x)

# Usage
x = keras.random.normal((32, 96, 7))
x_mark = keras.random.uniform((32, 96, 5), minval=0, maxval=24, dtype='int32')

output_with_temporal = embed_with_optional_temporal(x, x_mark)
output_without_temporal = embed_with_optional_temporal(x)
```

## 🔍 Visual Representation

```
┌─────────────────────────────────────────────────────────┐
│           DataEmbeddingWithoutPosition                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   INPUT 1: Values              INPUT 2: Temporal       │
│   (batch, time, c_in)          (batch, time, 5)        │
│          │                              │               │
│          ▼                              ▼               │
│   ┌────────────────┐          ┌──────────────────┐     │
│   │ TokenEmbedding │          │TemporalEmbedding │     │
│   │   Conv1D       │          │ Month/Day/...    │     │
│   └────────┬───────┘          └────────┬─────────┘     │
│            │                           │                │
│     (b, t, d_model)            (b, t, d_model)         │
│            │                           │                │
│            └───────────────┬───────────┘                │
│                            ▼                            │
│                  Element-wise Addition                  │
│                            │                            │
│                            ▼                            │
│                     ┌─────────────┐                     │
│                     │   Dropout   │                     │
│                     │    (p=...)  │                     │
│                     └─────┬───────┘                     │
│                           │                             │
│   OUTPUT: (batch, time, d_model)                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 💡 Best Practices

1. **Matching Dimensions**: Ensure d_model matches downstream layers
2. **Dropout Tuning**: 0.1-0.2 for most cases, increase for noisy data
3. **Temporal Features Order**: month → day → weekday → hour → minute
4. **Embed Type Selection**: 'fixed' for speed, 'learned' for accuracy
5. **Frequency Setting**: Match your data frequency (h/d/t)
6. **Optional Features**: Handles missing x_mark gracefully
7. **Normalization**: Consider LayerNorm after embedding

## ⚠️ Common Pitfalls

- ❌ **Wrong temporal range**: month should be 0-12, hour 0-23, minute 0-59
- ❌ **Mismatched c_in**: Using wrong feature count causes shape errors
- ❌ **No dropout**: Missing regularization can lead to overfitting
- ❌ **Ignoring x_mark**: Always provide temporal features when available
- ❌ **Wrong freq setting**: Mismatch between frequency and data sampling
- ❌ **d_model too small**: Underfitting with small embedding dimension

## 📚 References

- Vaswani, A., et al. (2017). "Attention Is All You Need"
- Zhou, H., et al. (2021). "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"
- Das, S., et al. (2023). "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"

## 🔗 Related Layers

- [`TokenEmbedding`](token-embedding.md) - Value embedding component
- [`TemporalEmbedding`](temporal-embedding.md) - Temporal embedding component
- [`PositionalEmbedding`](positional-embedding.md) - Add positional encoding
- [`ReversibleInstanceNorm`](reversible-instance-norm.md) - Normalize before embedding
- [`PastDecomposableMixing`](past-decomposable-mixing.md) - Main encoder block

## ✅ Serialization

```python
# Get configuration
config = data_emb.get_config()

# Save to file
import json
with open('data_embedding_config.json', 'w') as f:
    json.dump(config, f)

# Recreate from config
new_layer = DataEmbeddingWithoutPosition.from_config(config)
```

## 🧪 Testing & Validation

```python
import keras

# Test 1: With temporal features
data_emb = DataEmbeddingWithoutPosition(c_in=7, d_model=64, dropout=0.1)
x = keras.random.normal((32, 96, 7))
x_mark = keras.random.uniform((32, 96, 5), minval=0, maxval=24, dtype='int32')
output = data_emb([x, x_mark])
assert output.shape == (32, 96, 64), "Shape mismatch with temporal features"

# Test 2: Without temporal features
output_no_temporal = data_emb(x)
assert output_no_temporal.shape == (32, 96, 64), "Shape mismatch without temporal"

# Test 3: Different batch sizes
x_small = keras.random.normal((1, 96, 7))
x_large = keras.random.normal((256, 96, 7))
assert data_emb(x_small).shape[0] == 1
assert data_emb(x_large).shape[0] == 256

# Test 4: Different time steps
x_short = keras.random.normal((32, 24, 7))
x_long = keras.random.normal((32, 256, 7))
assert data_emb(x_short).shape[1] == 24
assert data_emb(x_long).shape[1] == 256

print("✓ All validation tests passed!")
```

---

**Last Updated**: 2025-11-04  
**Version**: 1.0  
**Keras**: 3.0+  
**Status**: ✅ Production Ready
