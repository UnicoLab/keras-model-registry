---
title: TemporalEmbedding - KMR
description: Embedding layer for temporal features with support for fixed and learned embeddings
keywords: [temporal embedding, time features, month, day, hour, embeddings, keras, time series]
---

# 🕐 TemporalEmbedding

<div class="layer-hero">
  <div class="layer-hero-content">
    <h1>🕐 TemporalEmbedding</h1>
    <div class="layer-badges">
      <span class="badge badge-intermediate">🟡 Intermediate</span>
      <span class="badge badge-stable">✅ Stable</span>
      <span class="badge badge-timeseries">⏱️ Time Series</span>
    </div>
  </div>
</div>

## 🎯 Overview

The `TemporalEmbedding` layer embeds temporal/calendar features (month, day, weekday, hour, minute) into a shared embedding space. It supports both:

1. **Fixed Embeddings**: Pre-defined sinusoidal patterns (no parameters)
2. **Learned Embeddings**: Trainable embeddings optimized for your task

Perfect for capturing:
- **Seasonal Patterns**: Monthly, weekly, daily cycles
- **Hourly Effects**: Rush hours, off-peak hours
- **Calendar Effects**: Holidays, weekends, special events
- **Time-of-Day Variations**: Energy demand, traffic patterns

## 🔍 How It Works

```
Input: Temporal Features
[month, day, weekday, hour, minute]
       |
       ├──> month_embed (0-12)
       ├──> day_embed (0-31)
       ├──> weekday_embed (0-6)
       ├──> hour_embed (0-23)
       └──> minute_embed (0-59) [if freq='t']
       |
       V
All embeddings: (batch, time, d_model)
       |
       └──> Element-wise Addition
       |
       V
Output: (batch, time, d_model)
```

Each temporal component is embedded independently, then summed to create a combined representation.

## 💡 Why Use This Layer?

| Scenario | Fixed | Learned | Result |
|----------|-------|---------|--------|
| **Fast Training** | ✅ No params | ❌ Slower | Use Fixed |
| **Accuracy** | ⚠️ Limited | ✅ Optimal | Use Learned |
| **Transfer Learning** | ✅ Generic | ⚠️ Task-specific | Use Fixed |
| **Data Scarcity** | ✅ Better | ❌ Overfits | Use Fixed |

## 📊 Use Cases

- **Load Forecasting**: Hour and month embeddings for energy demand
- **Traffic Prediction**: Weekday and hour-of-day patterns
- **Retail Sales**: Weekend/holiday effects, seasonal trends
- **Weather**: Seasonal patterns, daily cycles
- **Stock Market**: Trading hours, day-of-week effects
- **Healthcare**: Time-of-day symptoms, seasonal diseases

## 🚀 Quick Start

```python
import keras
from kmr.layers import TemporalEmbedding

# Create temporal embedding layer
temp_emb = TemporalEmbedding(
    d_model=64,
    embed_type='fixed',  # or 'learned'
    freq='h'             # hourly frequency
)

# Input temporal features: [month, day, weekday, hour, minute]
x_mark = keras.stack([
    keras.random.uniform((32, 96), minval=0, maxval=12, dtype='int32'),   # month
    keras.random.uniform((32, 96), minval=0, maxval=31, dtype='int32'),   # day
    keras.random.uniform((32, 96), minval=0, maxval=7, dtype='int32'),    # weekday
    keras.random.uniform((32, 96), minval=0, maxval=24, dtype='int32'),   # hour
], axis=-1)

# Get embeddings
output = temp_emb(x_mark)
print(output.shape)  # (32, 96, 64)
```

## 🔧 API Reference

```python
kmr.layers.TemporalEmbedding(
    d_model: int,
    embed_type: str = 'fixed',
    freq: str = 'h',
    name: str | None = None,
    **kwargs: Any
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | `int` | — | Output embedding dimension |
| `embed_type` | `str` | 'fixed' | 'fixed' or 'learned' embeddings |
| `freq` | `str` | 'h' | Frequency: 'h'(hourly), 'd'(daily), 't'(minutely) |
| `name` | `str \| None` | None | Optional layer name |

### Input Shape
- `(batch_size, time_steps, 5)` or `(batch_size, time_steps, 4)`
- Channels: [month(0-12), day(0-31), weekday(0-6), hour(0-23), minute(0-59)]

### Output Shape
- `(batch_size, time_steps, d_model)`

## 💡 Best Practices

1. **Choose Embed Type**: Fixed for speed/generality, Learned for accuracy
2. **Match Frequency**: hourly (h) / daily (d) / minutely (t)
3. **Proper Ranges**: month(1-12), day(1-31), weekday(0-6), hour(0-23)
4. **Combine with Values**: Use with TokenEmbedding for full context
5. **Layer Norm**: Consider LayerNorm after embedding

## ⚠️ Common Pitfalls

- ❌ **Out-of-range indices**: month>12, hour>23 causes embedding errors
- ❌ **Wrong frequency**: Mismatch between data and freq setting
- ❌ **Missing minute**: If freq='t', must provide 5 channels
- ❌ **Unused embeddings**: If not using minutes, set freq='h'

## 📚 References

- Vaswani, A., et al. (2017). "Attention Is All You Need"
- Zhou, H., et al. (2021). "Informer: Beyond Efficient Transformer"

## 🔗 Related Layers

- [`FixedEmbedding`](fixed-embedding.md) - Individual fixed embeddings
- [`TokenEmbedding`](token-embedding.md) - Value embeddings
- [`DataEmbeddingWithoutPosition`](data-embedding-without-position.md) - Combined embeddings

---

**Last Updated**: 2025-11-04 | **Keras**: 3.0+ | **Status**: ✅ Production Ready
