---
title: MultiScaleSeasonMixing - KMR
description: Bottom-up multi-scale seasonal pattern mixing and downsampling
keywords: [multi-scale, seasonal, mixing, hierarchical, time series, keras]
---

# 🌊 MultiScaleSeasonMixing

<div class="layer-hero">
  <div class="layer-hero-content">
    <h1>🌊 MultiScaleSeasonMixing</h1>
    <div class="layer-badges">
      <span class="badge badge-intermediate">🟡 Intermediate</span>
      <span class="badge badge-stable">✅ Stable</span>
      <span class="badge badge-timeseries">⏱️ Time Series</span>
    </div>
  </div>
</div>

## 🎯 Overview

The `MultiScaleSeasonMixing` layer mixes seasonal patterns across multiple time scales in a **bottom-up** (coarse-to-fine) fashion. It:

1. **Downsamples** seasonal patterns to coarser scales
2. **Applies Dense Transformations** at each scale
3. **Combines** information from multiple scales
4. **Produces Multi-Scale Representations** of seasonality

Used as part of TimeMixer's encoder to capture seasonality at different resolutions.

## 🔍 How It Works

```
Input Seasonal Patterns (Fine Scale)
            |
            V
    +-------------------+
    | Apply Dense       |
    | Transformations   |
    +--------+---------+
             |
             V
    Output Scale 1 (Fine)
             |
             V
    +-------------------+
    | Downsample x2     |
    +--------+---------+
             |
             V
    +-------------------+
    | Apply Dense       |
    | Transformations   |
    +--------+---------+
             |
             V
    Output Scale 2 (Coarser)
             |
             V
            ...
```

## 💡 Why Use This Layer?

| Challenge | Solution |
|-----------|----------|
| **Single Scale** | ✅ Multi-scale analysis |
| **Loss of Detail** | ✅ Bottom-up blending |
| **Seasonal Complexity** | ✅ Hierarchical patterns |

## 📊 Use Cases

- **Multi-Seasonal Data**: Multiple overlapping seasonal patterns
- **Hierarchical Forecasting**: Predictions at different granularities
- **Pattern Discovery**: Seasonal patterns at various scales
- **TimeMixer Encoder**: Core component of forecasting model

## 🚀 Quick Start

```python
import keras
from kmr.layers import MultiScaleSeasonMixing

# Create seasonal mixing layer
season_mix = MultiScaleSeasonMixing(
    seq_len=96,
    down_sampling_window=2,
    down_sampling_layers=2
)

# Input: list of seasonal patterns at different scales
x_list = [keras.random.normal((32, 96, 64))]

# Mix across scales
output = season_mix(x_list)
print(len(output))  # Number of output scales
```

## 🔧 API Reference

```python
kmr.layers.MultiScaleSeasonMixing(
    seq_len: int,
    down_sampling_window: int = 2,
    down_sampling_layers: int = 1,
    name: str | None = None,
    **kwargs: Any
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seq_len` | `int` | — | Sequence length |
| `down_sampling_window` | `int` | 2 | Downsampling factor |
| `down_sampling_layers` | `int` | 1 | Number of downsampling layers |
| `name` | `str \| None` | None | Optional layer name |

### Input
- List of tensors, each shape `(batch, channels, seq_len)`

### Output
- List of mixed seasonal patterns at multiple scales

## 💡 Best Practices

1. **Down-sampling Factor**: 2-4 typical values
2. **Number of Layers**: 1-3 for most cases
3. **Sequence Length**: Must be divisible by downsampling factors
4. **Input Order**: Pass finest to coarsest scales

## ⚠️ Common Pitfalls

- ❌ **Non-divisible seq_len**: Causes shape mismatches
- ❌ **Too many layers**: Loss of fine-scale information
- ❌ **Wrong input format**: List required, not concatenated

## 📚 References

- Zhou, T., et al. (2023). "TimeMixer: Decomposing Time Series for Forecasting"

## 🔗 Related Layers

- [`MultiScaleTrendMixing`](multi-scale-trend-mixing.md) - Trend version (top-down)
- [`PastDecomposableMixing`](past-decomposable-mixing.md) - Main encoder block

---

**Last Updated**: 2025-11-04 | **Keras**: 3.0+ | **Status**: ✅ Production Ready
