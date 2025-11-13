---
title: MultiScaleTrendMixing - KerasFactory
description: Top-down multi-scale trend pattern mixing and upsampling
keywords: [multi-scale, trend, mixing, hierarchical, time series, keras]
---

# 📈 MultiScaleTrendMixing

<div class="layer-hero">
  <div class="layer-hero-content">
    <h1>📈 MultiScaleTrendMixing</h1>
    <div class="layer-badges">
      <span class="badge badge-intermediate">🟡 Intermediate</span>
      <span class="badge badge-stable">✅ Stable</span>
      <span class="badge badge-timeseries">⏱️ Time Series</span>
    </div>
  </div>
</div>

## 🎯 Overview

The `MultiScaleTrendMixing` layer mixes trend patterns across multiple time scales in a **top-down** (fine-to-coarse) fashion. It:

1. **Upsamples** trend patterns from coarser scales
2. **Applies Dense Transformations** at each scale
3. **Combines** information from multiple scales
4. **Produces Multi-Scale Representations** of trends

Complements MultiScaleSeasonMixing for complete TimeMixer encoding.

## 🔍 How It Works

```
Input Coarse Trend
         |
         V
    [Apply Dense]
         |
         V
Output Coarse Scale
         |
         V
    [Upsample x2]
         |
         V
    [Apply Dense]
         |
         V
Output Medium Scale
         |
         V
      ...
```

## 💡 Why Use This Layer?

Multi-scale trend analysis captures:
- **Long-term patterns** at coarse scales
- **Short-term variations** at fine scales
- **Hierarchical structure** of trends

## 📊 Use Cases

- **Multi-Horizon Forecasting**: Different trend scales
- **Anomaly Detection**: Trend changes at multiple scales
- **TimeMixer Encoder**: Core component for trend decomposition

## 🚀 Quick Start

```python
import keras
from kerasfactory.layers import MultiScaleTrendMixing

trend_mix = MultiScaleTrendMixing(
    seq_len=96,
    down_sampling_window=2,
    down_sampling_layers=2
)

x_list = [keras.random.normal((32, 96, 64))]
output = trend_mix(x_list)
print(len(output))
```

## 🔧 API Reference

```python
kerasfactory.layers.MultiScaleTrendMixing(
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
| `down_sampling_window` | `int` | 2 | Sampling factor |
| `down_sampling_layers` | `int` | 1 | Number of layers |
| `name` | `str \| None` | None | Optional layer name |

## 🔗 Related Layers

- [`MultiScaleSeasonMixing`](multi-scale-season-mixing.md) - Seasonal version (bottom-up)
- [`PastDecomposableMixing`](past-decomposable-mixing.md) - Main encoder block

---

**Last Updated**: 2025-11-04 | **Keras**: 3.0+ | **Status**: ✅ Production Ready
