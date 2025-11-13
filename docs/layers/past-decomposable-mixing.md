---
title: PastDecomposableMixing - KerasFactory
description: Past decomposable mixing encoder block combining trend-seasonal decomposition and multi-scale mixing
keywords: [decomposition, mixing, encoder, time series, keras, forecasting]
---

# 🔀 PastDecomposableMixing

<div class="layer-hero">
  <div class="layer-hero-content">
    <h1>🔀 PastDecomposableMixing</h1>
    <div class="layer-badges">
      <span class="badge badge-advanced">🔴 Advanced</span>
      <span class="badge badge-stable">✅ Stable</span>
      <span class="badge badge-timeseries">⏱️ Time Series</span>
    </div>
  </div>
</div>

## 🎯 Overview

The `PastDecomposableMixing` layer is the core encoder block of TimeMixer. It combines:

1. **Series Decomposition**: Splits input into trend and seasonal components
2. **Multi-Scale Mixing**: Processes each component at multiple scales
3. **Cross-Component Learning**: Shared dense transformations between components
4. **Hierarchical Representation**: Captures patterns at different resolutions

This is the key innovation of TimeMixer - decomposable, multi-scale mixing for time series.

## 🔍 How It Works

```
Input Time Series
        |
        V
    Decomposition
    /            \
   /              \
  V                V
Trend         Seasonal
  |                |
  V                V
[Multi-Scale]  [Multi-Scale]
[Trend Mixing] [Season Mixing]
  |                |
  V                V
Trend Outputs  Seasonal Outputs
  |                |
  +-------- Output --------+
```

## 💡 Why Use This Layer?

| Advantage | Benefit |
|-----------|---------|
| **Decomposable** | Treat trend/seasonal separately |
| **Multi-Scale** | Capture patterns at different resolutions |
| **Efficient** | Reduced parameters vs monolithic |
| **Interpretable** | Understand which component contributes |

## 📊 Use Cases

- **Time Series Forecasting**: Primary encoder for TimeMixer
- **Multi-Scale Analysis**: Hierarchical pattern extraction
- **Decomposable Models**: Separable trend/seasonal processing
- **Long Sequence Forecasting**: Efficient multi-scale handling

## 🚀 Quick Start

```python
import keras
from kerasfactory.layers import PastDecomposableMixing

pdm = PastDecomposableMixing(
    seq_len=96,
    pred_len=12,
    down_sampling_window=2,
    down_sampling_layers=1,
    d_model=64,
    dropout=0.1,
    channel_independence=0,
    decomp_method='moving_avg',
    d_ff=256,
    moving_avg=25,
    top_k=5
)

# Input list of tensors
x_list = [keras.random.normal((32, 96, 64))]

# Process through encoder block
outputs = pdm(x_list)
print(len(outputs))  # Number of output scales
```

## 🔧 API Reference

```python
kerasfactory.layers.PastDecomposableMixing(
    seq_len: int,
    pred_len: int,
    down_sampling_window: int = 2,
    down_sampling_layers: int = 1,
    d_model: int = 64,
    dropout: float = 0.1,
    channel_independence: int = 0,
    decomp_method: str = 'moving_avg',
    d_ff: int = 256,
    moving_avg: int = 25,
    top_k: int = 5,
    name: str | None = None,
    **kwargs: Any
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seq_len` | `int` | — | Input sequence length |
| `pred_len` | `int` | — | Prediction length |
| `down_sampling_window` | `int` | 2 | Downsampling factor |
| `down_sampling_layers` | `int` | 1 | Number of scales |
| `d_model` | `int` | 64 | Model dimension |
| `dropout` | `float` | 0.1 | Dropout rate |
| `channel_independence` | `int` | 0 | Channel processing mode |
| `decomp_method` | `str` | 'moving_avg' | 'moving_avg' or 'dft' |
| `d_ff` | `int` | 256 | Feed-forward dimension |
| `moving_avg` | `int` | 25 | Moving average window |
| `top_k` | `int` | 5 | Top-k frequencies for DFT |

### Input
- List of tensors at different scales

### Output
- List of processed tensors at multiple scales

## 💡 Best Practices

1. **Decomposition Choice**: 'moving_avg' for speed, 'dft' for accuracy
2. **Scales**: 1-3 layers typical, more for very long sequences
3. **Channel Independence**: 0 for coupled, 1 for independent
4. **Down-sampling Factor**: Usually 2, can be 3-4 for long sequences
5. **Dropout Tuning**: 0.05-0.2 depending on data size

## ⚠️ Common Pitfalls

- ❌ **Too many scales**: Information loss in very coarse scales
- ❌ **Incompatible seq_len**: Must be divisible by sampling factors
- ❌ **Wrong decomp_method**: Mismatch with data characteristics
- ❌ **Unbalanced dropout**: Too high causes underfitting

## 📚 References

- Zhou, T., et al. (2023). "TimeMixer: Decomposing Time Series for Forecasting"
- Multi-scale processing for time series

## 🔗 Related Layers

- [`SeriesDecomposition`](series-decomposition.md) - Decomposition component
- [`DFTSeriesDecomposition`](dft-series-decomposition.md) - FFT-based decomposition
- [`MultiScaleSeasonMixing`](multi-scale-season-mixing.md) - Seasonal mixing
- [`MultiScaleTrendMixing`](multi-scale-trend-mixing.md) - Trend mixing

---

**Last Updated**: 2025-11-04 | **Keras**: 3.0+ | **Status**: ✅ Production Ready
