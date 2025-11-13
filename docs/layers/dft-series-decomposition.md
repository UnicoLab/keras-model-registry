---
title: DFTSeriesDecomposition - KerasFactory
description: Frequency-based series decomposition using Discrete Fourier Transform
keywords: [decomposition, FFT, frequency domain, time series, seasonal, trend, keras]
---

# 🔢 DFTSeriesDecomposition

<div class="layer-hero">
  <div class="layer-hero-content">
    <h1>🔢 DFTSeriesDecomposition</h1>
    <div class="layer-badges">
      <span class="badge badge-intermediate">🟡 Intermediate</span>
      <span class="badge badge-stable">✅ Stable</span>
      <span class="badge badge-timeseries">⏱️ Time Series</span>
    </div>
  </div>
</div>

## 🎯 Overview

The `DFTSeriesDecomposition` layer decomposes time series into seasonal and trend components using frequency-domain analysis. It extracts:

1. **Seasonal Component**: Periodic patterns at specific frequencies
2. **Trend Component**: Smooth, long-term variations

Based on the Discrete Fourier Transform (FFT) for extracting dominant frequencies.

## 🔍 How It Works

```
Input Time Series
       |
       V
FFT (Frequency Domain)
       |
       |---> Top-k Frequencies (Seasonal)
       |
       |---> Remaining (Trend + Noise)
       |
       V
Inverse FFT
       |
       |---> Seasonal Component
       |
       |---> Trend Component
```

## 💡 Why Use This Layer?

| vs. Moving Average | vs. STL | DFT Advantage |
|---|---|---|
| Less precise | More complex | ✅ **Explicit frequency** |
| Misses patterns | Slow | ✅ **FFT fast** |
| Limited accuracy | Hard tune | ✅ **Data-driven** |

## 📊 Use Cases

- **Periodic Pattern Detection**: Identify exact frequencies
- **Multi-seasonal Data**: Multiple seasonal periods
- **Spectral Analysis**: Frequency domain insights
- **Denoising**: Separate signal from noise
- **Anomaly Detection**: Detect frequency shifts

## 🚀 Quick Start

```python
import keras
from kerasfactory.layers import DFTSeriesDecomposition

# Create decomposition layer
dft_decomp = DFTSeriesDecomposition(top_k=5)

# Input time series
x = keras.random.normal((32, 100, 8))

# Decompose
seasonal, trend = dft_decomp(x)

print(f"Seasonal shape: {seasonal.shape}")  # (32, 100, 8)
print(f"Trend shape: {trend.shape}")        # (32, 100, 8)

# Verify: seasonal + trend ≈ original
reconstructed = seasonal + trend
```

## 🔧 API Reference

```python
kerasfactory.layers.DFTSeriesDecomposition(
    top_k: int = 5,
    name: str | None = None,
    **kwargs: Any
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_k` | `int` | 5 | Number of top frequencies to retain |
| `name` | `str \| None` | None | Optional layer name |

### Input Shape
- `(batch_size, time_steps, channels)`

### Output Shape
- Tuple of `(seasonal, trend)` each with shape `(batch_size, time_steps, channels)`

## 💡 Best Practices

1. **Top-k Selection**: Usually 3-10 for most applications
2. **Data Length**: Longer series yield better frequency estimates
3. **Preprocessing**: Normalize data before decomposition
4. **Combine**: Use with trend analysis for multi-scale patterns
5. **Validation**: Check seasonal patterns make domain sense

## ⚠️ Common Pitfalls

- ❌ **Small top_k**: May miss important patterns
- ❌ **Large top_k**: Too much noise/overfitting
- ❌ **Non-stationary data**: Apply differencing first
- ❌ **Aliasing**: Ensure proper sampling frequency

## 📚 References

- Cooley, J.W. & Tukey, J.W. (1965). "An algorithm for the machine computation of complex Fourier series"
- Zhou, H., et al. (2023). "TimeMixer: Decomposing Time Series for Forecasting"

## 🔗 Related Layers

- [`SeriesDecomposition`](series-decomposition.md) - Moving average method
- [`MovingAverage`](moving-average.md) - Trend extraction
- [`MultiScaleSeasonMixing`](multi-scale-season-mixing.md) - Process seasonal patterns

---

**Last Updated**: 2025-11-04 | **Keras**: 3.0+ | **Status**: ✅ Production Ready
