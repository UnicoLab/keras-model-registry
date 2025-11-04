---
title: SeriesDecomposition - KMR
description: Trend-seasonal decomposition layer using moving average for time series analysis
keywords: [decomposition, trend, seasonal, time series, moving average, keras]
---

# 🔀 SeriesDecomposition

<div class="layer-hero">
  <div class="layer-hero-content">
    <h1>🔀 SeriesDecomposition</h1>
    <div class="layer-badges">
      <span class="badge badge-intermediate">🟡 Intermediate</span>
      <span class="badge badge-stable">✅ Stable</span>
      <span class="badge badge-timeseries">⏱️ Time Series</span>
    </div>
  </div>
</div>

## 🎯 Overview

The `SeriesDecomposition` layer decomposes time series into trend and seasonal components using moving average. This is a fundamental technique in time series analysis that separates long-term trends from recurring patterns.

The layer:
- **Extracts Trend**: Using moving average smoothing
- **Captures Seasonality**: As residual after trend removal
- **Preserves Information**: No information loss in decomposition
- **Enables Hierarchical Analysis**: Process components separately

## 🔍 How It Works

```
Input Time Series
       |
       V
+------------------+
| Moving Average   | <- Extracts trend
+--------+--------+
         |
         V
    Trend Component
         |
Input - Trend
         |
         V
Seasonal Component
```

The trend is computed using a moving average filter, preserving temporal length through edge padding.

## 💡 Why Use This Layer?

| Problem | Solution |
|---------|----------|
| **Mixed Patterns** | Separate trend and seasonality |
| **Noisy Data** | Trend extraction via smoothing |
| **Pattern Analysis** | Analyze components independently |
| **Forecasting** | Model trend and seasonal separately |

## 📊 Use Cases

- **Classical Time Series Analysis**: Traditional decomposition
- **Trend Forecasting**: Separate trend prediction
- **Seasonal Adjustment**: Remove seasonality for analysis
- **Anomaly Detection**: Decompose before detection
- **Feature Engineering**: Use components as features

## 🚀 Quick Start

```python
from kmr.layers import SeriesDecomposition
import keras

# Create decomposition layer
decomp = SeriesDecomposition(kernel_size=25)

# Input data
x = keras.random.normal((32, 100, 8))

# Decompose
seasonal, trend = decomp(x)

print(f"Seasonal shape: {seasonal.shape}")  # (32, 100, 8)
print(f"Trend shape: {trend.shape}")        # (32, 100, 8)

# Verify decomposition: seasonal + trend ≈ original
reconstructed = seasonal + trend  # Approximately equals x
```

## 🔧 API Reference

```python
kmr.layers.SeriesDecomposition(
    kernel_size: int,
    name: str | None = None,
    **kwargs
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `kernel_size` | `int` | Moving average window size |
| `name` | `str \| None` | Optional layer name |

### Returns

Tuple of (seasonal, trend) tensors with same shape as input.

## 💡 Best Practices

1. **Kernel Size**: Choose based on seasonality frequency
   - Daily data: kernel_size=7 (weekly pattern)
   - Monthly data: kernel_size=12 (annual pattern)
2. **Edge Handling**: Automatically paddles edges to preserve length
3. **Multiple Scales**: Apply recursively for hierarchical decomposition
4. **Information Preservation**: Guaranteed: seasonal + trend = original

## ⚠️ Common Pitfalls

- ❌ **Small kernel_size**: Misses true trends
- ❌ **Large kernel_size**: Removes important patterns
- ❌ **Wrong frequency**: Choose kernel based on domain knowledge
- ❌ **Assuming perfect reconstruction**: Numerical precision limits

## 📚 References

- Classical time series decomposition (Additive/Multiplicative)
- Cleveland et al. (1990). "STL: A Seasonal-Trend Decomposition"
- Hyndman & Athanasopoulos. "Forecasting: Principles and Practice"

## 🔗 Related Layers

- [`DFTSeriesDecomposition`](dft-series-decomposition.md) - Frequency-based decomposition
- [`MovingAverage`](moving-average.md) - Trend extraction component
- [`MultiScaleSeasonMixing`](multi-scale-season-mixing.md) - Process seasonal components

---

**Last Updated**: 2025-11-04 | **Keras**: 3.0+ | **Status**: ✅ Production Ready
