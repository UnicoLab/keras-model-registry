---
title: ReversibleInstanceNormMultivariate - KerasFactory
description: Multivariate reversible instance normalization with batch-level statistics for time series
keywords: [normalization, instance norm, multivariate, reversible, time series, stability, keras]
---

# 🔄 ReversibleInstanceNormMultivariate

<div class="layer-hero">
  <div class="layer-hero-content">
    <h1>🔄 ReversibleInstanceNormMultivariate</h1>
    <div class="layer-badges">
      <span class="badge badge-advanced">🔴 Advanced</span>
      <span class="badge badge-stable">✅ Stable</span>
      <span class="badge badge-timeseries">⏱️ Time Series</span>
    </div>
  </div>
</div>

## 🎯 Overview

The `ReversibleInstanceNormMultivariate` layer extends reversible instance normalization to multivariate time series by computing statistics across the batch dimension. This is essential for scenarios where you need consistent normalization across multiple series with different scales.

Key features:
- **Batch-Level Normalization**: Computes mean/std across all samples in the batch
- **Reversible**: Exact denormalization preserves interpretability
- **Multivariate Support**: Handles multiple features simultaneously
- **Optional Affine**: Learnable scale and shift parameters
- **Training Stability**: Improves convergence with diverse scaling

## 🔍 How It Works

```
Input Time Series
(batch=B, time=T, features=F)
       |
       V
┌──────────────────────────────┐
│ Compute Batch Statistics     │
│ mean = mean(x, axis=[0,1])  │ <- Batch + Time
│ std = std(x, axis=[0,1])    │    (F,)
└──────────────────────────────┘
       |
       V
Normalize: (x - mean) / (std + eps)
       |
       V
Optional Affine: y * gamma + beta
       |
       V
Normalized Output (B, T, F)
```

The normalization uses statistics computed across both batch and time dimensions, creating a global normalization for the entire dataset.

## 💡 Why Use This Layer?

| Scenario | RevIN | RevIN Multivariate | Result |
|----------|-------|--------------------|--------|
| **Single Series** | ✅ Perfect | ⚠️ Overkill | Use RevIN |
| **Multiple Series** | ⚠️ Independent | ✅ Unified | Use RevINMulti |
| **Cross-Dataset** | ❌ Poor | ✅ Consistent | Use RevINMulti |
| **Scale Normalization** | ⚠️ Per-series | ✅ Global | Use RevINMulti |

## 📊 Use Cases

- **Multi-Sensor Forecasting**: Normalize multiple sensor readings together
- **Portfolio Returns**: Normalize stocks with different volatilities
- **Traffic Networks**: Normalize flows across multiple routes
- **Power Grids**: Normalize consumption across multiple substations
- **Climate Data**: Normalize multiple weather variables
- **Healthcare**: Normalize vital signs from multiple patients

## 🚀 Quick Start

```python
import keras
from kerasfactory.layers import ReversibleInstanceNormMultivariate

# Create normalization layer for multivariate data
normalizer = ReversibleInstanceNormMultivariate(num_features=5, affine=True)

# Input: batch of multivariate time series
x = keras.random.normal((32, 100, 5))  # 32 samples, 100 timesteps, 5 features

# Normalize for training
x_norm = normalizer(x, mode='norm')

# Use in model
# ... model forward pass ...

# Denormalize predictions
y_pred_norm = model(x_norm)
y_pred = normalizer(y_pred_norm, mode='denorm')
```

### Advanced Example: Multi-Scale Forecasting

```python
from kerasfactory.layers import ReversibleInstanceNormMultivariate

# Multiple scales with shared normalization
normalizer = ReversibleInstanceNormMultivariate(
    num_features=8,
    eps=1e-6,
    affine=True,
    name='multi_scale_norm'
)

# Different time scales
short_term = keras.random.normal((64, 24, 8))   # hourly
medium_term = keras.random.normal((64, 168, 8)) # weekly
long_term = keras.random.normal((64, 730, 8))   # yearly

# Normalize all with same statistics
short_norm = normalizer(short_term, mode='norm')
medium_norm = normalizer(medium_term, mode='norm')
long_norm = normalizer(long_term, mode='norm')

# Process separately
short_pred = short_model(short_norm)
medium_pred = medium_model(medium_norm)
long_pred = long_model(long_norm)

# Denormalize with same statistics
short_denorm = normalizer(short_pred, mode='denorm')
medium_denorm = normalizer(medium_pred, mode='denorm')
long_denorm = normalizer(long_pred, mode='denorm')
```

## 🔧 API Reference

```python
kerasfactory.layers.ReversibleInstanceNormMultivariate(
    num_features: int,
    eps: float = 1e-5,
    affine: bool = False,
    name: str | None = None,
    **kwargs
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_features` | `int` | — | Number of features/channels |
| `eps` | `float` | 1e-5 | Numerical stability constant |
| `affine` | `bool` | False | Learnable scale and shift parameters |
| `name` | `str \| None` | None | Optional layer name |

### Methods

#### `call(inputs, mode='norm')`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inputs` | `Tensor` | — | Input tensor (batch, time, features) |
| `mode` | `str` | 'norm' | 'norm' for normalization or 'denorm' for denormalization |

**Returns**: Normalized or denormalized tensor with same shape as input

### Input/Output Shapes

- **Input**: `(batch_size, time_steps, num_features)`
- **Output**: `(batch_size, time_steps, num_features)`

## 💡 Best Practices

1. **Batch Size**: Larger batches improve stability through better statistics
2. **Affine Transform**: Enable for flexible scaling in complex models
3. **Consistency**: Use same normalizer for train and inference
4. **Feature Scaling**: Handles features with different scales automatically
5. **Small eps**: Use eps=1e-6 for high precision, 1e-5 for stability
6. **Denormalization**: Always denormalize final predictions for interpretability

## ⚠️ Common Pitfalls

- ❌ **Different Normalizer**: Don't create new instance for inference
- ❌ **Forgetting Denormalization**: Loss of interpretability in predictions
- ❌ **Small Batch Size**: Poor statistics with batch_size < 16
- ❌ **Mode Confusion**: Mix up 'norm' and 'denorm' modes
- ❌ **Feature Dimension Mismatch**: Ensure consistent num_features

## 🔄 Comparison with RevIN

### ReversibleInstanceNorm
- Normalization per sample: `mean(x, axis=time)`
- Independent series processing
- Best for: Single series or independent datasets

### ReversibleInstanceNormMultivariate
- Normalization across batch: `mean(x, axis=[batch, time])`
- Unified statistics
- Best for: Related series or multi-sensor data

## 📚 References

- Instance Normalization (Ulyanov et al., 2016)
- RevIN for Time Series (Kim et al., 2021)
- Batch normalization concepts (Ioffe & Szegedy, 2015)

## 🔗 Related Layers

- [`ReversibleInstanceNorm`](reversible-instance-norm.md) - Per-sample normalization
- [`SeriesDecomposition`](series-decomposition.md) - Decompose before normalization
- [`DataEmbeddingWithoutPosition`](data-embedding-without-position.md) - Combined with embeddings

## 🧮 Mathematical Details

### Normalization Forward Pass

```
mean = (1 / (B×T×F)) × Σ(x)  over all dimensions
std = sqrt((1 / (B×T×F)) × Σ(x - mean)²)
x_norm = (x - mean) / (std + eps)
if affine: y = gamma * x_norm + beta
```

### Denormalization Reverse Pass

```
if affine: x_temp = (y - beta) / gamma
x_denorm = x_temp * (std + eps) + mean
```

## 💾 Serialization

```python
import keras

# Build and compile model
model = keras.Sequential([
    ReversibleInstanceNormMultivariate(num_features=8),
    # ... other layers ...
])
model.compile(optimizer='adam', loss='mse')

# Save model (includes layer configuration)
model.save('model.h5')

# Load model
loaded_model = keras.models.load_model('model.h5')
```

## 🧪 Testing & Validation

```python
import keras
import numpy as np
from kerasfactory.layers import ReversibleInstanceNormMultivariate

# Test exact reconstruction
normalizer = ReversibleInstanceNormMultivariate(num_features=8)
x_original = keras.random.normal((32, 100, 8))

# Normalize
x_norm = normalizer(x_original, mode='norm')

# Denormalize
x_reconstructed = normalizer(x_norm, mode='denorm')

# Check reconstruction error
error = keras.ops.mean(keras.ops.abs(x_original - x_reconstructed))
print(f"Reconstruction error: {error:.2e}")  # Should be < 1e-5

# Verify mean/std after normalization
mean_norm = keras.ops.mean(x_norm)
std_norm = keras.ops.std(x_norm)
print(f"Normalized mean: {mean_norm:.6f}")  # Should be close to 0
print(f"Normalized std: {std_norm:.6f}")   # Should be close to 1
```

## 🎯 Performance Characteristics

| Metric | Value |
|--------|-------|
| **Time Complexity** | O(B×T×F) |
| **Space Complexity** | O(F) for affine params |
| **Memory Per Sample** | O(F) |
| **Training Speed** | Fast |
| **Inference Speed** | Fast |

---

**Last Updated**: 2025-11-04 | **Keras**: 3.0+ | **Status**: ✅ Production Ready
