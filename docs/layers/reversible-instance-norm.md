---
title: ReversibleInstanceNorm - KerasFactory
description: Reversible instance normalization with optional affine transformation for time series
keywords: [normalization, instance norm, reversible, time series, stability, keras]
---

# 🔄 ReversibleInstanceNorm

<div class="layer-hero">
  <div class="layer-hero-content">
    <h1>🔄 ReversibleInstanceNorm</h1>
    <div class="layer-badges">
      <span class="badge badge-intermediate">🟡 Intermediate</span>
      <span class="badge badge-stable">✅ Stable</span>
      <span class="badge badge-timeseries">⏱️ Time Series</span>
    </div>
  </div>
</div>

## 🎯 Overview

The `ReversibleInstanceNorm` layer applies reversible instance normalization to time series data, enabling normalization for training and exact denormalization for inference. This is crucial for time series models where you need to restore predictions to the original data scale.

Key features:
- **Reversible**: Exact denormalization preserves interpretability
- **Optional Affine**: Learnable scale and shift parameters
- **Multiple Modes**: Normalize/denormalize in same layer
- **Training Stability**: Improves convergence and generalization

## 🔍 How It Works

The layer operates in two modes:

### Normalization (Training)
1. Compute statistics (mean, std) per instance
2. Subtract mean and divide by std
3. Optionally apply learnable affine transform
4. Store statistics for denormalization

### Denormalization (Inference)
1. Reverse affine transform (if used)
2. Multiply by stored std
3. Add stored mean
4. Restore to original scale

## 💡 Why Use This Layer?

| Challenge | Without RevIN | With RevIN |
|-----------|---|---|
| **Scale Sensitivity** | Model learns different scales poorly | ✨ Normalized training |
| **Interpretability** | Predictions in model scale | 🎯 Original data scale |
| **Stability** | Training instability | ⚡ Stable convergence |
| **Transfer Learning** | Limited generalization | 🔄 Better transfer capability |

## 📊 Use Cases

- **Time Series Forecasting**: Normalize input and denormalize output
- **Multivariate Scaling**: Handle different feature scales
- **Domain Adaptation**: Transfer models across datasets
- **Anomaly Detection**: Normalize for training, denormalize for detection
- **Data Augmentation**: Consistent scaling across augmented samples

## 🚀 Quick Start

### Basic Normalization

```python
import keras
from kerasfactory.layers import ReversibleInstanceNorm

# Create normalization layer
norm_layer = ReversibleInstanceNorm(num_features=8, eps=1e-5)

# Input data
x = keras.random.normal((32, 100, 8))

# Normalize for training
x_norm = norm_layer(x, mode='norm')

# Use normalized data in model
# ... model training ...

# Denormalize predictions
y_denorm = norm_layer(y_pred, mode='denorm')
```

### In a Forecasting Pipeline

```python
from kerasfactory.layers import ReversibleInstanceNorm, TokenEmbedding

# Setup pipeline
normalizer = ReversibleInstanceNorm(num_features=7, affine=True)
token_emb = TokenEmbedding(c_in=7, d_model=64)

# Training
x_raw = keras.random.normal((32, 96, 7))
x_norm = normalizer(x_raw, mode='norm')
x_emb = token_emb(x_norm)
# ... model forward pass ...

# Inference
y_pred_norm = model(x_norm)
y_pred = normalizer(y_pred_norm, mode='denorm')
```

## 🔧 API Reference

```python
kerasfactory.layers.ReversibleInstanceNorm(
    num_features: int,
    eps: float = 1e-5,
    affine: bool = False,
    subtract_last: bool = False,
    non_norm: bool = False,
    name: str | None = None,
    **kwargs
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_features` | `int` | — | Number of features |
| `eps` | `float` | 1e-5 | Numerical stability |
| `affine` | `bool` | False | Learnable scale/shift |
| `subtract_last` | `bool` | False | Normalize by last value |
| `non_norm` | `bool` | False | Disable normalization |
| `name` | `str \| None` | None | Layer name |

## 💡 Best Practices

1. **Use Before Embedding**: Normalize raw data before embeddings
2. **Affine Transform**: Enable for flexible scaling in complex models
3. **Denormalize Output**: Always denormalize final predictions
4. **Feature Scaling**: Ensures all features contribute equally
5. **Statistical Stability**: eps prevents division by zero

## ⚠️ Common Pitfalls

- ❌ **Forgetting denormalization**: Loss of interpretability
- ❌ **Wrong mode**: Using 'norm' when expecting 'denorm'
- ❌ **Batch dependency**: Ensure consistent batch processing
- ❌ **Shared statistics**: Don't mix statistics across batches

---

**Last Updated**: 2025-11-04 | **Keras**: 3.0+ | **Status**: ✅ Production Ready
