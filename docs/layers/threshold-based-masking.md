---
title: ThresholdBasedMasking - KMR
description: Apply threshold-based masking to filter values
keywords: [masking, thresholding, filtering, recommendation, keras, geospatial]
---

# Threshold Based Masking

<div class="layer-hero">
  <div class="layer-hero-content">
    <h1>Threshold Based Masking</h1>
    <div class="layer-badges">
      <span class="badge badge-intermediate">Intermediate</span>
      <span class="badge badge-stable">Stable</span>
      <span class="badge badge-utility">Utility</span>
    </div>
  </div>
</div>

## Overview

The `ThresholdBasedMasking` layer applies threshold-based masking to filter values in recommendation systems. It creates binary masks based on threshold comparison, enabling selective filtering of recommendations.

This layer is useful for implementing geospatial filtering or distance-based recommendations where values below/above certain thresholds should be masked.

## How It Works

The layer creates masks based on thresholds:

1. **Input Values**: Scores or distances to filter
2. **Compare to Threshold**: Check if values exceed threshold
3. **Generate Mask**: Create binary mask (0 or 1)
4. **Output Masks**: Binary mask tensor

## Why Use This Layer?

- **Distance Filtering**: Mask items beyond geographic distance thresholds
- **Score Filtering**: Filter recommendations below quality thresholds
- **Feature Masking**: Mask features in geospatial recommendations
- **Quality Control**: Ensure recommendation quality through filtering

## Use Cases

- **Distance Filtering**: Mask items beyond geographic distance thresholds
- **Score Filtering**: Filter recommendations below quality thresholds
- **Feature Masking**: Mask features in geospatial recommendations
- **Quality Control**: Ensure recommendation quality through filtering

## Quick Start

```python
import keras
from kmr.layers import ThresholdBasedMasking

# Create masking layer
masker = ThresholdBasedMasking(threshold=0.5)

# Apply masking
values = keras.random.normal((32, 100))
masks = masker(values)  # Binary masks (0 or 1)

print(f"Input values shape: {values.shape}")
print(f"Masks shape: {masks.shape}")
```

## API Reference

::: kmr.layers.ThresholdBasedMasking

## Parameters

### threshold (float)
- **Purpose**: Threshold value for masking
- **Range**: Any numeric value
- **Impact**: Controls which values are masked

## Performance Characteristics

- **Speed**: Very fast - O(n) comparison operation
- **Memory**: Minimal - output same size as input
- **Accuracy**: Perfect masking
- **Scalability**: Excellent scaling

## Examples

### Example 1: Distance-Based Masking

```python
import keras
from kmr.layers import ThresholdBasedMasking

# Create masking layer for 50km threshold
masker = ThresholdBasedMasking(threshold=50.0)

# Distance matrix (in km)
distances = keras.random.uniform((32, 100), 0, 200)
masks = masker(distances)

print(f"Distances: {distances.shape}")
print(f"Masks: {masks.shape}")
print(f"Masked items: {masks.sum()}")
```

### Example 2: Multiple Thresholds

```python
import keras
from kmr.layers import ThresholdBasedMasking

scores = keras.random.uniform((16, 50), 0, 1)

# Different thresholds
low_threshold = ThresholdBasedMasking(threshold=0.3)
medium_threshold = ThresholdBasedMasking(threshold=0.5)
high_threshold = ThresholdBasedMasking(threshold=0.7)

masks_low = low_threshold(scores)     # More items pass
masks_medium = medium_threshold(scores)   # Medium filtering
masks_high = high_threshold(scores)   # Strict filtering

print(f"Low threshold masked: {masks_low.sum()}")
print(f"Medium threshold masked: {masks_medium.sum()}")
print(f"High threshold masked: {masks_high.sum()}")
```

## Tips and Best Practices

- **Threshold Selection**: Choose thresholds based on domain knowledge
- **Distribution Analysis**: Analyze value distribution before setting threshold
- **Cascading Masking**: Combine multiple masking layers for complex filtering
- **Documentation**: Document why specific thresholds are chosen

## Common Pitfalls

- **Wrong Threshold**: Incorrect threshold filters too much or too little
- **No Items**: Threshold too strict results in no recommendations
- **Performance**: Very low thresholds keep too many items
- **Edge Cases**: Handle edge cases with extreme values

## Related Layers

- [DynamicBatchIndexGenerator](dynamic-batch-index-generator.md)
- [TensorDimensionExpander](tensor-dimension-expander.md)
- [HaversineGeospatialDistance](haversine-geospatial-distance.md)

## Further Reading

- [Masking in Neural Networks](https://en.wikipedia.org/wiki/Mask_(computing))
- [Filtering Techniques](https://en.wikipedia.org/wiki/Filter_(signal_processing))
