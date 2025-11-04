---
title: TensorDimensionExpander - KMR
description: Expand tensor dimensions for broadcasting and reshaping operations
keywords: [tensor manipulation, broadcasting, reshaping, dimension expansion, keras, utility]
---

# Tensor Dimension Expander

<div class="layer-hero">
  <div class="layer-hero-content">
    <h1>Tensor Dimension Expander</h1>
    <div class="layer-badges">
      <span class="badge badge-beginner">Beginner</span>
      <span class="badge badge-stable">Stable</span>
      <span class="badge badge-utility">Utility</span>
    </div>
  </div>
</div>

## Overview

The `TensorDimensionExpander` expands tensor dimensions for broadcasting and reshaping operations. It adds new axes at specified positions, enabling proper broadcasting in complex recommendation computations.

This layer is crucial for manipulating tensor shapes when combining data from different sources or preparing tensors for matrix operations in recommendation systems.

## How It Works

The layer expands tensor dimensions:

1. **Input Tensor**: Original tensor with shape (batch_size, ...)
2. **Axis Selection**: Choose position to insert new axis
3. **Dimension Expansion**: Add new axis at specified position
4. **Output**: Expanded tensor with additional dimension

## Why Use This Layer?

- **Broadcasting**: Prepare tensors for element-wise operations
- **Matrix Operations**: Reshape for compatibility with matrix multiplications
- **Feature Combination**: Combine features from different dimensions
- **Batch Processing**: Align tensor dimensions for batch operations

## Use Cases

- **Broadcasting**: Prepare tensors for broadcasting in similarity computation
- **Matrix Operations**: Reshape for compatibility with matrix multiplications
- **Feature Combination**: Combine features from different dimensions
- **Batch Processing**: Align tensor dimensions for batch operations

## Quick Start

```python
import keras
from kmr.layers import TensorDimensionExpander

# Create expander
expander = TensorDimensionExpander(axis=1)

# Expand tensor
input_tensor = keras.random.normal((32, 100))
expanded = expander(input_tensor)  # (32, 1, 100)

print(f"Input: {input_tensor.shape}")
print(f"Output: {expanded.shape}")
```

## API Reference

::: kmr.layers.TensorDimensionExpander

## Parameters

### axis (int)
- **Purpose**: Position to insert new dimension
- **Range**: 0 to len(shape)
- **Impact**: Where to add the new axis

## Performance Characteristics

- **Speed**: Very fast - O(1) reshape operation
- **Memory**: Minimal - no additional data
- **Accuracy**: Perfect - no information loss
- **Scalability**: Perfect scaling

## Examples

### Example 1: Broadcasting for Similarity

```python
import keras
from kmr.layers import TensorDimensionExpander, NormalizedDotProductSimilarity

# User representation
user_repr = keras.random.normal((32, 64))

# Item representations
item_repr = keras.random.normal((32, 100, 64))

# Expand user repr for broadcasting
expander = TensorDimensionExpander(axis=1)
user_expanded = expander(user_repr)  # (32, 1, 64)

# Compute similarity
similarity = NormalizedDotProductSimilarity()
scores = similarity([user_expanded, item_repr])

print(f"User expanded: {user_expanded.shape}")  # (32, 1, 64)
print(f"Scores: {scores.shape}")  # (32, 100)
```

### Example 2: Different Axis Positions

```python
import keras
from kmr.layers import TensorDimensionExpander

input_data = keras.random.normal((32, 100))

# Expand at different positions
exp_0 = TensorDimensionExpander(axis=0)
exp_1 = TensorDimensionExpander(axis=1)
exp_2 = TensorDimensionExpander(axis=2)

out_0 = exp_0(input_data)  # (1, 32, 100)
out_1 = exp_1(input_data)  # (32, 1, 100)
out_2 = exp_2(input_data)  # (32, 100, 1)

print(f"Axis 0: {out_0.shape}")
print(f"Axis 1: {out_1.shape}")
print(f"Axis 2: {out_2.shape}")
```

## Tips and Best Practices

- **Axis Selection**: Choose axis carefully for proper broadcasting
- **Documentation**: Comment why expansion is needed
- **Shape Verification**: Always verify output shapes match expectations
- **Efficiency**: Use before expensive operations

## Common Pitfalls

- **Wrong Axis**: Incorrect axis selection causes shape mismatches
- **Multiple Expansions**: Can lead to unexpected shapes
- **Broadcasting Errors**: Mismatched shapes after expansion

## Related Layers

- [DynamicBatchIndexGenerator](dynamic-batch-index-generator.md)
- [ThresholdBasedMasking](threshold-based-masking.md)
- [NormalizedDotProductSimilarity](normalized-dot-product-similarity.md)

## Further Reading

- [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [Tensor Operations](https://keras.io/api/ops/)
