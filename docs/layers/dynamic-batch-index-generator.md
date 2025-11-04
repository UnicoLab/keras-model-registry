---
title: DynamicBatchIndexGenerator - KMR
description: Generate dynamic batch indices for grouping and indexing operations in recommendation systems
keywords: [indexing, batching, grouping, dynamic indices, recommendation, keras, utility, batch processing]
---

# Dynamic Batch Index Generator

<div class="layer-hero">
  <div class="layer-hero-content">
    <h1>Dynamic Batch Index Generator</h1>
    <div class="layer-badges">
      <span class="badge badge-advanced">Advanced</span>
      <span class="badge badge-stable">Stable</span>
      <span class="badge badge-utility">Utility</span>
    </div>
  </div>
</div>

## Overview

The `DynamicBatchIndexGenerator` dynamically generates batch indices based on input shape, enabling flexible indexing and grouping operations for complex recommendation tasks. It creates index tensors that adapt to variable batch sizes automatically.

This layer is essential for advanced indexing operations in clustering, grouping, and multi-level recommendation systems where dynamic batch processing is required. It eliminates the need for manual batch size management and enables flexible batch-wise operations.

## How It Works

The layer generates indices adaptively:

1. Input Tensor: Any tensor with batch dimension
2. Extract Batch Size: Get dynamic batch size from input shape at runtime
3. Generate Range: Create index array [0, 1, 2, ..., batch_size-1]
4. Optional Expansion: Expand to required shape for broadcasting
5. Output Indices: Dynamic index tensor matching batch dimension

The layer automatically handles different batch sizes without requiring manual configuration.

## Why Use This Layer?

| Challenge | Traditional Approach | DynamicBatchIndexGenerator Solution |
|-----------|---------------------|-------------------------------------|
| Batch Size Management | Manual batch size tracking | Automatic batch size detection |
| Dynamic Batching | Fixed batch sizes | Adapts to any batch size |
| Index Generation | Manual index creation | Automatic index generation |
| Flexibility | Hard-coded batch dimensions | Runtime batch size adaptation |
| Code Simplicity | Complex batch management | Simple single-layer solution |

## Use Cases

- Batch-wise Operations: Dynamic indexing per batch element
- Grouping: Dynamic group assignment based on batch size
- Advanced Indexing: Complex multi-dimensional indexing operations
- Geospatial Clustering: Batch-wise clustering with dynamic indices
- Recommendation Batching: Handle variable user/item batch sizes
- Parallel Processing: Index generation for parallel batch processing

## Quick Start

### Basic Usage

```python
import keras
from kmr.layers import DynamicBatchIndexGenerator

# Create index generator
generator = DynamicBatchIndexGenerator()

# Generate indices for different batch sizes
for batch_size in [8, 16, 32]:
    input_data = keras.random.normal((batch_size, 100))
    indices = generator(input_data)
    print(f"Batch {batch_size}: indices shape = {indices.shape}")
    print(f"  Indices: {indices[:5].numpy()}")  # First 5 indices
```

### In Batch-wise Clustering

```python
import keras
from kmr.layers import DynamicBatchIndexGenerator, SpatialFeatureClustering

# Define inputs
features = keras.Input(shape=(100, 10), name='features')

# Generate batch indices
index_generator = DynamicBatchIndexGenerator()
batch_indices = index_generator(features)

# Use indices for batch-wise clustering
clustering = SpatialFeatureClustering(num_clusters=5)
clusters = clustering(features)

# Combine indices with clusters for batch tracking
print(f"Batch indices: {batch_indices.shape}")
print(f"Clusters: {clusters.shape}")

# Build model
model = keras.Model(inputs=features, outputs=[batch_indices, clusters])
```

## API Reference

::: kmr.layers.DynamicBatchIndexGenerator

## Parameters

This layer has no configurable parameters - it automatically adapts to input batch size.

### Automatic Behavior
- Batch Size Detection: Extracts batch size from input tensor shape
- Index Generation: Creates sequential indices [0, 1, 2, ..., batch_size-1]
- Shape Adaptation: Output shape matches batch dimension

## Performance Characteristics

- Speed: Very fast - O(batch_size) index generation
- Memory: Minimal - only stores index tensor
- Accuracy: Perfect - exact sequential indices
- Scalability: Excellent for any batch size
- Flexibility: Adapts to variable batch sizes automatically

## Examples

### Example 1: Basic Index Generation

```python
import keras
from kmr.layers import DynamicBatchIndexGenerator

# Create generator
generator = DynamicBatchIndexGenerator()

# Test with different batch sizes
for batch_size in [1, 8, 16, 32, 64]:
    input_data = keras.random.normal((batch_size, 100))
    indices = generator(input_data)
    
    print(f"Batch size {batch_size}:")
    print(f"  Input shape: {input_data.shape}")
    print(f"  Indices shape: {indices.shape}")
    print(f"  Indices range: [{indices.min()}, {indices.max()}]")
```

### Example 2: Batch-wise Grouping

```python
import keras
from kmr.layers import DynamicBatchIndexGenerator

generator = DynamicBatchIndexGenerator()

# Create batch data
batch_size = 16
features = keras.random.normal((batch_size, 50, 10))

# Generate batch indices
batch_indices = generator(features)

# Use for grouping operations
print(f"Features: {features.shape}")
print(f"Batch indices: {batch_indices.shape}")
print(f"Unique batch indices: {keras.ops.unique(batch_indices)}")
```

### Example 3: Integration with Clustering

```python
import keras
from kmr.layers import (
    DynamicBatchIndexGenerator,
    SpatialFeatureClustering,
    GeospatialScoreRanking
)

# Input data
distances = keras.Input(shape=(100,), dtype='float32')

# Generate batch indices for tracking
index_gen = DynamicBatchIndexGenerator()
batch_indices = index_gen(distances)

# Process with clustering
clustering = SpatialFeatureClustering(num_clusters=5)
clusters = clustering(keras.ops.expand_dims(distances, axis=0))

# Ranking
ranking = GeospatialScoreRanking()
scores = ranking(clusters)

# Model with batch tracking
model = keras.Model(
    inputs=distances,
    outputs=[batch_indices, clusters, scores]
)

# Usage
test_distances = keras.random.uniform((8, 100), 0, 200)
indices, clusters, scores = model(test_distances)

print(f"Batch indices: {indices.shape}")
print(f"Clusters: {clusters.shape}")
print(f"Scores: {scores.shape}")
```

## Tips and Best Practices

- Automatic Adaptation: Layer automatically handles different batch sizes
- No Configuration: No parameters needed - works out of the box
- Integration: Use with other layers for batch tracking
- Debugging: Useful for tracking batch elements during processing
- Performance: Very lightweight - minimal overhead
- Flexibility: Works with any tensor shape as long as it has batch dimension

## Common Pitfalls

- No Batch Dimension: Fails if input lacks batch dimension
- Static Batch Size: If you need static batch size, use fixed indices
- Shape Mismatch: Ensure output shape matches your use case
- Memory: Very large batch sizes may create large index tensors
- Broadcasting: May need to expand indices for broadcasting operations

## Related Layers

- TensorDimensionExpander - Expand dimensions for broadcasting
- ThresholdBasedMasking - Apply masking with batch indices
- SpatialFeatureClustering - Use indices for batch-wise clustering
- TopKRecommendationSelector - Track batch elements in selection

## Further Reading

- Tensor Indexing - Tensor manipulation techniques
- Batch Processing - Batch operation patterns
- Dynamic Computation - Runtime batch size handling
- Keras Layers - Custom layer implementation
