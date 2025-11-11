---
title: GeospatialScoreRanking - KMR
description: Rank recommendations based on geospatial clustering features for location-aware recommendations
keywords: [geospatial ranking, location-based, scoring, recommendation, keras, geographic, proximity]
---

# Geospatial Score Ranking

<div class="layer-hero">
  <div class="layer-hero-content">
    <h1>Geospatial Score Ranking</h1>
    <div class="layer-badges">
      <span class="badge badge-intermediate">Intermediate</span>
      <span class="badge badge-stable">Stable</span>
      <span class="badge badge-geospatial">Geospatial</span>
    </div>
  </div>
</div>

## Overview

The `GeospatialScoreRanking` layer ranks recommendations based on geospatial scores. It processes spatial clustering features to produce geographic proximity scores for ranking, enabling location-aware recommendation systems.

This layer is crucial for location-based recommendation systems, converting geographic clustering information into ranking scores that prioritize nearby or relevant geographic locations.

## How It Works

The layer processes cluster features:

1. Input Clusters: Spatial clustering features (batch_size, num_items, num_clusters)
2. Feature Processing: Process cluster features through dense layers
3. Score Generation: Generate proximity scores from cluster assignments
4. Output Scores: (batch_size, num_items) ranking scores

## Why Use This Layer?

| Challenge | Traditional Approach | GeospatialScoreRanking Solution |
|-----------|---------------------|--------------------------------|
| Geographic Ranking | Manual distance calculation | Automatic cluster-based ranking |
| Location Awareness | Ignore location | Incorporate location into ranking |
| Proximity Scoring | Fixed distance thresholds | Learnable proximity scoring |
| Scalability | Expensive distance computations | Efficient cluster-based scoring |
| Integration | Separate location logic | Unified ranking with location |

## Use Cases

- Geographic Ranking: Rank items by geographic proximity
- Proximity Scoring: Generate scores based on distance clusters
- Location-Aware Recommendations: Incorporate location into ranking
- Geospatial Filtering: Filter and rank by location
- Regional Recommendations: Rank by geographic regions
- Local Business Recommendations: Prioritize nearby businesses

## Quick Start

### Basic Usage

```python
import keras
from kmr.layers import GeospatialScoreRanking

# Create ranking layer
ranker = GeospatialScoreRanking()

# Rank based on clustering features
clusters = keras.random.normal((32, 100, 5))
scores = ranker(clusters)

print(f"Clusters shape: {clusters.shape}")    # (32, 100, 5)
print(f"Ranking scores: {scores.shape}")     # (32, 100)
```

### In Geospatial Recommendation Pipeline

```python
import keras
from kmr.layers import (
    HaversineGeospatialDistance,
    SpatialFeatureClustering,
    GeospatialScoreRanking,
    TopKRecommendationSelector
)

# Define inputs
user_lat = keras.Input(shape=(1,), dtype='float32')
user_lon = keras.Input(shape=(1,), dtype='float32')
item_lats = keras.Input(shape=(100,), dtype='float32')
item_lons = keras.Input(shape=(100,), dtype='float32')

# Compute distances
distance_layer = HaversineGeospatialDistance()
distances = distance_layer([user_lat, user_lon, item_lats, item_lons])

# Cluster by geography
clustering = SpatialFeatureClustering(num_clusters=5)
cluster_features = clustering(distances)

# Generate ranking scores
ranking = GeospatialScoreRanking()
scores = ranking(cluster_features)

# Select top-K
selector = TopKRecommendationSelector(k=10)
rec_indices, rec_scores = selector(scores)

# Build model
model = keras.Model(
    inputs=[user_lat, user_lon, item_lats, item_lons],
    outputs=[rec_indices, rec_scores]
)
```

## API Reference

::: kmr.layers.GeospatialScoreRanking

## Parameters

This layer processes cluster features automatically.

### Automatic Behavior
- Cluster Processing: Processes spatial cluster features
- Score Generation: Generates proximity-based ranking scores
- Location Awareness: Incorporates geographic information

## Performance Characteristics

- Speed: Fast - O(batch × items × clusters)
- Memory: Linear with cluster count
- Accuracy: Excellent for geographic ranking
- Scalability: Good for large item catalogs
- Location Awareness: Strong geographic signal integration

## Examples

### Example 1: Basic Geographic Ranking

```python
import keras
from kmr.layers import GeospatialScoreRanking

ranker = GeospatialScoreRanking()

# Cluster features from spatial clustering
clusters = keras.random.normal((16, 50, 5))
scores = ranker(clusters)

print(f"Scores shape: {scores.shape}")
print(f"Scores range: [{scores.min():.3f}, {scores.max():.3f}]")
```

### Example 2: Integration with Distance Pipeline

```python
import keras
from kmr.layers import (
    HaversineGeospatialDistance,
    SpatialFeatureClustering,
    GeospatialScoreRanking
)

# User location
user_lat = keras.constant([40.7128])  # NYC
user_lon = keras.constant([-74.0060])

# Item locations
item_lats = keras.random.uniform((1, 100), 35, 45)
item_lons = keras.random.uniform((1, 100), -80, -70)

# Full pipeline
distance_layer = HaversineGeospatialDistance()
distances = distance_layer([user_lat, user_lon, item_lats, item_lons])

clustering = SpatialFeatureClustering(num_clusters=5)
clusters = clustering(distances)

ranking = GeospatialScoreRanking()
scores = ranking(clusters)

print(f"Final ranking scores: {scores.shape}")
```

### Example 3: Regional Ranking Analysis

```python
import keras
from kmr.layers import GeospatialScoreRanking

ranker = GeospatialScoreRanking()

# Different cluster configurations
for num_clusters in [3, 5, 10]:
    clusters = keras.random.normal((32, 100, num_clusters))
    scores = ranker(clusters)
    print(f"Clusters {num_clusters}: scores std = {scores.std():.4f}")
```

## Tips and Best Practices

- Cluster Count: Use appropriate number of clusters for geographic granularity
- Distance Integration: Combine with distance computation for accurate ranking
- Score Normalization: Consider normalizing scores for consistency
- Regional Preferences: Learn user preferences for different regions
- Integration: Combine with other ranking signals for hybrid ranking

## Common Pitfalls

- Cluster Mismatch: Ensure cluster features match expected format
- Geographic Bias: Over-reliance on location can reduce diversity
- Cold Start: New locations may have limited cluster information
- Score Range: Ensure scores are in appropriate range for downstream layers
- Memory: Large cluster counts can increase memory usage

## Related Layers

- SpatialFeatureClustering - Generate cluster features
- HaversineGeospatialDistance - Compute geographic distances
- TopKRecommendationSelector - Select final recommendations
- ThresholdBasedMasking - Filter by distance thresholds

## Further Reading

- Location-Based Services - LBS overview
- Geographic Ranking - Spatial ranking techniques
- Proximity Algorithms - Distance-based ranking
- Geospatial Analysis - Spatial analysis methods
