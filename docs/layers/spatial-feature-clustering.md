---
title: SpatialFeatureClustering - KMR
description: Cluster spatial features into geographic regions
keywords: [clustering, spatial, geospatial, geographic regions, recommendation, keras]
---

# Spatial Feature Clustering

<div class="layer-hero">
  <div class="layer-hero-content">
    <h1>Spatial Feature Clustering</h1>
    <div class="layer-badges">
      <span class="badge badge-intermediate">Intermediate</span>
      <span class="badge badge-stable">Stable</span>
      <span class="badge badge-geospatial">Geospatial</span>
    </div>
  </div>
</div>

## Overview

The `SpatialFeatureClustering` layer clusters spatial features into geographic regions. It groups geospatial distances into clusters for location-based recommendation filtering.

This layer enables efficient geospatial recommendation by identifying which geographic cluster each item belongs to, supporting location-aware ranking and proximity-based filtering.

## How It Works

The layer processes distance data into spatial clusters:

1. **Input Distances**: Geographic distances between user and items
2. **Cluster Assignment**: Assign each distance to nearest cluster
3. **Cluster Features**: Generate cluster membership features
4. **Cluster Probabilities**: Compute soft assignments
5. **Output Clusters**: Cluster feature matrix

## Why Use This Layer?

- **Geographic Clustering**: Cluster items by geographic proximity
- **Region-Based Filtering**: Filter recommendations by geographic region
- **Location-Based Ranking**: Rank recommendations by geographic cluster
- **Spatial Grouping**: Group locations into geographic zones

## Use Cases

- **Geographic Clustering**: Cluster items by geographic proximity
- **Region-Based Filtering**: Filter recommendations by geographic region
- **Location-Based Ranking**: Rank recommendations by geographic cluster
- **Spatial Grouping**: Group locations into geographic zones
- **Multi-Tier Recommendation**: Use clusters for first-tier filtering

## Quick Start

```python
import keras
from kmr.layers import SpatialFeatureClustering

# Create clustering layer
clustering = SpatialFeatureClustering(num_clusters=5)

# Cluster distances
distances = keras.random.uniform((32, 100), 0, 100)
clusters = clustering(distances)

print(f"Input distances: {distances.shape}")
print(f"Cluster assignments: {clusters.shape}")
```

## API Reference

::: kmr.layers.SpatialFeatureClustering

## Parameters

### num_clusters (int)
- **Purpose**: Number of geographic clusters
- **Range**: 2 to 100 typically
- **Impact**: Granularity of geographic regions
- **Recommendation**: Start with 5-10 clusters

## Performance Characteristics

- **Speed**: Fast - O(batch x items x clusters)
- **Memory**: Linear with number of clusters
- **Accuracy**: Excellent for geographic grouping
- **Scalability**: Scales well to large catalogs

## Examples

### Example 1: Geographic Zone Clustering

```python
import keras
from kmr.layers import SpatialFeatureClustering

# Create clustering for 5 zones
clustering = SpatialFeatureClustering(num_clusters=5)

# Distance matrix (km)
distances = keras.random.uniform((16, 50), 0, 200)
clusters = clustering(distances)

print(f"Cluster probabilities shape: {clusters.shape}")
print(f"Probability range: [{clusters.min():.3f}, {clusters.max():.3f}]")
```

### Example 2: Different Granularity

```python
import keras
from kmr.layers import SpatialFeatureClustering

# Different clustering levels
coarse = SpatialFeatureClustering(num_clusters=3)
medium = SpatialFeatureClustering(num_clusters=10)
fine = SpatialFeatureClustering(num_clusters=50)

distances = keras.random.uniform((32, 100), 0, 500)

coarse_out = coarse(distances)   # (32, 100, 3)
medium_out = medium(distances)   # (32, 100, 10)
fine_out = fine(distances)       # (32, 100, 50)

print(f"Coarse: {coarse_out.shape}")
print(f"Medium: {medium_out.shape}")
print(f"Fine: {fine_out.shape}")
```

## Tips and Best Practices

- **Cluster Count**: Start with 5-10 clusters
- **Distance Normalization**: Normalize distances before clustering
- **Multi-Level**: Use multiple layers for hierarchical clustering
- **Integration**: Combine with distance and ranking layers

## Common Pitfalls

- **Too Few Clusters**: Loss of geographic information
- **Too Many Clusters**: Computational overhead
- **Wrong Distance Range**: Ensure distances are normalized
- **Memory Issues**: Large cluster counts increase memory

## Related Layers

- [HaversineGeospatialDistance](haversine-geospatial-distance.md)
- [GeospatialScoreRanking](geospatial-score-ranking.md)
- [ThresholdBasedMasking](threshold-based-masking.md)
- [TopKRecommendationSelector](top-k-recommendation-selector.md)

## Further Reading

- [K-Means Clustering](https://en.wikipedia.org/wiki/K-means_clustering)
- [Geospatial Analysis](https://en.wikipedia.org/wiki/Spatial_analysis)
- [Location-Based Services](https://arxiv.org/abs/1807.07274)
