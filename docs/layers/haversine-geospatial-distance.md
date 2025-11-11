---
title: HaversineGeospatialDistance - KMR
description: Compute Haversine great-circle distance between geographic coordinates for location-based recommendations
keywords: [geospatial, distance, haversine, location-based, recommendation, keras, geographic, great-circle]
---

# Haversine Geospatial Distance

<div class="layer-hero">
  <div class="layer-hero-content">
    <h1>Haversine Geospatial Distance</h1>
    <div class="layer-badges">
      <span class="badge badge-intermediate">Intermediate</span>
      <span class="badge badge-stable">Stable</span>
      <span class="badge badge-geospatial">Geospatial</span>
    </div>
  </div>
</div>

## Overview

The `HaversineGeospatialDistance` layer computes Haversine distance between geographic coordinates on Earth. It calculates great-circle distances between locations, enabling location-based recommendation filtering and proximity-based ranking.

This layer is essential for geospatial recommendation systems where proximity to location is important, such as local business recommendations, location-based services, or geographic item filtering.

## How It Works

The layer computes Haversine distance:

1. Input Coordinates: User and item latitude/longitude pairs
2. Coordinate Conversion: Convert degrees to radians
3. Haversine Formula: Apply Haversine formula for great-circle distance
4. Distance Calculation: Compute distance in kilometers
5. Output Distances: (batch_size, num_items) distance matrix

The Haversine formula accounts for Earth's spherical shape, providing accurate distance calculations for geographic coordinates.

## Why Use This Layer?

| Challenge | Traditional Approach | HaversineGeospatialDistance Solution |
|-----------|---------------------|-------------------------------------|
| Geographic Distance | Euclidean distance (inaccurate) | Accurate great-circle distance |
| Earth Curvature | Ignore Earth's shape | Account for spherical Earth |
| Location Filtering | Manual distance calculation | Integrated distance computation |
| Scalability | Expensive computations | Efficient batch processing |
| Accuracy | Approximate distances | Precise geographic distances |

## Use Cases

- Geographic Distance Calculation: Compute distances between user and item locations
- Proximity-Based Filtering: Filter items within geographic range
- Location-Based Recommendations: Recommend items near user location
- Geospatial Analysis: Analyze geographic patterns in recommendations
- Local Business Recommendations: Prioritize nearby businesses
- Regional Recommendations: Rank by geographic proximity

## Quick Start

### Basic Usage

```python
import keras
from kmr.layers import HaversineGeospatialDistance

# Create distance layer
distance_layer = HaversineGeospatialDistance(earth_radius_km=6371)

# Compute distances (latitude/longitude in degrees)
user_lat = keras.constant([40.7128])  # NYC latitude
user_lon = keras.constant([-74.0060])  # NYC longitude
item_lats = keras.random.uniform((1, 10), 35, 45)
item_lons = keras.random.uniform((1, 10), -80, -70)

distances = distance_layer([user_lat, user_lon, item_lats, item_lons])
print(f"Distances (km): {distances.shape}")  # (1, 10)
```

### In Location-Based Recommendation Pipeline

```python
import keras
from kmr.layers import (
    HaversineGeospatialDistance,
    SpatialFeatureClustering,
    GeospatialScoreRanking,
    ThresholdBasedMasking,
    TopKRecommendationSelector
)

# Define inputs
user_lat = keras.Input(shape=(1,), dtype='float32', name='user_lat')
user_lon = keras.Input(shape=(1,), dtype='float32', name='user_lon')
item_lats = keras.Input(shape=(100,), dtype='float32', name='item_lats')
item_lons = keras.Input(shape=(100,), dtype='float32', name='item_lons')

# Compute distances
distance_layer = HaversineGeospatialDistance(earth_radius_km=6371)
distances = distance_layer([user_lat, user_lon, item_lats, item_lons])

# Cluster by geography
clustering = SpatialFeatureClustering(num_clusters=5)
clusters = clustering(distances)

# Generate scores
ranking = GeospatialScoreRanking()
scores = ranking(clusters)

# Filter by distance threshold (50km max)
masker = ThresholdBasedMasking(threshold=50.0)
scores_filtered = masker(scores)

# Select top-K
selector = TopKRecommendationSelector(k=10)
rec_indices, rec_scores = selector(scores_filtered)

# Build model
model = keras.Model(
    inputs=[user_lat, user_lon, item_lats, item_lons],
    outputs=[rec_indices, rec_scores]
)
```

## API Reference

::: kmr.layers.HaversineGeospatialDistance

## Parameters Deep Dive

### earth_radius_km (float)
- Purpose: Earth's radius in kilometers for distance calculation
- Default: 6371.0 (mean Earth radius)
- Range: 6356.752 to 6378.137 (varies by latitude)
- Impact: Affects distance calculation accuracy
- Recommendation: Use default 6371.0 for most cases

## Performance Characteristics

- Speed: Fast - O(batch × items) trigonometric operations
- Memory: Minimal - no additional buffers
- Accuracy: Excellent - mathematically precise for spherical Earth
- Scalability: Good for large numbers of items
- Precision: Handles Earth's curvature accurately

## Examples

### Example 1: Basic Distance Calculation

```python
import keras
from kmr.layers import HaversineGeospatialDistance

distance_layer = HaversineGeospatialDistance()

# NYC to various cities
user_lat = keras.constant([40.7128])  # NYC
user_lon = keras.constant([-74.0060])

# Other cities (lat, lon)
item_lats = keras.constant([[34.0522, 41.8781, 39.9526]])  # LA, Chicago, Philly
item_lons = keras.constant([[-118.2437, -87.6298, -75.1652]])

distances = distance_layer([user_lat, user_lon, item_lats, item_lons])
print(f"Distances from NYC (km): {distances.numpy()}")
```

### Example 2: Batch Processing

```python
import keras
from kmr.layers import HaversineGeospatialDistance

distance_layer = HaversineGeospatialDistance()

# Multiple users
user_lats = keras.constant([40.7128, 34.0522, 41.8781])  # NYC, LA, Chicago
user_lons = keras.constant([-74.0060, -118.2437, -87.6298])

# Items for each user
item_lats = keras.random.uniform((3, 50), 30, 50)
item_lons = keras.random.uniform((3, 50), -120, -70)

distances = distance_layer([user_lats, user_lons, item_lats, item_lons])
print(f"Distance matrix: {distances.shape}")  # (3, 50)
```

### Example 3: Distance-Based Filtering

```python
import keras
from kmr.layers import HaversineGeospatialDistance, ThresholdBasedMasking

distance_layer = HaversineGeospatialDistance()
masker = ThresholdBasedMasking(threshold=100.0)  # 100km threshold

# User and items
user_lat = keras.constant([40.7128])
user_lon = keras.constant([-74.0060])
item_lats = keras.random.uniform((1, 100), 35, 45)
item_lons = keras.random.uniform((1, 100), -80, -70)

# Compute distances
distances = distance_layer([user_lat, user_lon, item_lats, item_lons])

# Filter items within 100km
masks = masker(distances)
print(f"Items within 100km: {masks.sum()}")
```

## Tips and Best Practices

- Coordinate Format: Use decimal degrees (not DMS)
- Latitude Range: -90 to 90 degrees
- Longitude Range: -180 to 180 degrees
- Earth Radius: Use default 6371.0 for general use
- Distance Units: Output is in kilometers
- Batch Processing: Efficiently handles multiple users/items

## Common Pitfalls

- Coordinate Format: Ensure coordinates are in decimal degrees
- Latitude/Longitude Order: Correct order is (lat, lon)
- Out of Range: Coordinates outside valid ranges cause errors
- Units: Output is kilometers, not miles
- Earth Radius: Different radius values affect accuracy
- Batch Dimensions: Ensure proper batch dimension matching

## Related Layers

- SpatialFeatureClustering - Cluster by geographic distance
- GeospatialScoreRanking - Rank based on distances
- ThresholdBasedMasking - Filter by distance thresholds
- TopKRecommendationSelector - Select nearby recommendations

## Further Reading

- Haversine Formula - Mathematical foundation
- Great-Circle Distance - Geographic distance calculation
- Geographic Coordinates - Coordinate system overview
- Location-Based Services - LBS applications
