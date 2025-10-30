---
title: DateEncodingLayer - KMR
description: Layer for encoding date components into cyclical features using sine and cosine transformations
keywords: [date encoding, cyclical features, sine cosine, time series, keras, neural networks, temporal features]
---

# 🔄 DateEncodingLayer

<div class="layer-hero">
  <div class="layer-hero-content">
    <h1>🔄 DateEncodingLayer</h1>
    <div class="layer-badges">
      <span class="badge badge-intermediate">🟡 Intermediate</span>
      <span class="badge badge-stable">✅ Stable</span>
      <span class="badge badge-popular">🔥 Popular</span>
    </div>
  </div>
</div>

## 🎯 Overview

The `DateEncodingLayer` takes date components (year, month, day, day of week) and encodes them into cyclical features using sine and cosine transformations. This approach preserves the cyclical nature of temporal data, which is crucial for neural networks to understand patterns like seasonality and periodicity.

This layer is particularly powerful for time series analysis where the cyclical nature of dates is important, such as seasonal patterns, weekly cycles, and daily rhythms.

## 🔍 How It Works

The DateEncodingLayer processes date components through cyclical encoding:

1. **Component Extraction**: Extracts year, month, day, and day of week
2. **Year Normalization**: Normalizes year to [0, 1] range based on min/max years
3. **Cyclical Encoding**: Applies sine and cosine transformations to each component
4. **Feature Combination**: Combines all cyclical encodings into a single tensor
5. **Output Generation**: Produces 8-dimensional cyclical feature vector

```mermaid
graph TD
    A[Date Components: year, month, day, day_of_week] --> B[Year Normalization]
    B --> C[Cyclical Encoding]
    
    C --> D[Year: sin(2π * year_norm), cos(2π * year_norm)]
    C --> E[Month: sin(2π * month/12), cos(2π * month/12)]
    C --> F[Day: sin(2π * day/31), cos(2π * day/31)]
    C --> G[Day of Week: sin(2π * dow/7), cos(2π * dow/7)]
    
    D --> H[Combine All Encodings]
    E --> H
    F --> H
    G --> H
    
    H --> I[Cyclical Features: 8 dimensions]
    
    style A fill:#e6f3ff,stroke:#4a86e8
    style I fill:#e8f5e9,stroke:#66bb6a
    style B fill:#fff9e6,stroke:#ffb74d
    style C fill:#f3e5f5,stroke:#9c27b0
    style H fill:#e1f5fe,stroke:#03a9f4
```

## 💡 Why Use This Layer?

| Challenge | Traditional Approach | DateEncodingLayer's Solution |
|-----------|---------------------|----------------------------|
| **Cyclical Nature** | Treats dates as linear values | 🎯 **Preserves cyclicality** with sine/cosine encoding |
| **Seasonal Patterns** | Misses seasonal relationships | ⚡ **Captures seasonality** through cyclical encoding |
| **Neural Network Understanding** | Linear encoding confuses networks | 🧠 **Neural-friendly** cyclical representation |
| **Temporal Relationships** | Loses temporal proximity | 🔗 **Maintains temporal** relationships through encoding |

## 📊 Use Cases

- **Time Series Analysis**: Encoding temporal features for neural networks
- **Seasonal Pattern Recognition**: Capturing seasonal and cyclical patterns
- **Event Prediction**: Predicting events based on temporal patterns
- **Financial Analysis**: Analyzing financial data with temporal components
- **Weather Forecasting**: Processing weather data with seasonal patterns

## 🚀 Quick Start

### Basic Usage

```python
import keras
from kmr.layers import DateEncodingLayer

# Create sample date components [year, month, day, day_of_week]
date_components = keras.ops.convert_to_tensor([
    [2023, 1, 15, 6],   # Sunday, January 15, 2023
    [2023, 6, 21, 2],   # Wednesday, June 21, 2023
    [2023, 12, 25, 0]   # Sunday, December 25, 2023
], dtype="float32")

# Apply cyclical encoding
encoder = DateEncodingLayer(min_year=1900, max_year=2100)
encoded = encoder(date_components)

print(f"Input shape: {date_components.shape}")    # (3, 4)
print(f"Output shape: {encoded.shape}")          # (3, 8)
print(f"Encoded features: {encoded}")
# Output: [year_sin, year_cos, month_sin, month_cos, day_sin, day_cos, dow_sin, dow_cos]
```

### In a Sequential Model

```python
import keras
from kmr.layers import DateEncodingLayer

model = keras.Sequential([
    DateEncodingLayer(min_year=1900, max_year=2100),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### In a Functional Model

```python
import keras
from kmr.layers import DateEncodingLayer

# Define inputs
inputs = keras.Input(shape=(4,))  # [year, month, day, day_of_week]

# Apply cyclical encoding
x = DateEncodingLayer(min_year=1900, max_year=2100)(inputs)

# Continue processing
x = keras.layers.Dense(32, activation='relu')(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(16, activation='relu')(x)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, outputs)
```

### Advanced Configuration

```python
# Advanced configuration with custom year range
def create_temporal_analysis_model():
    # Input for date components
    date_input = keras.Input(shape=(4,))  # [year, month, day, day_of_week]
    
    # Apply cyclical encoding
    cyclical_features = DateEncodingLayer(
        min_year=2000,  # Custom year range
        max_year=2030
    )(date_input)
    
    # Process cyclical features
    x = keras.layers.Dense(64, activation='relu')(cyclical_features)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    
    # Multi-task output
    season = keras.layers.Dense(4, activation='softmax', name='season')(x)
    weekday = keras.layers.Dense(7, activation='softmax', name='weekday')(x)
    is_weekend = keras.layers.Dense(1, activation='sigmoid', name='is_weekend')(x)
    
    return keras.Model(date_input, [season, weekday, is_weekend])

model = create_temporal_analysis_model()
model.compile(
    optimizer='adam',
    loss={'season': 'categorical_crossentropy', 'weekday': 'categorical_crossentropy', 'is_weekend': 'binary_crossentropy'},
    loss_weights={'season': 1.0, 'weekday': 0.5, 'is_weekend': 0.3}
)
```

## 📖 API Reference

::: kmr.layers.DateEncodingLayer

## 🔧 Parameters Deep Dive

### `min_year` (int)
- **Purpose**: Minimum year for normalization
- **Default**: 1900
- **Impact**: Affects year normalization range
- **Recommendation**: Set based on your data's year range

### `max_year` (int)
- **Purpose**: Maximum year for normalization
- **Default**: 2100
- **Impact**: Affects year normalization range
- **Recommendation**: Set based on your data's year range

## 📈 Performance Characteristics

- **Speed**: ⚡⚡⚡⚡ Very fast - simple mathematical operations
- **Memory**: 💾 Low memory usage - no additional parameters
- **Accuracy**: 🎯🎯🎯🎯 Excellent for cyclical temporal features
- **Best For**: Time series data requiring cyclical encoding

## 🎨 Examples

### Example 1: Seasonal Pattern Analysis

```python
import keras
import numpy as np
from kmr.layers import DateEncodingLayer

# Create seasonal analysis model
def create_seasonal_model():
    # Input for date components
    date_input = keras.Input(shape=(4,))  # [year, month, day, day_of_week]
    
    # Apply cyclical encoding
    cyclical_features = DateEncodingLayer(min_year=2000, max_year=2030)(date_input)
    
    # Process cyclical features
    x = keras.layers.Dense(64, activation='relu')(cyclical_features)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    
    # Predictions
    temperature = keras.layers.Dense(1, name='temperature')(x)
    humidity = keras.layers.Dense(1, name='humidity')(x)
    season = keras.layers.Dense(4, activation='softmax', name='season')(x)
    
    return keras.Model(date_input, [temperature, humidity, season])

model = create_seasonal_model()
model.compile(
    optimizer='adam',
    loss={'temperature': 'mse', 'humidity': 'mse', 'season': 'categorical_crossentropy'},
    loss_weights={'temperature': 1.0, 'humidity': 0.5, 'season': 0.3}
)

# Test with sample data
sample_dates = keras.ops.convert_to_tensor([
    [2023, 1, 15, 6],   # Winter Sunday
    [2023, 4, 15, 5],   # Spring Saturday
    [2023, 7, 15, 5],   # Summer Saturday
    [2023, 10, 15, 6]   # Fall Sunday
], dtype="float32")

predictions = model(sample_dates)
print(f"Predictions shape: {[p.shape for p in predictions]}")
```

### Example 2: Business Cycle Analysis

```python
# Analyze business cycles with cyclical encoding
def create_business_cycle_model():
    # Input for date components
    date_input = keras.Input(shape=(4,))  # [year, month, day, day_of_week]
    
    # Apply cyclical encoding
    cyclical_features = DateEncodingLayer(min_year=2020, max_year=2030)(date_input)
    
    # Process cyclical features
    x = keras.layers.Dense(128, activation='relu')(cyclical_features)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    
    # Business predictions
    sales_volume = keras.layers.Dense(1, name='sales_volume')(x)
    customer_traffic = keras.layers.Dense(1, name='customer_traffic')(x)
    is_peak_season = keras.layers.Dense(1, activation='sigmoid', name='is_peak_season')(x)
    
    return keras.Model(date_input, [sales_volume, customer_traffic, is_peak_season])

model = create_business_cycle_model()
model.compile(
    optimizer='adam',
    loss={'sales_volume': 'mse', 'customer_traffic': 'mse', 'is_peak_season': 'binary_crossentropy'},
    loss_weights={'sales_volume': 1.0, 'customer_traffic': 0.5, 'is_peak_season': 0.3}
)
```

### Example 3: Cyclical Feature Analysis

```python
# Analyze the cyclical features produced by the encoding
def analyze_cyclical_features():
    # Create sample date components
    dates = keras.ops.convert_to_tensor([
        [2023, 1, 1, 0],    # New Year's Day (Sunday)
        [2023, 3, 20, 0],   # Spring Equinox (Sunday)
        [2023, 6, 21, 2],   # Summer Solstice (Wednesday)
        [2023, 9, 22, 4],   # Fall Equinox (Friday)
        [2023, 12, 21, 3]   # Winter Solstice (Thursday)
    ], dtype="float32")
    
    # Apply cyclical encoding
    encoder = DateEncodingLayer(min_year=2000, max_year=2030)
    encoded = encoder(dates)
    
    # Analyze cyclical patterns
    print("Cyclical Feature Analysis:")
    print("=" * 50)
    print("Date\t\tYear\tMonth\tDay\tDOW\tYear_Sin\tYear_Cos\tMonth_Sin\tMonth_Cos")
    print("-" * 80)
    
    for i, date in enumerate(dates):
        year, month, day, dow = date.numpy()
        year_sin, year_cos, month_sin, month_cos, day_sin, day_cos, dow_sin, dow_cos = encoded[i].numpy()
        
        print(f"{int(year)}-{int(month):02d}-{int(day):02d}\t{int(year)}\t{int(month)}\t{int(day)}\t{int(dow)}\t"
              f"{year_sin:.3f}\t\t{year_cos:.3f}\t\t{month_sin:.3f}\t\t{month_cos:.3f}")
    
    return encoded

# Analyze cyclical features
# cyclical_data = analyze_cyclical_features()
```

## 💡 Tips & Best Practices

- **Year Range**: Set min_year and max_year based on your data's actual range
- **Input Format**: Input must be [year, month, day, day_of_week] format
- **Cyclical Nature**: The encoding preserves cyclical relationships
- **Neural Networks**: Works well with neural networks for temporal patterns
- **Seasonality**: Excellent for capturing seasonal and cyclical patterns
- **Integration**: Combines well with other temporal processing layers

## ⚠️ Common Pitfalls

- **Input Shape**: Must be (..., 4) tensor with date components
- **Year Range**: min_year must be less than max_year
- **Component Order**: Must be [year, month, day, day_of_week] in that order
- **Data Type**: Input should be float32 tensor
- **Missing Values**: Doesn't handle missing values - preprocess first

## 🔗 Related Layers

- [DateParsingLayer](date-parsing-layer.md) - Date string parsing
- [SeasonLayer](season-layer.md) - Seasonal information extraction
- [CastToFloat32Layer](cast-to-float32-layer.md) - Type casting utility
- [DifferentiableTabularPreprocessor](differentiable-tabular-preprocessor.md) - End-to-end preprocessing

## 📚 Further Reading

- [Cyclical Encoding in Time Series](https://en.wikipedia.org/wiki/Cyclical_encoding) - Cyclical encoding concepts
- [Sine and Cosine Transformations](https://en.wikipedia.org/wiki/Trigonometric_functions) - Trigonometric functions
- [Time Series Feature Engineering](https://en.wikipedia.org/wiki/Feature_engineering) - Feature engineering techniques
- [KMR Layer Explorer](../layers-explorer.md) - Browse all available layers
- [Data Preprocessing Tutorial](../tutorials/feature-engineering.md) - Complete guide to data preprocessing
