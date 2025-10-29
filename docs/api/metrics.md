# 📊 Metrics API Reference

Welcome to the KMR Metrics documentation! All metrics are designed to work exclusively with **Keras 3** and provide custom implementations for specialized evaluation tasks.

!!! tip "What You'll Find Here"
    Each metric includes detailed documentation with:
    - ✨ **Complete parameter descriptions** with types and defaults
    - 🎯 **Usage examples** showing real-world applications
    - ⚡ **Best practices** and performance considerations
    - 🎨 **When to use** guidance for each metric
    - 🔧 **Implementation notes** for developers

!!! success "Ready-to-Use Metrics"
    These metrics provide specialized functionality that extends Keras' built-in metrics for advanced use cases.

!!! note "Keras 3 Compatibility"
    All metrics inherit from `keras.metrics.Metric` ensuring full compatibility with Keras 3 and TensorFlow backends.

## 📊 Custom Metrics

### 📈 StandardDeviation
Custom metric for calculating the standard deviation of predictions, useful for anomaly detection and uncertainty quantification.

::: kmr.metrics.standard_deviation.StandardDeviation

### 📊 Median
Custom metric for calculating the median of predictions, providing robust central tendency measures for anomaly detection.

::: kmr.metrics.median.Median

## 🔧 Usage Examples

### Basic Usage

```python
from kmr.metrics import StandardDeviation, Median
import keras

# Create metrics
std_metric = StandardDeviation()
median_metric = Median()

# Update with predictions
predictions = keras.ops.random.normal((100, 1))
std_metric.update_state(predictions)
median_metric.update_state(predictions)

# Get results
print(f"Standard Deviation: {std_metric.result()}")
print(f"Median: {median_metric.result()}")
```

### Integration with Autoencoder

```python
from kmr.models import Autoencoder
from kmr.metrics import StandardDeviation, Median

# Create autoencoder
model = Autoencoder(input_dim=100, encoding_dim=32)

# Metrics are automatically used in threshold setup
model.fit(data, epochs=10, auto_setup_threshold=True)
```

## 🎯 Best Practices

1. **Reset State**: Always reset metric state between epochs or evaluation runs
2. **Batch Processing**: Update metrics with batches for better performance
3. **Memory Management**: Use `reset_state()` to clear accumulated values
4. **Integration**: These metrics work seamlessly with Keras training loops

## 🔧 Implementation Notes

- All metrics follow Keras 3 conventions
- Full serialization support with `get_config()` and `from_config()`
- Compatible with TensorFlow and other Keras backends
- Optimized for both eager and graph execution modes
