# 📊 Metrics API Reference

Welcome to the KerasFactory Metrics documentation! All metrics are designed to work exclusively with **Keras 3** and provide specialized statistical measurements for model analysis and anomaly detection tasks.

!!! tip "What You'll Find Here"
    Each metric includes detailed documentation with:
    - ✨ **Complete parameter descriptions** with types and defaults
    - 🎯 **Usage examples** showing real-world applications
    - ⚡ **Best practices** and performance considerations
    - 🎨 **When to use** guidance for each metric
    - 🔧 **Implementation notes** for developers

!!! success "Ready-to-Use Metrics"
    These metrics provide specialized implementations for statistical analysis that you can use out-of-the-box or integrate into your models.

!!! note "Keras 3 Compatible"
    All metrics are built on top of Keras base classes and are fully compatible with Keras 3.

## 📊 Statistical Metrics

### 📈 Median
Calculates the median of predicted values, providing a robust measure of central tendency less sensitive to outliers.

::: kerasfactory.metrics.Median

### 📉 StandardDeviation
Calculates the standard deviation of predicted values, useful for tracking prediction variability and uncertainty.

::: kerasfactory.metrics.StandardDeviation
