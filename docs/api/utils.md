# 🔧 Utils API Reference

Welcome to the KMR Utilities documentation! All utilities are designed to work exclusively with **Keras 3** and provide powerful tools for data analysis, generation, visualization, and development support.

!!! tip "What You'll Find Here"
    Each utility includes detailed documentation with:
    - ✨ **Complete parameter descriptions** with types and defaults
    - 🎯 **Usage examples** showing real-world applications
    - ⚡ **Best practices** and performance considerations
    - 🎨 **When to use** guidance for each utility
    - 🔧 **Implementation notes** for developers

!!! success "Comprehensive Toolkit"
    The KMR utilities provide intelligent data analysis, synthetic data generation, and professional visualization capabilities.

!!! note "Developer-Friendly"
    All utilities are designed for easy integration into your data science workflows and Jupyter notebooks.

## 🔍 Data Analysis

### 🧠 DataAnalyzer
Intelligent data analyzer that examines CSV files and recommends appropriate KMR layers based on data characteristics.

::: kmr.utils.data_analyzer.DataAnalyzer

### 💻 DataAnalyzerCLI
Command-line interface for the data analyzer, allowing easy analysis of datasets from the terminal.

::: kmr.utils.data_analyzer_cli

## 📊 Data Generation

### 🎲 KMRDataGenerator
Utility class for generating synthetic datasets for KMR model testing, demonstrations, and experimentation.

Features:
- **Tabular Data**: Regression, classification, anomaly detection, multi-input data
- **Time Series**: Basic, multivariate, seasonal, multi-scale, anomalous, long-horizon, energy demand
- **Dataset Creation**: Easy conversion to TensorFlow datasets with batching and shuffling

::: kmr.utils.data_generator.KMRDataGenerator

#### Time Series Methods

##### `generate_timeseries_data()`
Generates synthetic multivariate time series with optional trend and seasonality.

```python
from kmr.utils import KMRDataGenerator

X, y = KMRDataGenerator.generate_timeseries_data(
    n_samples=1000,
    seq_len=96,           # Input sequence length
    pred_len=12,          # Prediction horizon
    n_features=7,         # Number of channels
    include_trend=True,
    include_seasonality=True,
    trend_direction="up"  # "up", "down", "random"
)
```

##### `generate_multivariate_timeseries()`
Generates time series with inter-feature correlations for realistic multivariate data.

```python
X, y = KMRDataGenerator.generate_multivariate_timeseries(
    n_samples=1000,
    seq_len=96,
    pred_len=12,
    n_features=7,
    correlation_strength=0.5  # 0-1
)
```

##### `generate_seasonal_timeseries()`
Emphasized seasonal patterns, ideal for decomposition models like TimeMixer.

```python
X, y = KMRDataGenerator.generate_seasonal_timeseries(
    n_samples=1000,
    seq_len=96,
    pred_len=12,
    n_features=7,
    seasonal_period=12
)
```

##### `generate_multiscale_timeseries()`
Components at different frequencies for testing multi-scale mixing models.

```python
X, y = KMRDataGenerator.generate_multiscale_timeseries(
    n_samples=1000,
    seq_len=96,
    pred_len=12,
    n_features=7,
    scales=[7, 14, 28, 56]
)
```

##### `generate_anomalous_timeseries()`
Time series with injected anomalies for testing anomaly detection models.

```python
X, y, labels = KMRDataGenerator.generate_anomalous_timeseries(
    n_samples=1000,
    seq_len=96,
    pred_len=12,
    n_features=7,
    anomaly_ratio=0.1,        # 10% anomalies
    anomaly_magnitude=3.0     # 3 std deviations
)
```

##### `generate_long_horizon_timeseries()`
For benchmarking long-term forecasting (e.g., 2 weeks ahead).

```python
X, y = KMRDataGenerator.generate_long_horizon_timeseries(
    n_samples=500,
    seq_len=336,   # 2 weeks hourly
    pred_len=336,  # Forecast 2 weeks
    n_features=7
)
```

##### `generate_synthetic_energy_demand()`
Realistic energy consumption patterns with daily/weekly seasonality.

```python
X, y = KMRDataGenerator.generate_synthetic_energy_demand(
    n_samples=1000,
    seq_len=168,   # 1 week
    pred_len=24,   # 1 day forecast
    n_features=3   # Residential, Commercial, Industrial
)
```

##### `create_timeseries_dataset()`
Converts numpy arrays to TensorFlow datasets with batching and auto-tuning.

```python
dataset = KMRDataGenerator.create_timeseries_dataset(
    X=X_train,
    y=y_train,
    batch_size=32,
    shuffle=True
)

model.fit(dataset, epochs=10)
```

## 🎨 Visualization

### 📈 KMRPlotter
Utility class for creating consistent and professional visualizations for KMR models, metrics, and data analysis.

Features:
- **Time Series Visualization**: Multiple visualization styles for forecasts
- **Training History**: Training and validation metrics
- **Classification Metrics**: ROC, precision-recall, confusion matrix
- **Anomaly Detection**: Anomaly score distributions
- **Performance Metrics**: Bar charts and comparison visualizations

::: kmr.utils.plotting.KMRPlotter

#### Time Series Plotting Methods

##### `plot_timeseries()`
Plot time series with input, true target, and predictions for multiple samples.

```python
from kmr.utils import KMRPlotter

fig = KMRPlotter.plot_timeseries(
    X=X_test,
    y_true=y_test,
    y_pred=predictions,
    n_samples_to_plot=5,
    feature_idx=0,  # Which feature to plot
    title="Time Series Forecast"
)
fig.show()
```

**Use When**: Visualizing multiple forecast examples side-by-side to understand model behavior.

##### `plot_timeseries_comparison()`
Compare single forecast with true values.

```python
fig = KMRPlotter.plot_timeseries_comparison(
    y_true=y_test,
    y_pred=predictions,
    sample_idx=0,
    title="Forecast Comparison"
)
fig.show()
```

**Use When**: Detailed analysis of a single sample forecast.

##### `plot_decomposition()`
Visualize time series decomposition into components (trend, seasonal, residual).

```python
fig = KMRPlotter.plot_decomposition(
    original=time_series,
    trend=trend_component,
    seasonal=seasonal_component,
    residual=residual_component,
    title="Time Series Decomposition"
)
fig.show()
```

**Use When**: Understanding component contributions in time series models.

##### `plot_forecasting_metrics()`
Calculate and display MAE, RMSE, and MAPE metrics.

```python
fig = KMRPlotter.plot_forecasting_metrics(
    y_true=y_test,
    y_pred=predictions,
    title="Forecasting Performance"
)
fig.show()
```

**Use When**: Quick performance overview of forecasting model.

##### `plot_forecast_horizon_analysis()`
Analyze forecast error across different forecast horizons (how far ahead).

```python
fig = KMRPlotter.plot_forecast_horizon_analysis(
    y_true=y_test,
    y_pred=predictions,
    title="Error by Forecast Horizon"
)
fig.show()
```

**Use When**: Understanding if model degrades for longer forecasts.

##### `plot_multiple_features_forecast()`
Plot forecasts for multiple features side-by-side.

```python
fig = KMRPlotter.plot_multiple_features_forecast(
    X=X_test,
    y_true=y_test,
    y_pred=predictions,
    sample_idx=0,
    n_features_to_plot=4,
    title="Multi-Feature Forecast"
)
fig.show()
```

**Use When**: Comparing forecast quality across multiple time series channels.

#### Training & Metrics Methods

##### `plot_training_history()`
Visualize training and validation metrics over epochs.

```python
fig = KMRPlotter.plot_training_history(
    history=model.history,
    metrics=['loss', 'mae', 'accuracy'],
    title="Training Progress"
)
fig.show()
```

##### `plot_confusion_matrix()`
Heatmap of classification confusion matrix.

```python
fig = KMRPlotter.plot_confusion_matrix(
    y_true=y_test,
    y_pred=y_pred_labels,
    title="Confusion Matrix"
)
fig.show()
```

##### `plot_roc_curve()`
ROC curve with AUC score.

```python
fig = KMRPlotter.plot_roc_curve(
    y_true=y_test,
    y_scores=y_pred_probs,
    title="ROC Curve"
)
fig.show()
```

##### `plot_precision_recall_curve()`
Precision-recall curve visualization.

```python
fig = KMRPlotter.plot_precision_recall_curve(
    y_true=y_test,
    y_scores=y_pred_probs,
    title="Precision-Recall Curve"
)
fig.show()
```

##### `plot_anomaly_scores()`
Distribution of anomaly scores with threshold visualization.

```python
fig = KMRPlotter.plot_anomaly_scores(
    scores=anomaly_scores,
    labels=true_labels,
    threshold=5.0,
    title="Anomaly Scores"
)
fig.show()
```

##### `plot_performance_metrics()`
Bar chart of performance metrics.

```python
metrics = {
    "Accuracy": 0.95,
    "Precision": 0.92,
    "Recall": 0.88,
    "F1": 0.90
}

fig = KMRPlotter.plot_performance_metrics(metrics)
fig.show()
```

## 🛠️ Decorators

### ✨ Decorators
Utility decorators for common functionality in KMR components and enhanced development experience.

::: kmr.utils.decorators

## 📚 Complete Example

```python
from kmr.utils import KMRDataGenerator, KMRPlotter
from kmr.models import TSMixer
import keras

# 1. Generate synthetic time series data
X_train, y_train = KMRDataGenerator.generate_seasonal_timeseries(
    n_samples=500, seq_len=96, pred_len=12, n_features=7
)
X_test, y_test = KMRDataGenerator.generate_seasonal_timeseries(
    n_samples=100, seq_len=96, pred_len=12, n_features=7
)

# 2. Create model
model = TSMixer(seq_len=96, pred_len=12, n_features=7)
model.compile(optimizer='adam', loss='mse')

# 3. Train model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=10)

# 4. Visualize training
fig = KMRPlotter.plot_training_history(history, metrics=['loss'])
fig.show()

# 5. Make predictions
predictions = model.predict(X_test)

# 6. Visualize forecasts
fig = KMRPlotter.plot_timeseries(
    X_test, y_test, predictions, n_samples_to_plot=3
)
fig.show()

# 7. Analyze performance
fig = KMRPlotter.plot_forecasting_metrics(y_test, predictions)
fig.show()

# 8. Detailed analysis
fig = KMRPlotter.plot_forecast_horizon_analysis(y_test, predictions)
fig.show()
```

## 🎯 Best Practices

1. **Always use `KMRDataGenerator`** for synthetic data in notebooks
2. **Leverage `KMRPlotter`** for consistent visualizations across projects
3. **Create TensorFlow datasets** with `create_timeseries_dataset()` for efficient training
4. **Use semantic data generation** methods (e.g., `generate_seasonal_timeseries()`) that match your use case
5. **Chain visualizations** to tell a complete story about model performance

## 📦 Testing

All utilities are thoroughly tested. Run tests with:

```bash
pytest tests/utils/ -v
```

Test coverage includes:
- ✓ Time series generation with various configurations
- ✓ Data distribution validation  
- ✓ Plotting function robustness with edge cases
- ✓ Different data shapes and dimensions
- ✓ Error handling and validation
