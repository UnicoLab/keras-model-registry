# Autoencoder

Anomaly Detection Model with Optional Preprocessing Integration

## Overview

Autoencoder is a neural network model designed for anomaly detection. It learns to reconstruct normal patterns and identifies anomalies as data points with high reconstruction error. The model can optionally integrate with preprocessing models for production use, making it a unified solution for both training and inference.

## Key Features

- **Anomaly Detection**: Identifies anomalies through reconstruction error
- **Adaptive Threshold**: Learns threshold based on training data distribution
- **Preprocessing Integration**: Optional preprocessing model for unified pipelines
- **Flexible Architecture**: Configurable encoding and intermediate dimensions
- **Production Ready**: Supports preprocessing models for deployment
- **Statistical Metrics**: Tracks median and standard deviation of anomaly scores

## Parameters

- **input_dim** (int): Dimension of the input data. Must be positive.
- **encoding_dim** (int, default=64): Dimension of the encoded representation.
- **intermediate_dim** (int, default=32): Dimension of the intermediate layer.
- **threshold** (float, default=2.0): Initial threshold for anomaly detection.
- **preprocessing_model** (Model, optional): Optional preprocessing model.
- **inputs** (dict[str, tuple], optional): Input shapes for preprocessing model.
- **name** (str, optional): Model name.

## Input/Output Shapes

**Input:**
- Shape: (batch_size, input_dim)
- Or dictionary with feature inputs when using preprocessing model
- Type: Float32

**Output:**
- Shape: (batch_size, input_dim) - Reconstructed input
- Type: Float32

## Architecture Flow

1. **Encoder**: Compresses input to encoding_dim
   - Dense layer to intermediate_dim
   - Dense layer to encoding_dim
2. **Decoder**: Reconstructs input from encoding
   - Dense layer to intermediate_dim
   - Dense layer to input_dim
3. **Reconstruction Error**: Computes difference between input and output
4. **Anomaly Detection**: Compares error to learned threshold

## Usage Example

```python
from kerasfactory.models import Autoencoder
import keras
import numpy as np

# Create model
model = Autoencoder(
    input_dim=32,
    encoding_dim=16,
    intermediate_dim=8,
    threshold=2.0
)

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Generate dummy data (normal patterns)
X_train = np.random.randn(1000, 32).astype('float32')

# Train on normal data
model.fit(X_train, X_train, epochs=50, batch_size=32)

# Detect anomalies
test_data = np.random.randn(100, 32).astype('float32')
reconstructions = model.predict(test_data)

# Get anomaly results
anomaly_results = model.is_anomaly(test_data)
print(anomaly_results.keys())  # ['reconstruction_error', 'anomaly', 'anomaly_score']
```

## Advanced Usage

### Anomaly Detection

```python
# Train model
model.fit(X_train, X_train, epochs=50)

# Detect anomalies
anomaly_results = model.is_anomaly(test_data)

# Access results
reconstruction_error = anomaly_results['reconstruction_error']
is_anomaly = anomaly_results['anomaly']
anomaly_score = anomaly_results['anomaly_score']

print(f"Anomalies detected: {is_anomaly.numpy().sum()}")
print(f"Anomaly scores: {anomaly_score.numpy()[:5]}")
```

### Custom Threshold

```python
# Create model with custom threshold
model = Autoencoder(
    input_dim=32,
    encoding_dim=16,
    threshold=3.0  # Higher threshold = fewer anomalies detected
)

# Or update threshold after training
model.update_threshold(2.5)
```

### With Preprocessing Model

```python
from kerasfactory.utils.data_analyzer import DataAnalyzer
import pandas as pd

# Create preprocessing model
df = pd.DataFrame(np.random.randn(1000, 32))
analyzer = DataAnalyzer(df)
preprocessing_model = analyzer.create_preprocessing_model()

# Create model with preprocessing
model = Autoencoder(
    input_dim=32,
    encoding_dim=16,
    preprocessing_model=preprocessing_model
)

# Train
model.fit(X_train, X_train, epochs=50)
```

### Different Architectures

```python
# Small bottleneck (more compression)
model_small = Autoencoder(
    input_dim=32,
    encoding_dim=8,  # Smaller encoding
    intermediate_dim=4
)

# Large bottleneck (less compression)
model_large = Autoencoder(
    input_dim=32,
    encoding_dim=24,  # Larger encoding
    intermediate_dim=16
)

# Deep architecture (more layers)
# Note: You may need to modify the model to add more layers
model_deep = Autoencoder(
    input_dim=32,
    encoding_dim=16,
    intermediate_dim=8
)
```

### Evaluation Metrics

```python
import keras

# Create metrics
accuracy_metric = keras.metrics.BinaryAccuracy()
precision_metric = keras.metrics.Precision()
recall_metric = keras.metrics.Recall()

# Get predictions
anomaly_results = model.is_anomaly(test_data)
predicted_anomalies = anomaly_results['anomaly'].numpy().astype(np.float32)

# Update metrics
test_labels = (test_labels > 0).astype(np.float32)  # Convert to binary
accuracy_metric.update_state(test_labels, predicted_anomalies)
precision_metric.update_state(test_labels, predicted_anomalies)
recall_metric.update_state(test_labels, predicted_anomalies)

print(f"Accuracy: {accuracy_metric.result().numpy()}")
print(f"Precision: {precision_metric.result().numpy()}")
print(f"Recall: {recall_metric.result().numpy()}")
```

### Serialization

```python
# Save model
model.save('autoencoder_model.keras')

# Load model
loaded_model = keras.models.load_model('autoencoder_model.keras')

# Save weights only
model.save_weights('autoencoder_weights.h5')

# Load weights
model_new = Autoencoder(input_dim=32, encoding_dim=16)
model_new.load_weights('autoencoder_weights.h5')
```

## Best Use Cases

- **Anomaly Detection**: Identifying outliers in normal data patterns
- **Fraud Detection**: Detecting fraudulent transactions or activities
- **Quality Control**: Identifying defective products or processes
- **Network Security**: Detecting intrusions or unusual network behavior
- **Production Monitoring**: Detecting anomalies in production systems

## Performance Considerations

- **encoding_dim**: Smaller values create stronger compression but may lose important information
- **intermediate_dim**: Affects model capacity and reconstruction quality
- **threshold**: Higher values detect fewer anomalies (more conservative)
- **Training Data**: Should contain mostly normal patterns for best results
- **Input Dimension**: Higher dimensions require more capacity

## Architecture Details

- **Encoder**: Compresses input to lower-dimensional representation
- **Decoder**: Reconstructs input from compressed representation
- **Reconstruction Error**: Measures how well the model reconstructs input
- **Adaptive Threshold**: Learned threshold based on training data statistics
- **Anomaly Score**: Normalized reconstruction error for easier interpretation

## Anomaly Detection Workflow

1. **Train on Normal Data**: Model learns to reconstruct normal patterns
2. **Compute Reconstruction Error**: Measure error for new data points
3. **Compare to Threshold**: Identify points with error above threshold
4. **Update Statistics**: Track median and std of anomaly scores
5. **Adjust Threshold**: Fine-tune threshold based on validation data

## Notes

- The model learns to reconstruct normal patterns during training
- Anomalies are identified as data points with high reconstruction error
- The threshold is adaptive and can be updated based on validation data
- Preprocessing model integration enables unified training/inference pipelines
- The model tracks statistical metrics (median, std) for threshold adjustment
- Reconstruction error is normalized to create anomaly scores for easier interpretation
- Boolean anomaly flags are converted to float32 for compatibility with Keras metrics

