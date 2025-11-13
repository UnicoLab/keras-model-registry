# TimeMixer

Decomposable Multi-Scale Mixing for Time Series Forecasting

## Overview

TimeMixer is a state-of-the-art time series forecasting model that uses series decomposition and multi-scale mixing to capture both trend and seasonal patterns. It employs a decomposable architecture that separates trend and seasonal components, then applies multi-scale mixing operations to learn complex temporal patterns.

## Key Features

- **Series Decomposition**: Separates trend and seasonal components using moving average or DFT decomposition
- **Multi-Scale Mixing**: Captures patterns at different time scales through downsampling layers
- **Reversible Instance Normalization**: Optional normalization for improved training stability
- **Channel Independence**: Supports both channel-dependent and channel-independent processing
- **Flexible Architecture**: Configurable encoder layers, downsampling, and decomposition methods
- **Efficient**: Designed for multivariate time series forecasting with linear complexity

## Parameters

- **seq_len** (int): Input sequence length (number of lookback steps). Must be positive.
- **pred_len** (int): Prediction horizon (forecast length). Must be positive.
- **n_features** (int): Number of time series features. Must be positive.
- **d_model** (int, default=32): Model dimension (hidden size).
- **d_ff** (int, default=32): Feed-forward network dimension.
- **e_layers** (int, default=4): Number of encoder layers.
- **dropout** (float, default=0.1): Dropout rate between 0 and 1.
- **decomp_method** (str, default='moving_avg'): Decomposition method ('moving_avg' or 'dft_decomp').
- **moving_avg** (int, default=25): Moving average window size for trend extraction.
- **top_k** (int, default=5): Top-k frequencies for DFT decomposition.
- **channel_independence** (int, default=0): 0 for channel-dependent, 1 for independent processing.
- **down_sampling_layers** (int, default=1): Number of downsampling layers.
- **down_sampling_window** (int, default=2): Downsampling window size.
- **down_sampling_method** (str, default='avg'): Downsampling method ('avg', 'max', or 'conv').
- **use_norm** (bool, default=True): Whether to use Reversible Instance Normalization.
- **decoder_input_size_multiplier** (float, default=0.5): Decoder input size multiplier.
- **name** (str, optional): Model name.

## Input/Output Shapes

**Input:**
- Shape: (batch_size, seq_len, n_features)
- Type: Float32

**Output:**
- Shape: (batch_size, pred_len, n_features)
- Type: Float32

## Architecture Flow

1. **Data Embedding**: Embed input without positional encoding
2. **Reversible Instance Normalization** (optional): Normalize input
3. **Series Decomposition**: Separate trend and seasonal components
4. **Multi-Scale Encoder**: Apply encoder layers with downsampling
5. **Past Decomposable Mixing**: Mix trend and seasonal components
6. **Output Projection**: Generate predictions
7. **Reverse Instance Normalization** (optional): Denormalize output

## Usage Example

```python
from kerasfactory.models import TimeMixer
import keras

# Create model
model = TimeMixer(
    seq_len=96,
    pred_len=12,
    n_features=7,
    d_model=32,
    d_ff=32,
    e_layers=2,
    dropout=0.1,
    decomp_method='moving_avg',
    use_norm=True
)

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse'
)

# Generate dummy data
import numpy as np
X_train = np.random.randn(100, 96, 7).astype('float32')
y_train = np.random.randn(100, 12, 7).astype('float32')

# Train
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Predict
predictions = model.predict(X_train[:5])
print(predictions.shape)  # (5, 12, 7)
```

## Advanced Usage

### Different Decomposition Methods

```python
# Moving average decomposition (default)
model_ma = TimeMixer(
    seq_len=96, pred_len=12, n_features=7,
    decomp_method='moving_avg',
    moving_avg=25
)

# DFT decomposition
model_dft = TimeMixer(
    seq_len=96, pred_len=12, n_features=7,
    decomp_method='dft_decomp',
    top_k=5
)
```

### Channel Independence

```python
# Channel-dependent processing (default)
model_dependent = TimeMixer(
    seq_len=96, pred_len=12, n_features=7,
    channel_independence=0
)

# Channel-independent processing
model_independent = TimeMixer(
    seq_len=96, pred_len=12, n_features=7,
    channel_independence=1
)
```

### Downsampling Configuration

```python
# With downsampling
model_downsample = TimeMixer(
    seq_len=96, pred_len=12, n_features=7,
    down_sampling_layers=2,
    down_sampling_window=4,
    down_sampling_method='avg'
)
```

### Serialization

```python
# Save model
model.save('timemixer_model.keras')

# Load model
loaded_model = keras.models.load_model('timemixer_model.keras')

# Save weights only
model.save_weights('timemixer_weights.h5')

# Load weights
model_new = TimeMixer(seq_len=96, pred_len=12, n_features=7)
model_new.load_weights('timemixer_weights.h5')
```

## Best Use Cases

- **Multivariate Time Series Forecasting**: Multiple related time series with trend and seasonal patterns
- **Long-Horizon Forecasting**: Effective for longer prediction horizons
- **Complex Temporal Patterns**: Captures both trend and seasonal components
- **Multi-Scale Patterns**: Handles patterns at different time scales through downsampling
- **Production Systems**: Efficient inference with optional normalization

## Performance Considerations

- **seq_len**: Larger values capture longer-term dependencies but increase computation
- **e_layers**: More encoder layers improve capacity but increase training time
- **d_model**: Larger dimensions improve expressiveness but increase parameters
- **decomp_method**: Moving average is faster, DFT may capture more complex patterns
- **down_sampling_layers**: More layers capture multi-scale patterns but increase complexity
- **use_norm**: Instance normalization improves training stability, especially for non-stationary data

## Comparison with Other Architectures

### vs. TSMixer
- **Advantage**: Multi-scale mixing and decomposition for better pattern capture
- **Disadvantage**: More complex architecture with more hyperparameters

### vs. Transformers
- **Advantage**: More efficient, explicit decomposition of trend/seasonal
- **Disadvantage**: May not capture very long-range dependencies as well

### vs. NLinear/DLinear
- **Advantage**: Multi-scale mixing and flexible decomposition methods
- **Disadvantage**: More parameters and complexity

## Notes

- Reversible Instance Normalization (RevIN) is enabled by default and helps with non-stationary data
- Series decomposition separates trend and seasonal components for better pattern learning
- Multi-scale downsampling captures patterns at different temporal resolutions
- Channel independence option allows flexible feature interaction modeling
- The model supports both moving average and DFT decomposition methods

