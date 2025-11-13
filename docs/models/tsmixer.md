# TSMixer

MLP-based Multivariate Time Series Forecasting Model

## Overview

TSMixer (Time-Series Mixer) is an all-MLP architecture for multivariate time series forecasting. It jointly learns temporal and cross-sectional representations by repeatedly combining time- and feature information using stacked mixing layers. Unlike transformer-based architectures, TSMixer is computationally efficient and interpretable.

## Key Features

- **All-MLP Architecture**: No attention mechanisms or complex attention patterns
- **Temporal & Feature Mixing**: Alternating MLPs across time and feature dimensions
- **Reversible Instance Normalization**: Optional normalization for improved training
- **Multivariate Support**: Handles multiple related time series simultaneously
- **Residual Connections**: Enables training of deep architectures
- **Efficient**: Linear computational complexity in sequence length

## Parameters

- **seq_len** (int): Sequence length (number of lookback steps). Must be positive.
- **pred_len** (int): Prediction length (forecast horizon). Must be positive.
- **n_features** (int): Number of features/time series. Must be positive.
- **n_blocks** (int, default=2): Number of mixing layers in the model.
- **ff_dim** (int, default=64): Hidden dimension for feed-forward networks.
- **dropout** (float, default=0.1): Dropout rate between 0 and 1.
- **use_norm** (bool, default=True): Whether to use Reversible Instance Normalization.
- **name** (str, optional): Model name.

## Input/Output Shapes

**Input:**
- Shape: (batch_size, seq_len, n_features)
- Type: Float32

**Output:**
- Shape: (batch_size, pred_len, n_features)
- Type: Float32

## Architecture Flow

1. **Instance Normalization** (optional): Normalize input to zero mean and unit variance
2. **Stacked Mixing Layers**: Apply n_blocks mixing layers sequentially
   - Each layer combines TemporalMixing and FeatureMixing
3. **Output Projection**: Project temporal dimension from seq_len to pred_len
4. **Reverse Instance Normalization** (optional): Denormalize output

## Usage Example

```python
from kerasfactory.models import TSMixer
import keras

# Create model
model = TSMixer(
    seq_len=96,
    pred_len=12,
    n_features=7,
    n_blocks=2,
    ff_dim=64,
    dropout=0.1,
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

### Model with Different Configurations

```python
# Small model for fast training
small_model = TSMixer(
    seq_len=96, pred_len=12, n_features=7,
    n_blocks=1, ff_dim=32, dropout=0.1
)

# Large model for high accuracy
large_model = TSMixer(
    seq_len=96, pred_len=12, n_features=7,
    n_blocks=4, ff_dim=256, dropout=0.2
)

# Model without normalization
no_norm_model = TSMixer(
    seq_len=96, pred_len=12, n_features=7,
    n_blocks=2, ff_dim=64, use_norm=False
)
```

### Serialization

```python
# Save model
model.save('tsmixer_model.keras')

# Load model
loaded_model = keras.models.load_model('tsmixer_model.keras')

# Save weights only
model.save_weights('tsmixer_weights.h5')

# Load weights
model_new = TSMixer(seq_len=96, pred_len=12, n_features=7)
model_new.load_weights('tsmixer_weights.h5')
```

## Best Use Cases

- **Multivariate Time Series Forecasting**: Multiple related time series with complex dependencies
- **Efficient Models**: When computational efficiency is critical
- **Interpretability**: All-MLP models are more interpretable than attention-based methods
- **Long Sequences**: Linear complexity allows handling long sequences
- **Resource-Constrained Environments**: Lower memory footprint than transformers

## Performance Considerations

- **seq_len**: Larger values capture longer-term dependencies but increase computation
- **n_blocks**: More blocks improve performance but increase model size and training time
- **ff_dim**: Larger dimensions improve expressiveness but increase parameters
- **dropout**: Helps prevent overfitting; use higher values with limited data
- **use_norm**: Instance normalization can improve training stability

## Comparison with Other Architectures

### vs. Transformers
- **Advantage**: Simpler, more efficient, linear complexity
- **Disadvantage**: May not capture long-range dependencies as well

### vs. LSTM/GRU
- **Advantage**: Parallel processing, faster training
- **Disadvantage**: Different inductive bias for temporal sequences

### vs. NLinear/DLinear
- **Advantage**: Captures both temporal and feature interactions
- **Disadvantage**: More parameters and complexity

## References

Chen, Si-An, Chun-Liang Li, Nate Yoder, Sercan O. Arik, and Tomas Pfister (2023).
"TSMixer: An All-MLP Architecture for Time Series Forecasting."
arXiv preprint arXiv:2303.06053.

## Notes

- Instance normalization (RevIN) is enabled by default and helps with training
- Residual connections in mixing layers prevent gradient issues in deep models
- Batch normalization parameters in mixing layers are learned during training
- The model is fully differentiable and supports all Keras optimizers and losses
