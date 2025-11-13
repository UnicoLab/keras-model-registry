# SFNEBlock

Slow-Fast Neural Engine Block for Advanced Feature Processing

## Overview

SFNEBlock (Slow-Fast Neural Engine Block) combines slow and fast processing paths for feature extraction. It uses a SlowNetwork to generate hyper-kernels, which are then processed by a HyperZZWOperator to compute context-dependent weights. These weights are further processed through global and local convolutions before being combined. This architecture is designed as a building block for complex tabular data modeling tasks.

## Key Features

- **Dual-Path Architecture**: Slow and fast processing paths for multi-scale feature extraction
- **Hyper-Kernel Generation**: SlowNetwork generates adaptive kernels for feature processing
- **Context-Dependent Weights**: HyperZZWOperator computes dynamic weights based on input context
- **Multi-Scale Processing**: Global and local convolutions capture different feature scales
- **Flexible Dimensions**: Configurable input/output dimensions
- **Preprocessing Support**: Optional preprocessing model integration

## Parameters

- **input_dim** (int): Dimension of the input features. Must be positive.
- **output_dim** (int, optional): Dimension of the output features. Defaults to input_dim.
- **hidden_dim** (int, default=64): Number of hidden units in the network.
- **num_layers** (int, default=2): Number of layers in the network.
- **slow_network_layers** (int, default=3): Number of layers in the slow network.
- **slow_network_units** (int, default=128): Number of units per layer in the slow network.
- **preprocessing_model** (Model, optional): Optional preprocessing model.
- **name** (str, optional): Model name.

## Input/Output Shapes

**Input:**
- Shape: (batch_size, input_dim)
- Or dictionary with feature inputs when using preprocessing model
- Type: Float32

**Output:**
- Shape: (batch_size, output_dim)
- Type: Float32

## Architecture Flow

1. **Input Processing**: Dense layer to hidden_dim
2. **Hidden Layers**: Multiple dense layers with ReLU activation
3. **Slow Network**: Generates hyper-kernels for adaptive processing
4. **HyperZZWOperator**: Computes context-dependent weights from hyper-kernels
5. **Global Convolution**: Processes features globally
6. **Local Convolution**: Processes features locally
7. **Combination**: Combines global and local features
8. **Output Projection**: Projects to output_dim

## Usage Example

```python
from kerasfactory.models import SFNEBlock
import keras
import numpy as np

# Create model
model = SFNEBlock(
    input_dim=16,
    output_dim=8,
    hidden_dim=64,
    num_layers=2,
    slow_network_layers=3,
    slow_network_units=128
)

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Generate dummy data
X_train = np.random.randn(100, 16).astype('float32')
y_train = np.random.randn(100, 8).astype('float32')

# Train
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Predict
predictions = model.predict(X_train)
print(predictions.shape)  # (100, 8)
```

## Advanced Usage

### Different Configurations

```python
# Small model
small_model = SFNEBlock(
    input_dim=16,
    output_dim=8,
    hidden_dim=32,
    num_layers=1,
    slow_network_layers=2,
    slow_network_units=64
)

# Large model
large_model = SFNEBlock(
    input_dim=16,
    output_dim=8,
    hidden_dim=128,
    num_layers=3,
    slow_network_layers=4,
    slow_network_units=256
)

# Same input/output dimension
same_dim_model = SFNEBlock(
    input_dim=16,
    output_dim=16,  # Explicitly set
    hidden_dim=64
)
```

### With Preprocessing Model

```python
from kerasfactory.utils.data_analyzer import DataAnalyzer
import pandas as pd

# Create preprocessing model
df = pd.DataFrame(np.random.randn(100, 16))
analyzer = DataAnalyzer(df)
preprocessing_model = analyzer.create_preprocessing_model()

# Create model with preprocessing
model = SFNEBlock(
    input_dim=16,
    output_dim=8,
    preprocessing_model=preprocessing_model
)
```

### Feature Extraction

```python
# Use as feature extractor
feature_extractor = SFNEBlock(
    input_dim=64,
    output_dim=32,  # Reduced dimension
    hidden_dim=128
)

# Extract features
features = feature_extractor(X_train)
print(features.shape)  # (100, 32)
```

### Serialization

```python
# Save model
model.save('sfne_block_model.keras')

# Load model
loaded_model = keras.models.load_model('sfne_block_model.keras')

# Save weights only
model.save_weights('sfne_block_weights.h5')

# Load weights
model_new = SFNEBlock(input_dim=16, output_dim=8)
model_new.load_weights('sfne_block_weights.h5')
```

## Best Use Cases

- **Feature Extraction**: Advanced feature processing for tabular data
- **Building Block**: Component for larger architectures (e.g., TerminatorModel)
- **Complex Feature Interactions**: When you need multi-scale feature processing
- **Adaptive Processing**: When feature processing should adapt to input context
- **Dimensionality Reduction**: Flexible input/output dimensions

## Performance Considerations

- **hidden_dim**: Larger values improve capacity but increase parameters
- **num_layers**: More layers can learn complex patterns but may overfit
- **slow_network_layers**: More layers generate richer hyper-kernels but increase computation
- **slow_network_units**: Larger values improve hyper-kernel quality but increase parameters
- **output_dim**: Match your downstream task requirements

## Architecture Details

- **Slow Network**: Generates hyper-kernels that adapt to input patterns
- **HyperZZWOperator**: Computes context-dependent weights dynamically
- **Dual Convolution**: Global and local convolutions capture different scales
- **Residual Connections**: Help with gradient flow in deep architectures
- **Adaptive Processing**: Weights adapt based on input context

## Notes

- SFNEBlock is designed as a building block for larger architectures
- The slow network generates adaptive kernels for context-dependent processing
- Global and local convolutions capture features at different scales
- The model supports flexible input/output dimensions
- Preprocessing model integration enables unified training/inference pipelines
- Commonly used as a component in TerminatorModel for complex tabular data tasks

