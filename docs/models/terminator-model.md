# TerminatorModel

Advanced Feature Processing Model with Stacked SFNE Blocks

## Overview

TerminatorModel combines multiple SFNE (Slow-Fast Neural Engine) blocks for advanced feature processing. It's designed for complex tabular data modeling tasks where feature interactions are important. The model stacks multiple SFNE blocks to process features in a hierarchical manner, enabling deep feature interactions and complex pattern learning.

## Key Features

- **Stacked Architecture**: Multiple SFNE blocks for hierarchical feature processing
- **Dual Input Support**: Handles both input features and context features
- **Deep Feature Interactions**: Enables complex feature relationship learning
- **Flexible Configuration**: Configurable number of blocks and network dimensions
- **Preprocessing Integration**: Optional preprocessing model support
- **Production Ready**: Supports unified training/inference pipelines

## Parameters

- **input_dim** (int): Dimension of the input features. Must be positive.
- **context_dim** (int): Dimension of the context features. Must be positive.
- **output_dim** (int): Dimension of the output. Must be positive.
- **hidden_dim** (int, default=64): Number of hidden units in the network.
- **num_layers** (int, default=2): Number of layers in each SFNE block.
- **num_blocks** (int, default=3): Number of SFNE blocks to stack.
- **slow_network_layers** (int, default=3): Number of layers in each slow network.
- **slow_network_units** (int, default=128): Number of units per layer in each slow network.
- **preprocessing_model** (Model, optional): Optional preprocessing model.
- **name** (str, optional): Model name.

## Input/Output Shapes

**Input:**
- List of 2D tensors: `[(batch_size, input_dim), (batch_size, context_dim)]`
- Or dictionary with feature inputs when using preprocessing model
- Type: Float32

**Output:**
- Shape: (batch_size, output_dim)
- Type: Float32

## Architecture Flow

1. **Input Processing**: Separate input and context features
2. **Stacked SFNE Blocks**: Apply num_blocks SFNE blocks sequentially
   - Each block processes features hierarchically
   - Features flow through slow and fast paths
3. **Feature Combination**: Combine processed features
4. **Output Projection**: Project to output_dim

## Usage Example

```python
from kerasfactory.models import TerminatorModel
import keras
import numpy as np

# Create model
model = TerminatorModel(
    input_dim=16,
    context_dim=8,
    output_dim=1,
    hidden_dim=64,
    num_layers=2,
    num_blocks=3,
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
X_input = np.random.randn(100, 16).astype('float32')
X_context = np.random.randn(100, 8).astype('float32')
y_train = np.random.randn(100, 1).astype('float32')

# Train
model.fit([X_input, X_context], y_train, epochs=10, batch_size=32)

# Predict
predictions = model.predict([X_input, X_context])
print(predictions.shape)  # (100, 1)
```

## Advanced Usage

### Different Configurations

```python
# Small model
small_model = TerminatorModel(
    input_dim=16,
    context_dim=8,
    output_dim=1,
    hidden_dim=32,
    num_layers=1,
    num_blocks=2,
    slow_network_layers=2,
    slow_network_units=64
)

# Large model
large_model = TerminatorModel(
    input_dim=16,
    context_dim=8,
    output_dim=1,
    hidden_dim=128,
    num_layers=3,
    num_blocks=5,
    slow_network_layers=4,
    slow_network_units=256
)
```

### With Preprocessing Model

```python
from kerasfactory.utils.data_analyzer import DataAnalyzer
import pandas as pd

# Create preprocessing model for both inputs
df_input = pd.DataFrame(np.random.randn(100, 16))
df_context = pd.DataFrame(np.random.randn(100, 8))

analyzer_input = DataAnalyzer(df_input)
analyzer_context = DataAnalyzer(df_context)

preprocessing_model_input = analyzer_input.create_preprocessing_model()
preprocessing_model_context = analyzer_context.create_preprocessing_model()

# Note: You may need to combine preprocessing models or use separate models
# This is a simplified example
model = TerminatorModel(
    input_dim=16,
    context_dim=8,
    output_dim=1,
    preprocessing_model=preprocessing_model_input  # Simplified
)
```

### Regression Task

```python
# Regression with multiple outputs
model_regression = TerminatorModel(
    input_dim=16,
    context_dim=8,
    output_dim=3,  # Multiple outputs
    hidden_dim=64,
    num_blocks=3
)

model_regression.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae', 'mse']
)
```

### Classification Task

```python
# Binary classification
model_binary = TerminatorModel(
    input_dim=16,
    context_dim=8,
    output_dim=1,
    hidden_dim=64,
    num_blocks=3
)

model_binary.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Multi-class classification
model_multiclass = TerminatorModel(
    input_dim=16,
    context_dim=8,
    output_dim=10,  # 10 classes
    hidden_dim=64,
    num_blocks=3
)

model_multiclass.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Serialization

```python
# Save model
model.save('terminator_model.keras')

# Load model
loaded_model = keras.models.load_model('terminator_model.keras')

# Save weights only
model.save_weights('terminator_weights.h5')

# Load weights
model_new = TerminatorModel(
    input_dim=16,
    context_dim=8,
    output_dim=1
)
model_new.load_weights('terminator_weights.h5')
```

## Best Use Cases

- **Complex Tabular Data**: When feature interactions are important
- **Dual Input Scenarios**: When you have both main features and context features
- **Deep Feature Learning**: When you need hierarchical feature processing
- **High-Dimensional Data**: When you need to learn complex feature relationships
- **Production Systems**: With preprocessing model integration

## Performance Considerations

- **num_blocks**: More blocks enable deeper feature interactions but increase computation
- **hidden_dim**: Larger values improve capacity but increase parameters
- **num_layers**: More layers per block can learn complex patterns but may overfit
- **slow_network_units**: Larger values improve hyper-kernel quality but increase parameters
- **input_dim/context_dim**: Match your data dimensions

## Architecture Details

- **Stacked SFNE Blocks**: Each block processes features hierarchically
- **Dual Input**: Separate processing for input and context features
- **Hierarchical Processing**: Features flow through multiple processing stages
- **Adaptive Kernels**: Slow networks generate context-dependent kernels
- **Multi-Scale Features**: Global and local feature processing

## Comparison with Other Models

### vs. BaseFeedForwardModel
- **Advantage**: Deeper feature interactions, dual input support
- **Disadvantage**: More complex, higher computational cost

### vs. SFNEBlock
- **Advantage**: Stacked architecture for deeper processing
- **Disadvantage**: More parameters, longer training time

## Notes

- TerminatorModel stacks multiple SFNE blocks for hierarchical feature processing
- The model supports both input features and context features
- More blocks enable deeper feature interactions but increase computation
- The architecture is designed for complex tabular data modeling tasks
- Preprocessing model integration enables unified training/inference pipelines
- Each SFNE block processes features through slow and fast paths

