# BaseFeedForwardModel

Flexible Feed-Forward Neural Network for Tabular Data

## Overview

BaseFeedForwardModel is a configurable feed-forward neural network designed for tabular data. It provides a flexible architecture with configurable hidden layers, activations, regularization options, and optional preprocessing integration. It's ideal for regression and classification tasks on structured data.

## Key Features

- **Flexible Architecture**: Configurable hidden layers and units
- **Feature-Based Inputs**: Named feature inputs for better interpretability
- **Regularization Options**: Dropout, kernel/bias regularizers and constraints
- **Preprocessing Integration**: Optional preprocessing model support
- **Customizable Activations**: Configurable activation functions
- **Production Ready**: Supports preprocessing models for unified training/inference

## Parameters

- **feature_names** (list[str]): List of feature names. Defines input structure.
- **hidden_units** (list[int]): List of hidden layer units. Each element is a layer.
- **output_units** (int, default=1): Number of output units.
- **dropout_rate** (float, default=0.0): Dropout rate between 0 and 1.
- **activation** (str, default='relu'): Activation function for hidden layers.
- **preprocessing_model** (Model, optional): Optional preprocessing model.
- **kernel_initializer** (str/Any, default='glorot_uniform'): Weight initializer.
- **bias_initializer** (str/Any, default='zeros'): Bias initializer.
- **kernel_regularizer** (str/Any, optional): Weight regularizer (L1/L2).
- **bias_regularizer** (str/Any, optional): Bias regularizer.
- **activity_regularizer** (str/Any, optional): Activity regularizer.
- **kernel_constraint** (str/Any, optional): Weight constraint.
- **bias_constraint** (str/Any, optional): Bias constraint.
- **name** (str, optional): Model name.

## Input/Output Shapes

**Input:**
- Dictionary of named features: `{feature_name: (batch_size, 1)}`
- Or single tensor: `(batch_size, n_features)` when using preprocessing model
- Type: Float32

**Output:**
- Shape: (batch_size, output_units)
- Type: Float32

## Architecture Flow

1. **Input Layers**: Create named input layers for each feature
2. **Concatenation**: Concatenate all feature inputs
3. **Hidden Layers**: Apply configurable dense layers with activation
4. **Dropout** (optional): Apply dropout regularization
5. **Output Layer**: Dense layer with output_units

## Usage Example

```python
from kerasfactory.models import BaseFeedForwardModel
import keras
import numpy as np

# Create model
model = BaseFeedForwardModel(
    feature_names=['feature1', 'feature2', 'feature3'],
    hidden_units=[64, 32, 16],
    output_units=1,
    dropout_rate=0.2,
    activation='relu'
)

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Generate dummy data
X_train = {
    'feature1': np.random.randn(100, 1).astype('float32'),
    'feature2': np.random.randn(100, 1).astype('float32'),
    'feature3': np.random.randn(100, 1).astype('float32')
}
y_train = np.random.randn(100, 1).astype('float32')

# Train
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Predict
predictions = model.predict(X_train)
print(predictions.shape)  # (100, 1)
```

## Advanced Usage

### With Regularization

```python
# L2 regularization
model_l2 = BaseFeedForwardModel(
    feature_names=['f1', 'f2', 'f3'],
    hidden_units=[64, 32],
    output_units=1,
    kernel_regularizer='l2',
    bias_regularizer='l2'
)

# L1 regularization
model_l1 = BaseFeedForwardModel(
    feature_names=['f1', 'f2', 'f3'],
    hidden_units=[64, 32],
    output_units=1,
    kernel_regularizer='l1'
)
```

### Classification Task

```python
# Binary classification
model_binary = BaseFeedForwardModel(
    feature_names=['f1', 'f2', 'f3'],
    hidden_units=[64, 32],
    output_units=1,
    activation='relu'
)

model_binary.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Multi-class classification
model_multiclass = BaseFeedForwardModel(
    feature_names=['f1', 'f2', 'f3'],
    hidden_units=[64, 32],
    output_units=10,  # 10 classes
    activation='relu'
)

model_multiclass.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### With Preprocessing Model

```python
from kerasfactory.utils.data_analyzer import DataAnalyzer
import pandas as pd

# Create preprocessing model
df = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'feature3': np.random.randn(100)
})

analyzer = DataAnalyzer(df)
preprocessing_model = analyzer.create_preprocessing_model()

# Create model with preprocessing
model = BaseFeedForwardModel(
    feature_names=['feature1', 'feature2', 'feature3'],
    hidden_units=[64, 32],
    output_units=1,
    preprocessing_model=preprocessing_model
)
```

### Different Activations

```python
# ReLU (default)
model_relu = BaseFeedForwardModel(
    feature_names=['f1', 'f2'],
    hidden_units=[64, 32],
    activation='relu'
)

# Tanh
model_tanh = BaseFeedForwardModel(
    feature_names=['f1', 'f2'],
    hidden_units=[64, 32],
    activation='tanh'
)

# Swish
model_swish = BaseFeedForwardModel(
    feature_names=['f1', 'f2'],
    hidden_units=[64, 32],
    activation='swish'
)
```

### Serialization

```python
# Save model
model.save('feedforward_model.keras')

# Load model
loaded_model = keras.models.load_model('feedforward_model.keras')

# Save weights only
model.save_weights('feedforward_weights.h5')

# Load weights
model_new = BaseFeedForwardModel(
    feature_names=['feature1', 'feature2', 'feature3'],
    hidden_units=[64, 32],
    output_units=1
)
model_new.load_weights('feedforward_weights.h5')
```

## Best Use Cases

- **Tabular Data**: Structured data with named features
- **Regression Tasks**: Continuous value prediction
- **Classification Tasks**: Binary and multi-class classification
- **Feature Engineering**: When you need explicit feature control
- **Production Systems**: With preprocessing model integration

## Performance Considerations

- **hidden_units**: Deeper networks (more layers) can learn complex patterns but may overfit
- **dropout_rate**: Higher dropout helps prevent overfitting; use 0.2-0.5 for small datasets
- **activation**: ReLU is default and works well; try Swish for better performance
- **regularization**: Use L2 regularization for weight decay, L1 for feature selection
- **output_units**: 1 for regression/binary classification, n_classes for multi-class

## Architecture Tips

- Start with 2-3 hidden layers for most problems
- Use dropout (0.2-0.5) when you have limited training data
- Increase hidden units gradually; 64-256 is a good range
- Use batch normalization (via preprocessing) for better training stability
- Regularization helps prevent overfitting on small datasets

## Notes

- Feature names define the input structure and must match your data
- All features are concatenated before passing through hidden layers
- Dropout is applied between hidden layers, not after the output layer
- The model supports any Keras-compatible optimizer and loss function
- Preprocessing model integration enables unified training/inference pipelines

