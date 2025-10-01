# 🚀 BaseFeedForwardModel Guide

This guide demonstrates how to use the `BaseFeedForwardModel` from KMR for tabular data processing and model training. The `BaseFeedForwardModel` is designed to handle multiple input features and can optionally include a preprocessing model for feature engineering.

## ✨ Key Features

- **Multi-feature Input**: Handles multiple input features of different types (numeric, categorical, boolean)
- **Preprocessing Integration**: Supports custom preprocessing models for feature engineering
- **Flexible Architecture**: Configurable hidden layers, dropout, and activation functions
- **Keras Compatibility**: Full compatibility with Keras 3 and TensorFlow
- **Serialization**: Complete model saving and loading support

## 🏗️ Architecture

The `BaseFeedForwardModel` follows this architecture:

```
Input Features → Concatenation → Preprocessing Model → Hidden Layers → Output
```

1. **Input Layer**: Individual input layers for each feature
2. **Concatenation**: Combines all features into a single tensor
3. **Preprocessing Model** (optional): Custom preprocessing pipeline
4. **Hidden Layers**: Configurable dense layers with dropout
5. **Output Layer**: Final prediction layer

## 📊 Example Usage

### Basic Example

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras.optimizers import Adam
from keras.losses import MeanSquaredError

from kmr.models.feed_forward import BaseFeedForwardModel

# Create sample data
data = {
    'numeric_feature_1': np.random.normal(10, 3, 1000),
    'numeric_feature_2': np.random.exponential(2, 1000),
    'categorical_feature': np.random.choice(['A', 'B', 'C', 'D'], 1000),
    'boolean_feature': np.random.choice([True, False], 1000),
    'target': np.random.normal(5, 1, 1000)
}
df = pd.DataFrame(data)

# Define features
feature_names = ['numeric_feature_1', 'numeric_feature_2', 'categorical_feature', 'boolean_feature']

# Create preprocessing model
preprocessing_model = tf.keras.Sequential([
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.1)
])

# Build BaseFeedForwardModel
model = BaseFeedForwardModel(
    feature_names=feature_names,
    hidden_units=[64, 32, 16],
    output_units=1,
    dropout_rate=0.2,
    activation='relu',
    preprocessing_model=preprocessing_model,
    name='my_feed_forward_model'
)

# Compile and train
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=MeanSquaredError(),
    metrics=['mae']
)

# Prepare data
X_train = {name: df[name].values for name in feature_names}
y_train = df['target'].values

# Train
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Make predictions
predictions = model.predict(X_train)
```

### Advanced Example with Custom Preprocessing

```python
from keras import Model, layers

# Create custom preprocessing model
def create_advanced_preprocessing(input_dim: int) -> Model:
    inputs = layers.Input(shape=(input_dim,))
    
    # Feature engineering layers
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(32, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    outputs = layers.Dense(16, activation='relu')(x)
    
    return Model(inputs=inputs, outputs=outputs, name='advanced_preprocessing')

# Use advanced preprocessing
preprocessing_model = create_advanced_preprocessing(len(feature_names))

model = BaseFeedForwardModel(
    feature_names=feature_names,
    hidden_units=[128, 64, 32],
    output_units=1,
    dropout_rate=0.3,
    activation='relu',
    preprocessing_model=preprocessing_model
)
```

## 🔧 Configuration Options

### Model Parameters

- **`feature_names`**: List of feature names (required)
- **`hidden_units`**: List of hidden layer sizes (required)
- **`output_units`**: Number of output units (default: 1)
- **`dropout_rate`**: Dropout rate for hidden layers (default: 0.0)
- **`activation`**: Activation function (default: 'relu')
- **`preprocessing_model`**: Optional preprocessing model
- **`kernel_initializer`**: Weight initialization (default: 'glorot_uniform')
- **`bias_initializer`**: Bias initialization (default: 'zeros')

### Preprocessing Model Requirements

The preprocessing model should:
- Accept a single input tensor (concatenated features)
- Output a single tensor for the hidden layers
- Be a valid Keras Model

## 💾 Model Serialization

```python
# Save model
model.save('my_model')

# Load model
loaded_model = tf.keras.models.load_model('my_model')

# JSON serialization
config = model.get_config()
reconstructed_model = BaseFeedForwardModel.from_config(config)
```

## 🧪 Testing and Validation

The model includes comprehensive testing:

- **End-to-end training and prediction**
- **Model serialization and loading**
- **Error handling with invalid data**
- **Performance testing with large datasets**
- **Different architecture configurations**

## 📈 Best Practices

1. **Feature Engineering**: Use preprocessing models for complex feature transformations
2. **Regularization**: Apply appropriate dropout rates to prevent overfitting
3. **Data Validation**: Ensure input data matches expected feature names and types
4. **Model Saving**: Always save models after training for reproducibility
5. **Error Handling**: Validate input data before prediction

## 🔍 Troubleshooting

### Common Issues

1. **Missing Features**: Ensure all `feature_names` are present in input data
2. **Data Types**: Convert categorical features to appropriate numeric types
3. **Shape Mismatches**: Verify preprocessing model output shape matches hidden layer input
4. **Memory Issues**: Use appropriate batch sizes for large datasets

### Debug Tips

```python
# Check model architecture
model.summary()

# Verify input shapes
for name in feature_names:
    print(f"{name}: {X_train[name].shape}")

# Test preprocessing model separately
preprocessed = preprocessing_model(tf.concat([X_train[name] for name in feature_names], axis=1))
print(f"Preprocessed shape: {preprocessed.shape}")
```

## 🚀 Next Steps

- Explore the [KDP Integration Guide](kdp_integration_guide.md) for advanced preprocessing
- Check out the [API Reference](../api/models.md) for detailed documentation
- Visit the [Examples](../examples/README.md) for more use cases
