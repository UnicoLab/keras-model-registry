# KMR Models Documentation

## SFNEBlock

### SFNEBlock

Slow-Fast Neural Engine Block for feature processing.

This model combines a slow network path and a fast processing path to extract
features. It uses a SlowNetwork to generate hyper-kernels, which are then used
by a HyperZZWOperator to compute context-dependent weights. These weights are
further processed by global and local convolutions before being combined.

Args:
    input_dim: Dimension of the input features.
    output_dim: Dimension of the output features. Default is same as input_dim.
    hidden_dim: Number of hidden units in the network. Default is 64.
    num_layers: Number of layers in the network. Default is 2.
    slow_network_layers: Number of layers in the slow network. Default is 3.
    slow_network_units: Number of units per layer in the slow network. Default is 128.
    preprocessing_model: Optional preprocessing model to apply before the main processing.
    name: Optional name for the model.
    
Input shape:
    2D tensor with shape: `(batch_size, input_dim)` or a dictionary with feature inputs
    
Output shape:
    2D tensor with shape: `(batch_size, output_dim)`
    
Example:
    ```python
    import keras
    from kmr.models import SFNEBlock
    
    # Create sample input data
    x = keras.random.normal((32, 16))  # 32 samples, 16 features
    
    # Create the model
    sfne = SFNEBlock(input_dim=16, output_dim=8)
    y = sfne(x)
    print("Output shape:", y.shape)  # (32, 8)
    ```

#### Methods

**call**

Forward pass of the model.

Args:
    inputs: Input tensor with shape (batch_size, input_dim) or a dictionary with feature inputs.
    training: Boolean indicating whether the model should behave in training mode.
        
Returns:
    Output tensor with shape (batch_size, output_dim).

**get_config**

Returns the config of the model.

Returns:
    Python dictionary containing the model configuration.

**from_config**

Creates a model from its configuration.

Args:
    config: Dictionary containing the model configuration.
    
Returns:
    A new instance of the model.

---

## TerminatorModel

### TerminatorModel

Terminator model for advanced feature processing.

This model stacks multiple SFNE blocks to process features in a hierarchical manner.
It's designed for complex tabular data modeling tasks where feature interactions
are important.

Args:
    input_dim: Dimension of the input features.
    context_dim: Dimension of the context features.
    output_dim: Dimension of the output.
    hidden_dim: Number of hidden units in the network. Default is 64.
    num_layers: Number of layers in the network. Default is 2.
    num_blocks: Number of SFNE blocks to stack. Default is 3.
    slow_network_layers: Number of layers in each slow network. Default is 3.
    slow_network_units: Number of units per layer in each slow network. Default is 128.
    preprocessing_model: Optional preprocessing model to apply before the main processing.
    name: Optional name for the model.
    
Input shape:
    List of 2D tensors with shapes: `[(batch_size, input_dim), (batch_size, context_dim)]`
    
Output shape:
    2D tensor with shape: `(batch_size, output_dim)`
    
Example:
    ```python
    import keras
    from kmr.models import TerminatorModel
    
    # Create sample input data
    x = keras.random.normal((32, 16))  # 32 samples, 16 features
    context = keras.random.normal((32, 8))  # 32 samples, 8 context features
    
    # Create the model
    terminator = TerminatorModel(input_dim=16, context_dim=8, output_dim=1)
    y = terminator([x, context])
    print("Output shape:", y.shape)  # (32, 1)
    ```

#### Methods

**call**

Forward pass of the model.

Args:
    inputs: List of input tensors [input_tensor, context_tensor] or a single input tensor.
    training: Boolean indicating whether the model should behave in training mode.
        
Returns:
    Output tensor with shape (batch_size, output_dim).

**get_config**

Returns the config of the model.

Returns:
    Python dictionary containing the model configuration.

**from_config**

Creates a model from its configuration.

Args:
    config: Dictionary containing the model configuration.
    
Returns:
    A new instance of the model.

---

## feed_forward

### BaseFeedForwardModel

Base feed forward neural network model.

This model implements a basic feed forward neural network with configurable
hidden layers, activations, and regularization options.

Example:
    ```python
    # Create a simple feed forward model
    model = BaseFeedForwardModel(
        feature_names=['feature1', 'feature2'],
        hidden_units=[64, 32],
        output_units=1
    )
    
    # Compile and train the model
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_dataset, epochs=10)
    ```

#### Methods

**build_model**

Build the model architecture.

**call**

Forward pass of the model.

Args:
    inputs: Dictionary of input tensors or a single tensor.
    training: Whether in training mode.
    
Returns:
    Model output tensor.

**get_config**

Get model configuration.

Returns:
    Dict containing model configuration.

**from_config**

Create model from configuration.

Args:
    config: Dict containing model configuration.

Returns:
    Instantiated model.

---
