# KMR Layers Documentation

## AdvancedGraphFeature

### AdvancedGraphFeatureLayer

Advanced graph-based feature layer for tabular data.

This layer projects scalar features into an embedding space and then applies
multi-head self-attention to compute data-dependent dynamic adjacencies between
features. It learns edge attributes by considering both the raw embeddings and
their differences. Optionally, a hierarchical aggregation is applied, where
features are grouped via a learned soft-assignment and then re-expanded back to
the original feature space. A residual connection and layer normalization are
applied before the final projection back to the original feature space.

The layer is highly configurable, allowing for control over the embedding dimension,
number of attention heads, dropout rate, and hierarchical aggregation.

Notes:
    **When to Use This Layer:**
    - When working with tabular data where feature interactions are important
    - For complex feature engineering tasks where manual feature crosses are insufficient
    - When dealing with heterogeneous features that require dynamic, learned relationships
    - In scenarios where feature importance varies across different samples
    - When hierarchical feature relationships exist in your data

    **Best Practices:**
    - Start with a small embed_dim (e.g., 16 or 32) and increase if needed
    - Use num_heads=4 or 8 for most applications
    - Enable hierarchical=True when you have many features (>20) or known grouping structure
    - Set dropout_rate=0.1 or 0.2 for regularization during training
    - Use layer normalization (enabled by default) to stabilize training

    **Performance Considerations:**
    - Memory usage scales quadratically with the number of features
    - Consider using hierarchical mode for large feature sets to reduce complexity
    - The layer works best with normalized input features
    - For very large feature sets (>100), consider feature pre-selection

Args:
    embed_dim (int): Dimensionality of the projected feature embeddings. Determines the size
        of the learned feature representations.
    num_heads (int): Number of attention heads. Must divide embed_dim evenly. Each head
        learns different aspects of feature relationships.
    dropout_rate (float, optional): Dropout rate applied to attention weights during training.
        Helps prevent overfitting. Defaults to 0.0.
    hierarchical (bool, optional): Whether to apply hierarchical aggregation. If True, features
        are grouped into clusters, and aggregation is performed at the cluster level.
        Defaults to False.
    num_groups (int, optional): Number of groups to cluster features into when hierarchical is True.
        Must be provided if hierarchical is True. Controls the granularity of hierarchical
        aggregation.

Raises:
    ValueError: If embed_dim is not divisible by num_heads. Ensures that the embedding dimension
        can be evenly split across attention heads.
    ValueError: If hierarchical is True but num_groups is not provided. The number of groups
        must be specified when hierarchical aggregation is enabled.

Examples:

    **Basic Usage:**

    ```python
    import keras
    from kmr.layers import AdvancedGraphFeatureLayer

    # Dummy tabular data with 10 features for 32 samples.
    x = keras.random.normal((32, 10))
    # Create the advanced graph layer with an embedding dimension of 16 and 4 heads.
    layer = AdvancedGraphFeatureLayer(embed_dim=16, num_heads=4)
    y = layer(x, training=True)
    print("Output shape:", y.shape)  # Expected: (32, 10)
    ```

    **With Hierarchical Aggregation:**

    ```python
    import keras
    from kmr.layers import AdvancedGraphFeatureLayer

    # Dummy tabular data with 10 features for 32 samples.
    x = keras.random.normal((32, 10))
    # Create the advanced graph layer with hierarchical aggregation into 4 groups.
    layer = AdvancedGraphFeatureLayer(embed_dim=16, num_heads=4, hierarchical=True, num_groups=4)
    y = layer(x, training=True)
    print("Output shape:", y.shape)  # Expected: (32, 10)
    ```

    **Without Training:**

    ```python
    import keras
    from kmr.layers import AdvancedGraphFeatureLayer

    # Dummy tabular data with 10 features for 32 samples.
    x = keras.random.normal((32, 10))
    # Create the advanced graph layer with an embedding dimension of 16 and 4 heads.
    layer = AdvancedGraphFeatureLayer(embed_dim=16, num_heads=4)
    y = layer(x, training=False)
    print("Output shape:", y.shape)  # Expected: (32, 10)
    ```

#### Methods

**build**

No documentation available.

**call**

Args:
    inputs (KerasTensor): Input tensor of shape (batch, num_features).
    training (bool, optional): Whether the call is in training mode.

Returns:
    KerasTensor: Output tensor of shape (batch, num_features).

**get_config**

Returns the configuration of the layer.

This method is used to serialize the layer and restore it later.

Returns:
    dict: A dictionary containing the configuration of the layer.

**compute_output_shape**

Compute the output shape of the layer.

Args:
    input_shape: Shape tuple (batch_size, num_features)

Returns:
    Output shape tuple (batch_size, num_features)

---

## AdvancedNumericalEmbedding

### AdvancedNumericalEmbedding

Advanced numerical embedding layer for continuous features.

This layer embeds each continuous numerical feature into a higher-dimensional space by
combining two branches:

  1. Continuous Branch: Each feature is processed via a small MLP.
  2. Discrete Branch: Each feature is discretized into bins using learnable min/max boundaries
     and then an embedding is looked up for its bin.

A learnable gate combines the two branch outputs per feature and per embedding dimension.
Additionally, the continuous branch uses a residual connection and optional batch normalization
to improve training stability.

Args:
    embedding_dim (int): Output embedding dimension per feature.
    mlp_hidden_units (int): Hidden units for the continuous branch MLP.
    num_bins (int): Number of bins for discretization.
    init_min (float or list): Initial minimum values for discretization boundaries. If a scalar is
        provided, it is applied to all features.
    init_max (float or list): Initial maximum values for discretization boundaries.
    dropout_rate (float): Dropout rate applied to the continuous branch.
    use_batch_norm (bool): Whether to apply batch normalization to the continuous branch.
    name (str, optional): Name for the layer.

Input shape:
    Tensor with shape: `(batch_size, num_features)`

Output shape:
    Tensor with shape: `(batch_size, num_features, embedding_dim)` or
    `(batch_size, embedding_dim)` if num_features=1

Example:
    ```python
    import keras
    from kmr.layers import AdvancedNumericalEmbedding
    
    # Create sample input data
    x = keras.random.normal((32, 5))  # 32 samples, 5 features
    
    # Create the layer
    embedding = AdvancedNumericalEmbedding(
        embedding_dim=8,
        mlp_hidden_units=16,
        num_bins=10
    )
    y = embedding(x)
    print("Output shape:", y.shape)  # (32, 5, 8)
    ```

#### Methods

**build**

Builds the layer with the given input shape.

Args:
    input_shape: Tuple of integers defining the input shape.

**call**

Forward pass of the layer.

Args:
    inputs: Input tensor with shape (batch_size, num_features).
    training: Boolean indicating whether the layer should behave in
        training mode or inference mode.
        
Returns:
    Output tensor with shape (batch_size, num_features, embedding_dim) or
    (batch_size, embedding_dim) if num_features=1.

**compute_output_shape**

Compute the output shape of the layer.

Args:
    input_shape: Shape of the input tensor.
    
Returns:
    Shape of the output tensor.

**get_config**

Get layer configuration.

Returns:
    Dictionary containing the layer configuration.

---

## BoostingBlock

### BoostingBlock

A neural network layer that simulates gradient boosting behavior.

This layer implements a weak learner that computes a correction term via a configurable MLP
and adds a scaled version of this correction to the input. Stacking several such blocks
can mimic the iterative residual-correction process of gradient boosting.

The output is computed as:
    output = inputs + gamma * f(inputs)
where:
    - f is a configurable MLP (default: two-layer network)
    - gamma is a learnable or fixed scaling factor

Args:
    hidden_units: Number of units in the hidden layer(s). Can be an int for single hidden layer
        or a list of ints for multiple hidden layers. Default is 64.
    hidden_activation: Activation function for hidden layers. Default is 'relu'.
    output_activation: Activation function for the output layer. Default is None.
    gamma_trainable: Whether the scaling factor gamma is trainable. Default is True.
    gamma_initializer: Initializer for the gamma scaling factor. Default is 'ones'.
    use_bias: Whether to include bias terms in the dense layers. Default is True.
    kernel_initializer: Initializer for the dense layer kernels. Default is 'glorot_uniform'.
    bias_initializer: Initializer for the dense layer biases. Default is 'zeros'.
    dropout_rate: Optional dropout rate to apply after hidden layers. Default is None.
    name: Optional name for the layer.

Input shape:
    N-D tensor with shape: (batch_size, ..., input_dim)

Output shape:
    Same shape as input: (batch_size, ..., input_dim)

Example:
    ```python
    import tensorflow as tf
    from kmr.layers import BoostingBlock

    # Create sample input data
    x = tf.random.normal((32, 16))  # 32 samples, 16 features

    # Basic usage
    block = BoostingBlock(hidden_units=64)
    y = block(x)
    print("Output shape:", y.shape)  # (32, 16)

    # Advanced configuration
    block = BoostingBlock(
        hidden_units=[32, 16],  # Two hidden layers
        hidden_activation='selu',
        dropout_rate=0.1,
        gamma_trainable=False
    )
    y = block(x)
    ```

#### Methods

**build**

Builds the layer with the given input shape.

Args:
    input_shape: Tuple of integers defining the input shape.

**call**

Forward pass of the layer.

Args:
    inputs: Input tensor.
    training: Boolean indicating whether the layer should behave in
        training mode or inference mode. Only relevant when using dropout.

Returns:
    Output tensor of same shape as input.

**get_config**

Returns the config of the layer.

Returns:
    Python dictionary containing the layer configuration.

---

## BoostingEnsembleLayer

### BoostingEnsembleLayer

Ensemble layer of boosting blocks for tabular data.

This layer aggregates multiple boosting blocks (weak learners) in parallel. Each 
learner produces a correction to the input. A gating mechanism (via learnable weights)
then computes a weighted sum of the learners' outputs.

Args:
    num_learners: Number of boosting blocks in the ensemble. Default is 3.
    learner_units: Number of hidden units in each boosting block. Can be an int for 
        single hidden layer or a list of ints for multiple hidden layers. Default is 64.
    hidden_activation: Activation function for hidden layers in boosting blocks. Default is 'relu'.
    output_activation: Activation function for the output layer in boosting blocks. Default is None.
    gamma_trainable: Whether the scaling factor gamma in boosting blocks is trainable. Default is True.
    dropout_rate: Optional dropout rate to apply in boosting blocks. Default is None.
    name: Optional name for the layer.

Input shape:
    N-D tensor with shape: (batch_size, ..., input_dim)

Output shape:
    Same shape as input: (batch_size, ..., input_dim)

Example:
    ```python
    import keras
    from kmr.layers import BoostingEnsembleLayer

    # Create sample input data
    x = keras.random.normal((32, 16))  # 32 samples, 16 features

    # Basic usage
    ensemble = BoostingEnsembleLayer(num_learners=3, learner_units=64)
    y = ensemble(x)
    print("Ensemble output shape:", y.shape)  # (32, 16)

    # Advanced configuration
    ensemble = BoostingEnsembleLayer(
        num_learners=5,
        learner_units=[32, 16],  # Two hidden layers in each learner
        hidden_activation='selu',
        dropout_rate=0.1
    )
    y = ensemble(x)
    ```

#### Methods

**build**

Builds the layer with the given input shape.

Args:
    input_shape: Tuple of integers defining the input shape.

**call**

Forward pass of the layer.

Args:
    inputs: Input tensor.
    training: Boolean indicating whether the layer should behave in
        training mode or inference mode. Only relevant when using dropout.

Returns:
    Output tensor of same shape as input.

**get_config**

Returns the config of the layer.

Returns:
    Python dictionary containing the layer configuration.

---

## BusinessRulesLayer

### BusinessRulesLayer

Evaluates business-defined rules for anomaly detection.

This layer applies user-defined business rules to detect anomalies. Rules can be
defined for both numerical and categorical features.

For numerical features:
    - Comparison operators: '>' and '<'
    - Example: [(">", 0), ("<", 100)] for range validation

For categorical features:
    - Set operators: '==', 'in', '!=', 'not in'
    - Example: [("in", ["red", "green", "blue"])] for valid categories

Attributes:
    rules: List of rule tuples (operator, value).
    feature_type: Type of feature ('numerical' or 'categorical').

Example:
    ```python
    # Numerical rules
    layer = BusinessRulesLayer(rules=[(">", 0), ("<", 100)], feature_type="numerical")
    outputs = layer(tf.constant([[50.0], [-10.0]]))
    print(outputs['business_anomaly'])  # [[False], [True]]

    # Categorical rules
    layer = BusinessRulesLayer(
        rules=[("in", ["red", "green"])],
        feature_type="categorical"
    )
    outputs = layer(tf.constant([["red"], ["blue"]]))
    print(outputs['business_anomaly'])  # [[False], [True]]
    ```

#### Methods

**build**

Build the layer.

Args:
    input_shape: Shape of input tensor.

**compute_output_shape**

No documentation available.

**call**

Forward pass of the layer.

Args:
    inputs: Input tensor.
    training: Whether in training mode. Not used.

Returns:
    dictionary containing:
        - business_score: Violation score (0 = no violation)
        - business_proba: Probability of violation (0-100)
        - business_anomaly: Boolean mask of violations
        - business_reason: String tensor explaining violations
        - business_value: Original input values

**get_config**

Return the config of the layer.

Returns:
    Layer configuration dictionary.

---

## CastToFloat32Layer

### CastToFloat32Layer

Layer that casts input tensors to float32 data type.

This layer is useful for ensuring consistent data types in a model,
especially when working with mixed precision or when receiving inputs
of various data types.

Args:
    name: Optional name for the layer.
    
Input shape:
    Tensor of any shape and numeric data type.
    
Output shape:
    Same as input shape, but with float32 data type.
    
Example:
    ```python
    import keras
    import numpy as np
    from kmr.layers import CastToFloat32Layer
    
    # Create sample input data with int64 type
    x = keras.ops.convert_to_tensor(np.array([1, 2, 3], dtype=np.int64))
    
    # Apply casting layer
    cast_layer = CastToFloat32Layer()
    y = cast_layer(x)
    
    print(y.dtype)  # float32
    ```

#### Methods

**call**

Cast inputs to float32.

Args:
    inputs: Input tensor of any numeric data type.
    
Returns:
    Input tensor cast to float32.

**compute_output_shape**

Compute the output shape of the layer.

Args:
    input_shape: Shape of the input tensor.
    
Returns:
    Same shape as input.

---

## CategoricalAnomalyDetectionLayer

### CategoricalAnomalyDetectionLayer

Backend-agnostic anomaly detection for categorical features.

This layer detects anomalies in categorical features by checking if values belong to
a predefined set of valid categories. Values not in this set are considered anomalous.

The layer uses a Keras StringLookup or IntegerLookup layer internally to efficiently
map input values to indices, which are then used to determine if a value is valid.

Attributes:
    dtype: The data type of input values ('string' or 'int32').
    lookup: A Keras lookup layer for mapping values to indices.
    vocabulary: list of valid categorical values.

Example:
    ```python
    layer = CategoricalAnomalyDetectionLayer(dtype='string')
    layer.initialize_from_stats(vocabulary=['red', 'green', 'blue'])
    outputs = layer(tf.constant([['red'], ['purple']]))
    print(outputs['anomaly'])  # [[False], [True]]
    ```

#### Methods

**build**

Builds the layer.

Args:
    input_shape: Shape of input tensor (batch_size, feature_dim).

**dtype**

No documentation available.

**set_dtype**

Set the dtype and initialize the appropriate lookup layer.

**initialize_from_stats**

Initializes the layer with a vocabulary of valid values.

Args:
    vocabulary: list of valid categorical values.

**compute_output_shape**

No documentation available.

**call**

No documentation available.

**get_config**

No documentation available.

**from_config**

No documentation available.

---

## ColumnAttention

### ColumnAttention

Column attention mechanism to weight features dynamically.

This layer applies attention weights to each feature (column) in the input tensor.
The attention weights are computed using a two-layer neural network that takes
the input features and outputs attention weights for each feature.

Example:

    ```python
    import tensorflow as tf
    from kmr.layers import ColumnAttention
    
    # Create sample data
    batch_size = 32
    input_dim = 10
    inputs = tf.random.normal((batch_size, input_dim))
    
    # Apply column attention
    attention = ColumnAttention(input_dim=input_dim)
    weighted_outputs = attention(inputs)
    ```

#### Methods

**build**

Build the layer.

Args:
    input_shape: Shape of input tensor

**call**

Apply column attention.

Args:
    inputs: Input tensor of shape [batch_size, input_dim]

Returns:
    Attention weighted tensor of shape [batch_size, input_dim]

**get_config**

Get layer configuration.

Returns:
    Layer configuration dictionary

**from_config**

Create layer from configuration.

Args:
    config: Layer configuration dictionary

Returns:
    ColumnAttention instance

---

## DateEncodingLayer

### DateEncodingLayer

Layer for encoding date components into cyclical features.

This layer takes date components (year, month, day, day of week) and encodes them
into cyclical features using sine and cosine transformations. The year is normalized
to a range between 0 and 1 based on min_year and max_year.

Args:
    min_year: Minimum year for normalization (default: 1900)
    max_year: Maximum year for normalization (default: 2100)
    **kwargs: Additional layer arguments

Input shape:
    Tensor with shape: `(..., 4)` containing [year, month, day, day_of_week]

Output shape:
    Tensor with shape: `(..., 8)` containing cyclical encodings:
    [year_sin, year_cos, month_sin, month_cos, day_sin, day_cos, dow_sin, dow_cos]

#### Methods

**call**

Apply the layer to the inputs.

Args:
    inputs: Tensor with shape (..., 4) containing [year, month, day, day_of_week]

Returns:
    Tensor with shape (..., 8) containing cyclical encodings

**compute_output_shape**

Compute the output shape of the layer.

Args:
    input_shape: Shape of the input tensor

Returns:
    Output shape

**get_config**

Return the configuration of the layer.

Returns:
    Dictionary containing the layer configuration

---

## DateParsingLayer

### DateParsingLayer

Layer for parsing date strings into numerical components.

This layer takes date strings in a specified format and returns a tensor
containing the year, month, day of the month, and day of the week.

Args:
    date_format: Format of the date strings. Currently supports 'YYYY-MM-DD'
        and 'YYYY/MM/DD'. Default is 'YYYY-MM-DD'.
    **kwargs: Additional keyword arguments to pass to the base layer.

Input shape:
    String tensor of any shape.

Output shape:
    Same as input shape with an additional dimension of size 4 appended.
    For example, if input shape is [batch_size], output shape will be
    [batch_size, 4].

#### Methods

**call**

Parse date strings into numerical components.

Args:
    inputs: String tensor containing date strings.

Returns:
    Tensor with shape [..., 4] containing [year, month, day_of_month, day_of_week].

**compute_output_shape**

Compute the output shape of the layer.

Args:
    input_shape: Shape of the input tensor.

Returns:
    Shape of the output tensor.

**get_config**

Return the configuration of the layer.

Returns:
    Dictionary containing the configuration of the layer.

---

## DifferentiableTabularPreprocessor

### DifferentiableTabularPreprocessor

A differentiable preprocessing layer for numeric tabular data.

This layer:
  - Replaces missing values (NaNs) with a learnable imputation vector.
  - Applies a learned affine transformation (scaling and shifting) to each feature.
  
The idea is to integrate preprocessing into the model so that the optimal 
imputation and normalization parameters are learned end-to-end.

Args:
    num_features: Number of numeric features in the input.
    name: Optional name for the layer.

Input shape:
    2D tensor with shape: `(batch_size, num_features)`

Output shape:
    2D tensor with shape: `(batch_size, num_features)` (same as input)

Example:
    ```python
    import keras
    import numpy as np
    from kmr.layers import DifferentiableTabularPreprocessor
    
    # Suppose we have tabular data with 5 numeric features
    x = keras.ops.convert_to_tensor([
        [1.0, np.nan, 3.0, 4.0, 5.0],
        [2.0, 2.0, np.nan, 4.0, 5.0]
    ], dtype="float32")
    
    preproc = DifferentiableTabularPreprocessor(num_features=5)
    y = preproc(x)
    print(y)
    ```

#### Methods

**build**

Builds the layer with the given input shape.

Args:
    input_shape: Tuple of integers defining the input shape.

**call**

Forward pass of the layer.

Args:
    inputs: Input tensor with shape (batch_size, num_features).
    training: Boolean indicating whether the layer should behave in
        training mode or inference mode.

Returns:
    Output tensor with the same shape as input.

**get_config**

Returns the config of the layer.

Returns:
    Python dictionary containing the layer configuration.

---

## DifferentialPreprocessingLayer

### DifferentialPreprocessingLayer

Differentiable preprocessing layer for numeric tabular data with multiple candidate transformations.

This layer:
  1. Imputes missing values using a learnable imputation vector.
  2. Applies several candidate transformations:
     - Identity (pass-through)
     - Affine transformation (learnable scaling and bias)
     - Nonlinear transformation via a small MLP
     - Log transformation (using a softplus to ensure positivity)
  3. Learns softmax combination weights to aggregate the candidates.

The entire preprocessing pipeline is differentiable, so the network learns the optimal
imputation and transformation jointly with downstream tasks.

Args:
    num_features: Number of numeric features in the input.
    mlp_hidden_units: Number of hidden units in the nonlinear branch. Default is 4.
    name: Optional name for the layer.

Input shape:
    2D tensor with shape: `(batch_size, num_features)`

Output shape:
    2D tensor with shape: `(batch_size, num_features)` (same as input)

Example:
    ```python
    import keras
    import numpy as np
    from kmr.layers import DifferentialPreprocessingLayer

    # Create dummy data: 6 samples, 4 features (with some missing values)
    x = keras.ops.convert_to_tensor([
        [1.0, 2.0, float('nan'), 4.0],
        [2.0, float('nan'), 3.0, 4.0],
        [float('nan'), 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, float('nan')],
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
    ], dtype="float32")
    
    # Instantiate the layer for 4 features.
    preproc_layer = DifferentialPreprocessingLayer(num_features=4, mlp_hidden_units=8)
    y = preproc_layer(x)
    print(y)
    ```

#### Methods

**build**

Builds the layer with the given input shape.

Args:
    input_shape: Tuple of integers defining the input shape.

**call**

Forward pass of the layer.

Args:
    inputs: Input tensor with shape (batch_size, num_features).
    training: Boolean indicating whether the layer should behave in
        training mode or inference mode.

Returns:
    Output tensor with the same shape as input.

**get_config**

Returns the config of the layer.

Returns:
    Python dictionary containing the layer configuration.

---

## DifferentialPreprocssing

### DifferentialPreprocssingLayer

Differentiable preprocessing layer for numeric tabular data with multiple candidate transformations.

This layer provides an end-to-end differentiable preprocessing pipeline for numeric
features, automatically learning the optimal combination of different transformations
for each feature. It handles missing values and applies multiple transformations
in parallel, learning their relative importance through trainable weights.

The layer performs three main operations:
  1. Imputes missing values using a learnable imputation vector
  2. Applies several candidate transformations in parallel:
     - Identity (pass-through for linear relationships)
     - Affine transformation (learnable scaling and bias for feature normalization)
     - Nonlinear transformation via MLP (for complex nonlinear patterns)
     - Log transformation (for handling skewed distributions)
  3. Learns softmax combination weights to aggregate the transformations

The entire pipeline is differentiable and trained jointly with the downstream task,
allowing the network to automatically discover the most effective preprocessing
strategy for each feature.

Notes:
    - Best suited for numeric tabular data with potential missing values
    - Particularly effective when different features require different transformations
    - Useful when the optimal preprocessing strategy is unclear or data-dependent
    - Can replace manual feature engineering in many cases
    - Computational cost scales linearly with the number of features
    - Memory usage is ~4x the input size due to parallel transformations

Recommendations:
    - Start with a small number of hidden units (4-8) in the MLP branch
    - Monitor the learned combination weights to understand which transformations
      are most effective for your data
    - Consider using this layer early in the model, right after the input
    - For very large feature sets, consider applying to subsets of related features
    - If all features are known to require the same transformation, use a simpler
      preprocessing approach

Example:
    ```python
    import keras
    import numpy as np
    from kmr.layers import DifferentialPreprocssingLayer

    # Create sample data with missing values
    x = np.array([
        [1.0, 2.0, float('nan'), 4.0],  # Some missing values
        [2.0, float('nan'), 3.0, 4.0],
        [float('nan'), 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, float('nan')],
        [1.0, 2.0, 3.0, 4.0],           # Complete samples
        [2.0, 3.0, 4.0, 5.0],
    ])

    # Create a model with differential preprocessing
    inputs = keras.Input(shape=(4,))
    preprocessed = DifferentialPreprocssingLayer(
        num_features=4,
        mlp_hidden_units=8
    )(inputs)
    outputs = keras.layers.Dense(1)(preprocessed)
    model = keras.Model(inputs, outputs)

    # The preprocessing layer will automatically learn the best
    # transformation for each feature during model training
    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs=10)
    ```

#### Methods

**build**

Build the layer by creating trainable weights and sublayers.

This method is called automatically when the layer is first used. It creates
all the trainable weights and sublayers needed for the preprocessing pipeline.

Args:
    input_shape: Shape tuple of the input tensor, expected to be
        (batch_size, num_features) where num_features matches the value
        provided in __init__.

Raises:
    ValueError: If the last dimension of input_shape does not match
        num_features specified during initialization. This ensures that
        the layer's weights are compatible with the input data.

Notes:
    The layer creates the following components:
    1. Imputation vector for handling missing values
    2. Affine transformation parameters (gamma for scaling, beta for bias)
    3. MLP layers for nonlinear transformation
    4. Combination weights for aggregating transformations

    All weights are trainable by default and will be updated during
    model training to optimize the preprocessing for the specific task.

**call**

Forward pass of the layer, applying the preprocessing pipeline.

This method implements the core preprocessing logic, applying multiple
transformations in parallel and combining them optimally.

Args:
    inputs: Input tensor of shape (batch_size, num_features) containing
        the raw feature values. Can contain NaN values, which will be
        imputed using learned values.
    training: Boolean indicating whether the layer is in training mode.
        Currently not used directly but passed to sublayers for consistency
        with Keras API.

Returns:
    Preprocessed tensor of shape (batch_size, num_features) containing
    the transformed features. The output is guaranteed to be finite and
    contain no NaN values.

Notes:
    The preprocessing pipeline consists of these steps:
    1. Missing value imputation using learned values
    2. Parallel application of transformations:
       - Identity: preserves linear relationships
       - Affine: learns feature-specific scaling and bias
       - MLP: captures complex nonlinear patterns
       - Log: handles skewed distributions
    3. Weighted combination using learned importance weights

    The importance weights are normalized using softmax, ensuring
    they sum to 1 for each feature. This allows the layer to
    smoothly transition between different transformations during
    training.

Examples:
    ```python
    # Basic usage
    layer = DifferentialPreprocssingLayer(num_features=3)
    x = np.array([[1.0, np.nan, 3.0],
                 [4.0, 5.0, np.nan]])
    y = layer(x)
    # y will have shape (2, 3) with no NaN values

    # Access transformation weights after training
    weights = keras.backend.eval(layer.alpha)
    # weights shows relative importance of each transformation
    ```

**get_config**

Return layer configuration for serialization.

This method is used by Keras for layer serialization. It returns a dictionary
containing all configuration needed to reconstruct the layer.

Returns:
    Dictionary containing the layer configuration with the following keys:
    - num_features: Number of features the layer was initialized with
    - mlp_hidden_units: Number of hidden units in the MLP branch
    Plus all base layer configuration from parent classes.

Example:
    ```python
    layer = DifferentialPreprocssingLayer(num_features=4, mlp_hidden_units=8)
    config = layer.get_config()
    new_layer = DifferentialPreprocssingLayer.from_config(config)
    # new_layer will be identical to the original layer
    ```

---

## DistributionAwareEncoder

### DistributionAwareEncoder

Layer that automatically detects and encodes data based on its distribution.

This layer first detects the distribution type of the input data and then applies
appropriate transformations and encodings. It builds upon the DistributionTransformLayer
but adds more sophisticated distribution detection and specialized encoding for
different distribution types.

Args:
    embedding_dim: Dimension of the output embedding. If None, the output will have
        the same dimension as the input. Default is None.
    auto_detect: Whether to automatically detect the distribution type. If False,
        the layer will use the specified distribution_type. Default is True.
    distribution_type: The distribution type to use if auto_detect is False.
        Options are "normal", "exponential", "lognormal", "uniform", "beta",
        "bimodal", "heavy_tailed", "mixed", "bounded", "unknown". Default is "unknown".
    transform_type: The transformation type to use. If "auto", the layer will
        automatically select the best transformation based on the detected distribution.
        See DistributionTransformLayer for available options. Default is "auto".
    add_distribution_embedding: Whether to add a learned embedding of the distribution
        type to the output. Default is False.
    name: Optional name for the layer.
    
Input shape:
    N-D tensor with shape: `(batch_size, ..., features)`.
    
Output shape:
    If embedding_dim is None, same shape as input: `(batch_size, ..., features)`.
    If embedding_dim is specified: `(batch_size, ..., embedding_dim)`.
    If add_distribution_embedding is True, the output will have an additional
    dimension for the distribution embedding.
    
Example:
    ```python
    import keras
    import numpy as np
    from kmr.layers import DistributionAwareEncoder
    
    # Create sample input data with different distributions
    # Normal distribution
    normal_data = keras.ops.convert_to_tensor(
        np.random.normal(0, 1, (100, 10)), dtype="float32"
    )
    
    # Exponential distribution
    exp_data = keras.ops.convert_to_tensor(
        np.random.exponential(1, (100, 10)), dtype="float32"
    )
    
    # Create the encoder
    encoder = DistributionAwareEncoder(embedding_dim=16, add_distribution_embedding=True)
    
    # Apply to normal data
    normal_encoded = encoder(normal_data)
    print("Normal encoded shape:", normal_encoded.shape)  # (100, 16)
    
    # Apply to exponential data
    exp_encoded = encoder(exp_data)
    print("Exponential encoded shape:", exp_encoded.shape)  # (100, 16)
    ```

#### Methods

**build**

Builds the layer with the given input shape.

Args:
    input_shape: Tuple of integers defining the input shape.

**call**

Apply distribution-aware encoding to the inputs.

Args:
    inputs: Input tensor
    training: Boolean indicating whether the layer should behave in
        training mode or inference mode
        
Returns:
    Encoded tensor

**get_config**

Returns the config of the layer.

Returns:
    Python dictionary containing the layer configuration.

---

## DistributionTransformLayer

### DistributionTransformLayer

Layer for transforming data distributions to improve anomaly detection.

This layer applies various transformations to make data more normally distributed
or to handle specific distribution types better. Supported transformations include
log, square root, Box-Cox, Yeo-Johnson, arcsinh, cube-root, logit, quantile, 
robust-scale, and min-max.

When transform_type is set to 'auto', the layer automatically selects the most
appropriate transformation based on the data characteristics during training.

Args:
    transform_type: Type of transformation to apply. Options are 'none', 'log', 'sqrt', 
        'box-cox', 'yeo-johnson', 'arcsinh', 'cube-root', 'logit', 'quantile', 
        'robust-scale', 'min-max', or 'auto'. Default is 'none'.
    lambda_param: Parameter for parameterized transformations like Box-Cox and Yeo-Johnson.
        Default is 0.0.
    epsilon: Small value added to prevent numerical issues like log(0). Default is 1e-10.
    min_value: Minimum value for min-max scaling. Default is 0.0.
    max_value: Maximum value for min-max scaling. Default is 1.0.
    clip_values: Whether to clip values to the specified range in min-max scaling. Default is True.
    auto_candidates: list of transformation types to consider when transform_type is 'auto'.
        If None, all available transformations will be considered. Default is None.
    name: Optional name for the layer.

Input shape:
    N-D tensor with shape: (batch_size, ..., features)

Output shape:
    Same shape as input: (batch_size, ..., features)

Example:
    ```python
    import keras
    import numpy as np
    from kmr.layers import DistributionTransformLayer

    # Create sample input data with skewed distribution
    x = keras.random.exponential((32, 10))  # 32 samples, 10 features

    # Apply log transformation
    log_transform = DistributionTransformLayer(transform_type="log")
    y = log_transform(x)
    print("Transformed output shape:", y.shape)  # (32, 10)

    # Apply Box-Cox transformation with lambda=0.5
    box_cox = DistributionTransformLayer(transform_type="box-cox", lambda_param=0.5)
    z = box_cox(x)
    
    # Apply arcsinh transformation (handles both positive and negative values)
    arcsinh_transform = DistributionTransformLayer(transform_type="arcsinh")
    a = arcsinh_transform(x)
    
    # Apply min-max scaling to range [0, 1]
    min_max = DistributionTransformLayer(transform_type="min-max", min_value=0.0, max_value=1.0)
    b = min_max(x)
    
    # Use automatic transformation selection
    auto_transform = DistributionTransformLayer(transform_type="auto")
    c = auto_transform(x)  # Will select the best transformation during training
    ```

#### Methods

**build**

Builds the layer with the given input shape.

Args:
    input_shape: tuple of integers defining the input shape.

**call**

Apply the selected transformation to the inputs.

Args:
    inputs: Input tensor
    training: Boolean indicating whether the layer should behave in
        training mode or inference mode

Returns:
    Transformed tensor with the same shape as input

**get_config**

Returns the config of the layer.

Returns:
    Python dictionary containing the layer configuration.

---

## FeatureCutout

### FeatureCutout

Feature cutout regularization layer.

This layer randomly masks out (sets to zero) a specified fraction of features
during training to improve model robustness and prevent overfitting. During
inference, all features are kept intact.

Example:
    ```python
    from keras import random
    from kmr.layers import FeatureCutout
    
    # Create sample data
    batch_size = 32
    feature_dim = 10
    inputs = random.normal((batch_size, feature_dim))
    
    # Apply feature cutout
    cutout = FeatureCutout(cutout_prob=0.2)
    masked_outputs = cutout(inputs, training=True)
    ```

#### Methods

**build**

Build the layer.

Args:
    input_shape: Shape of input tensor

**call**

Apply feature cutout.

Args:
    inputs: Input tensor of shape [batch_size, feature_dim]
    training: Whether in training mode

Returns:
    Masked tensor of shape [batch_size, feature_dim]

**compute_output_shape**

Compute output shape.

Args:
    input_shape: Input shape tuple

Returns:
    Output shape tuple

**get_config**

Get layer configuration.

Returns:
    Layer configuration dictionary

**from_config**

Create layer from configuration.

Args:
    config: Layer configuration dictionary

Returns:
    FeatureCutout instance

---

## GatedFeatureFusion

### GatedFeatureFusion

Gated feature fusion layer for combining two feature representations.

This layer takes two inputs (e.g., numerical features and their embeddings) and fuses
them using a learned gate to balance their contributions. The gate is computed using
a dense layer with sigmoid activation, applied to the concatenation of both inputs.

Args:
    activation: Activation function to use for the gate. Default is 'sigmoid'.
    name: Optional name for the layer.

Input shape:
    A list of 2 tensors with shape: `[(batch_size, ..., features), (batch_size, ..., features)]`
    Both inputs must have the same shape.

Output shape:
    Tensor with shape: `(batch_size, ..., features)`, same as each input.

Example:
    ```python
    import keras
    from kmr.layers import GatedFeatureFusion

    # Two representations for the same 10 features
    feat1 = keras.random.normal((32, 10))
    feat2 = keras.random.normal((32, 10))
    
    fusion_layer = GatedFeatureFusion()
    fused = fusion_layer([feat1, feat2])
    print("Fused output shape:", fused.shape)  # Expected: (32, 10)
    ```

#### Methods

**build**

Builds the layer with the given input shape.

Args:
    input_shape: list of two input shapes, each a tuple of integers.

**call**

Forward pass of the layer.

Args:
    inputs: list of two input tensors to be fused.
    training: Boolean indicating whether the layer should behave in
        training mode or inference mode.

Returns:
    Fused output tensor with the same shape as each input.

**get_config**

Returns the config of the layer.

Returns:
    Python dictionary containing the layer configuration.

---

## GatedFeaturesSelection

### GatedFeatureSelection

Gated feature selection layer with residual connection.

This layer implements a learnable feature selection mechanism using a gating network.
Each feature is assigned a dynamic importance weight between 0 and 1 through a multi-layer
gating network. The gating network includes batch normalization and ReLU activations
for stable training. A small residual connection (0.1) is added to maintain gradient flow.

The layer is particularly useful for:
1. Dynamic feature importance learning
2. Feature selection in time-series data
3. Attention-like mechanisms for tabular data
4. Reducing noise in input features

Example:
```python
import numpy as np
from keras import layers, Model
from kmr.layers import GatedFeatureSelection

# Create sample input data
input_dim = 20
x = np.random.normal(size=(100, input_dim))

# Build model with gated feature selection
inputs = layers.Input(shape=(input_dim,))
x = GatedFeatureSelection(input_dim=input_dim, reduction_ratio=4)(inputs)
outputs = layers.Dense(1)(x)
model = Model(inputs=inputs, outputs=outputs)

# The layer will learn which features are most important
# and dynamically adjust their contribution to the output
```

Args:
    input_dim: Dimension of the input features
    reduction_ratio: Ratio to reduce the hidden dimension of the gating network.
        A higher ratio means fewer parameters but potentially less expressive gates.
        Default is 4, meaning the hidden dimension will be input_dim // 4.

#### Methods

**build**

Build the gating network.

Creates a multi-layer gating network with batch normalization and ReLU
activations. The network architecture is:
1. Dense(hidden_dim) -> ReLU -> BatchNorm
2. Dense(hidden_dim) -> ReLU -> BatchNorm
3. Dense(input_dim) -> Sigmoid

Args:
    input_shape: Shape of input tensor, expected to be (..., input_dim)

Raises:
    ValueError: If the last dimension of input_shape doesn't match input_dim

**call**

Apply gated feature selection with residual connection.

The layer computes feature importance gates using the gating network
and applies them to the input features. A small residual connection (0.1)
is added to maintain gradient flow and prevent features from being
completely masked.

Args:
    inputs: Input tensor of shape (..., input_dim)

Returns:
    Tensor of same shape as input with gated features.
    The output is computed as: inputs * gates + 0.1 * inputs

**get_config**

Get layer configuration for serialization.

Returns:
    Dictionary containing the layer configuration:
    - input_dim: Dimension of input features
    - reduction_ratio: Ratio for hidden dimension reduction

**from_config**

Create layer from configuration.

Args:
    config: Layer configuration dictionary

Returns:
    GatedFeatureSelection instance

---

## GatedLinearUnit

### GatedLinearUnit

GatedLinearUnit is a custom Keras layer that implements a gated linear unit.

This layer applies a dense linear transformation to the input tensor and multiplies the result with the output
of a dense sigmoid transformation. The result is a tensor where the input data is filtered based on the learned
weights and biases of the layer.

Args:
    units (int): Positive integer, dimensionality of the output space.
    name (str, optional): Name for the layer.

Input shape:
    Tensor with shape: `(batch_size, ..., input_dim)`

Output shape:
    Tensor with shape: `(batch_size, ..., units)`

Example:
    ```python
    import keras
    from kmr.layers import GatedLinearUnit
    
    # Create sample input data
    x = keras.random.normal((32, 16))  # 32 samples, 16 features
    
    # Create the layer
    glu = GatedLinearUnit(units=8)
    y = glu(x)
    print("Output shape:", y.shape)  # (32, 8)
    ```

#### Methods

**build**

Builds the layer with the given input shape.

Args:
    input_shape: Tuple of integers defining the input shape.

**call**

Forward pass of the layer.

Args:
    inputs: Input tensor.
    
Returns:
    Output tensor after applying gated linear transformation.

**get_config**

Get layer configuration.

Returns:
    Dictionary containing the layer configuration.

---

## GatedResidualNetwork

### GatedResidualNetwork

GatedResidualNetwork is a custom Keras layer that implements a gated residual network.

This layer applies a series of transformations to the input tensor and combines the result with the input
using a residual connection. The transformations include a dense layer with ELU activation, a dense linear
layer, a dropout layer, a gated linear unit layer, layer normalization, and a final dense layer.

Args:
    units (int): Positive integer, dimensionality of the output space.
    dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.2.
    name (str, optional): Name for the layer.

Input shape:
    Tensor with shape: `(batch_size, ..., input_dim)`

Output shape:
    Tensor with shape: `(batch_size, ..., units)`

Example:
    ```python
    import keras
    from kmr.layers import GatedResidualNetwork
    
    # Create sample input data
    x = keras.random.normal((32, 16))  # 32 samples, 16 features
    
    # Create the layer
    grn = GatedResidualNetwork(units=16, dropout_rate=0.2)
    y = grn(x)
    print("Output shape:", y.shape)  # (32, 16)
    ```

#### Methods

**build**

Builds the layer with the given input shape.

Args:
    input_shape: Tuple of integers defining the input shape.

**call**

Forward pass of the layer.

Args:
    inputs: Input tensor.
    training: Boolean indicating whether the layer should behave in
        training mode or inference mode.
        
Returns:
    Output tensor after applying gated residual transformations.

**get_config**

Get layer configuration.

Returns:
    Dictionary containing the layer configuration.

---

## GraphFeatureAggregation

### GraphFeatureAggregation

Graph-based feature aggregation layer with self-attention for tabular data.

This layer treats each input feature as a node and projects it into an embedding space.
It then computes pairwise attention scores between features and aggregates feature
information based on these scores. Finally, it projects the aggregated features back
to the original feature space and adds a residual connection.

The process involves:
  1. Projecting each scalar feature to an embedding (shape: [batch, num_features, embed_dim]).
  2. Computing pairwise concatenated embeddings and scoring them via a learnable attention vector.
  3. Normalizing the scores with softmax to yield a dynamic adjacency (attention) matrix.
  4. Aggregating neighboring features via weighted sum.
  5. Projecting back to a vector of original dimension, then adding a residual connection.

Args:
    embed_dim: Dimensionality of the projected feature embeddings. Default is 8.
    dropout_rate: Dropout rate to apply on attention weights. Default is 0.0.
    leaky_relu_alpha: Alpha parameter for the LeakyReLU activation. Default is 0.2.
    name: Optional name for the layer.

Input shape:
    2D tensor with shape: `(batch_size, num_features)`

Output shape:
    2D tensor with shape: `(batch_size, num_features)` (same as input)

Example:
    ```python
    import keras
    from kmr.layers import GraphFeatureAggregation

    # Tabular data with 10 features
    x = keras.random.normal((32, 10))
    
    # Create the layer with an embedding dimension of 8 and dropout rate of 0.1
    graph_layer = GraphFeatureAggregation(embed_dim=8, dropout_rate=0.1)
    y = graph_layer(x, training=True)
    print("Output shape:", y.shape)  # Expected: (32, 10)
    ```

#### Methods

**build**

Builds the layer with the given input shape.

Args:
    input_shape: Tuple of integers defining the input shape.

**call**

Forward pass of the layer.

Args:
    inputs: Input tensor with shape (batch_size, num_features).
    training: Boolean indicating whether the layer should behave in
        training mode or inference mode. Only relevant when using dropout.

Returns:
    Output tensor with the same shape as input.

**get_config**

Returns the config of the layer.

Returns:
    Python dictionary containing the layer configuration.

---

## HyperZZWOperator

### HyperZZWOperator

A layer that computes context-dependent weights by multiplying inputs with hyper-kernels.

This layer takes two inputs: the original input tensor and a context tensor.
It generates hyper-kernels from the context and performs a context-dependent transformation
of the input.

Args:
    input_dim: Dimension of the input features.
    context_dim: Optional dimension of the context features. If not provided, it will be inferred.
    name: Optional name for the layer.
    
Input:
    A list of two tensors:
    - inputs[0]: Input tensor with shape (batch_size, input_dim).
    - inputs[1]: Context tensor with shape (batch_size, context_dim).
    
Output shape:
    2D tensor with shape: `(batch_size, input_dim)` (same as input)
    
Example:
    ```python
    import keras
    from kmr.layers import HyperZZWOperator
    
    # Create sample input data
    inputs = keras.random.normal((32, 16))  # 32 samples, 16 features
    context = keras.random.normal((32, 8))  # 32 samples, 8 context features
    
    # Create the layer
    zzw_op = HyperZZWOperator(input_dim=16, context_dim=8)
    context_weights = zzw_op([inputs, context])
    print("Output shape:", context_weights.shape)  # (32, 16)
    ```

#### Methods

**build**

Builds the layer with the given input shape.

Args:
    input_shape: List of tuples of integers defining the input shapes.
        input_shape[0] is the shape of the input tensor.
        input_shape[1] is the shape of the context tensor.

**call**

Forward pass of the layer.

Args:
    inputs: A list of two tensors:
        inputs[0]: Input tensor with shape (batch_size, input_dim).
        inputs[1]: Context tensor with shape (batch_size, context_dim).
        
Returns:
    Context-dependent weights tensor with the same shape as input.

**get_config**

Returns the config of the layer.

Returns:
    Python dictionary containing the layer configuration.

---

## InterpretableMultiHeadAttention

### InterpretableMultiHeadAttention

Interpretable Multi-Head Attention layer.

This layer wraps Keras MultiHeadAttention and stores the attention scores
for interpretability purposes. The attention scores can be accessed via
the `attention_scores` attribute after calling the layer.

Args:
    d_model: Size of each attention head for query, key, value.
    n_head: Number of attention heads.
    dropout_rate: Dropout probability. Default: 0.1.
    **kwargs: Additional arguments passed to MultiHeadAttention.
        Supported arguments:
        - value_dim: Size of each attention head for value.
        - use_bias: Whether to use bias. Default: True.
        - output_shape: Expected output shape. Default: None.
        - attention_axes: Axes for attention. Default: None.
        - kernel_initializer: Initializer for kernels. Default: 'glorot_uniform'.
        - bias_initializer: Initializer for biases. Default: 'zeros'.
        - kernel_regularizer: Regularizer for kernels. Default: None.
        - bias_regularizer: Regularizer for biases. Default: None.
        - activity_regularizer: Regularizer for activity. Default: None.
        - kernel_constraint: Constraint for kernels. Default: None.
        - bias_constraint: Constraint for biases. Default: None.
        - seed: Random seed for dropout. Default: None.

Call Args:
    query: Query tensor of shape `(B, S, E)` where B is batch size,
        S is sequence length, and E is the feature dimension.
    key: Key tensor of shape `(B, S, E)`.
    value: Value tensor of shape `(B, S, E)`.
    training: Python boolean indicating whether the layer should behave in
        training mode (applying dropout) or in inference mode (no dropout).

Returns:
    output: Attention output of shape `(B, S, E)`.

Example:
    ```python
    d_model = 64
    n_head = 4
    seq_len = 10
    batch_size = 32

    layer = InterpretableMultiHeadAttention(
        d_model=d_model,
        n_head=n_head,
        kernel_initializer='he_normal',
        use_bias=False
    )
    query = tf.random.normal((batch_size, seq_len, d_model))
    output = layer(query, query, query)
    attention_scores = layer.attention_scores  # Access attention weights
    ```

#### Methods

**call**

Forward pass of the layer.

Args:
    query: Query tensor of shape (B, S, E).
    key: Key tensor of shape (B, S, E).
    value: Value tensor of shape (B, S, E).
    training: Whether the model is in training mode.

Returns:
    Attention output tensor of shape (B, S, E).

**get_config**

Return the config dictionary for serialization.

**from_config**

Create layer from configuration.

Args:
    config: Layer configuration dictionary

Returns:
    Layer instance

---

## MultiHeadGraphFeaturePreprocessor

### MultiHeadGraphFeaturePreprocessor

Multi-head graph-based feature preprocessor for tabular data.

This layer treats each feature as a node and applies multi-head self-attention
to capture and aggregate complex interactions among features. The process is:

1. Project each scalar input into an embedding of dimension `embed_dim`.
2. Split the embedding into `num_heads` heads.
3. For each head, compute queries, keys, and values and calculate scaled dot-product
   attention across the feature dimension.
4. Concatenate the head outputs, project back to the original feature dimension,
   and add a residual connection.

This mechanism allows the network to learn multiple relational views among features,
which can significantly boost performance on tabular data.

Args:
    embed_dim: Dimension of the feature embeddings. Default is 16.
    num_heads: Number of attention heads. Default is 4.
    dropout_rate: Dropout rate applied to attention weights. Default is 0.0.
    name: Optional name for the layer.

Input shape:
    2D tensor with shape: `(batch_size, num_features)`
    
Output shape:
    2D tensor with shape: `(batch_size, num_features)` (same as input)

Example:
    ```python
    import keras
    from kmr.layers import MultiHeadGraphFeaturePreprocessor
    
    # Tabular data with 10 features
    x = keras.random.normal((32, 10))
    
    # Create the layer with 16-dim embeddings and 4 attention heads
    graph_preproc = MultiHeadGraphFeaturePreprocessor(embed_dim=16, num_heads=4)
    y = graph_preproc(x, training=True)
    print("Output shape:", y.shape)  # Expected: (32, 10)
    ```

#### Methods

**build**

Build the layer.

Args:
    input_shape: Shape of the input tensor.

**split_heads**

Split the last dimension into (num_heads, depth) and transpose.

Args:
    x: Input tensor with shape (batch_size, num_features, embed_dim).
    batch_size: Batch size tensor.
    
Returns:
    Tensor with shape (batch_size, num_heads, num_features, depth).

**call**

Forward pass of the layer.

Args:
    inputs: Input tensor with shape (batch_size, num_features).
    training: Boolean indicating whether the layer should behave in
        training mode or inference mode.
        
Returns:
    Output tensor with the same shape as input.

**get_config**

Returns the config of the layer.

Returns:
    Python dictionary containing the layer configuration.

---

## MultiResolutionTabularAttention

### MultiResolutionTabularAttention

Custom layer to apply multi-resolution attention for mixed-type tabular data.

This layer implements separate attention mechanisms for numerical and categorical features,
along with cross-attention between them. It's designed to handle the different characteristics
of numerical and categorical features in tabular data.

Args:
    num_heads (int): Number of attention heads
    d_model (int): Dimensionality of the attention model
    dropout_rate (float): Dropout rate for regularization
    name (str, optional): Name for the layer
    
Input shape:
    List of two tensors:
    - Numerical features: `(batch_size, num_samples, num_numerical_features)`
    - Categorical features: `(batch_size, num_samples, num_categorical_features)`
    
Output shape:
    List of two tensors with shapes:
    - `(batch_size, num_samples, d_model)` (numerical features)
    - `(batch_size, num_samples, d_model)` (categorical features)
    
Example:
    ```python
    import keras
    from kmr.layers import MultiResolutionTabularAttention
    
    # Create sample input data
    numerical = keras.random.normal((32, 100, 10))  # 32 batches, 100 samples, 10 numerical features
    categorical = keras.random.normal((32, 100, 5))  # 32 batches, 100 samples, 5 categorical features
    
    # Apply multi-resolution attention
    attention = MultiResolutionTabularAttention(num_heads=4, d_model=32, dropout_rate=0.1)
    num_out, cat_out = attention([numerical, categorical])
    print("Numerical output shape:", num_out.shape)  # (32, 100, 32)
    print("Categorical output shape:", cat_out.shape)  # (32, 100, 32)
    ```

#### Methods

**build**

Builds the layer with the given input shape.

Args:
    input_shape: List of two tuples defining the input shapes for numerical and categorical features.

**call**

Forward pass of the layer.

Args:
    inputs: List of two tensors [numerical_features, categorical_features]
    training: Boolean indicating whether the layer should behave in
        training mode or inference mode.
        
Returns:
    Tuple of two tensors (numerical_output, categorical_output)

**compute_output_shape**

Compute the output shape of the layer.

Args:
    input_shape: List of shapes of the input tensors.
    
Returns:
    List of shapes of the output tensors.

**get_config**

Get layer configuration.

Returns:
    Dictionary containing the layer configuration.

---

## NumericalAnomalyDetection

### NumericalAnomalyDetection

Numerical anomaly detection layer for identifying outliers in numerical features.

This layer learns a distribution for each numerical feature and outputs an anomaly
score for each feature based on how far it deviates from the learned distribution.
The layer uses a combination of mean, variance, and autoencoder reconstruction error
to detect anomalies.

Example:
    ```python
    import tensorflow as tf
    from kmr.layers import NumericalAnomalyDetection

    # Suppose we have 5 numerical features
    x = tf.random.normal((32, 5))  # Batch of 32 samples
    # Create a NumericalAnomalyDetection layer
    anomaly_layer = NumericalAnomalyDetection(
        hidden_dims=[8, 4],
        reconstruction_weight=0.5,
        distribution_weight=0.5
    )
    anomaly_scores = anomaly_layer(x)
    print("Anomaly scores shape:", anomaly_scores.shape)  # Expected: (32, 5)
    ```

#### Methods

**build**

Build the layer.

Args:
    input_shape: Input shape tuple.

**call**

Forward pass.

Args:
    inputs: Input tensor of shape (batch_size, num_features).

Returns:
    Anomaly scores tensor of shape (batch_size, num_features).

**compute_output_shape**

Compute output shape.

Args:
    input_shape: Input shape tuple.

Returns:
    Output shape tuple.

**get_config**

Get layer configuration.

Returns:
    Layer configuration dictionary.

---

## RowAttention

### RowAttention

Row attention mechanism to weight samples dynamically.

This layer applies attention weights to each sample (row) in the input tensor.
The attention weights are computed using a two-layer neural network that takes
each sample as input and outputs a scalar attention weight.

Example:
    ```python
    import tensorflow as tf
    from kmr.layers import RowAttention
    
    # Create sample data
    batch_size = 32
    feature_dim = 10
    inputs = tf.random.normal((batch_size, feature_dim))
    
    # Apply row attention
    attention = RowAttention(feature_dim=feature_dim)
    weighted_outputs = attention(inputs)
    ```

#### Methods

**build**

Build the layer.

Args:
    input_shape: Shape of input tensor

**call**

Apply row attention.

Args:
    inputs: Input tensor of shape [batch_size, feature_dim]

Returns:
    Attention weighted tensor of shape [batch_size, feature_dim]

**get_config**

Get layer configuration.

Returns:
    Layer configuration dictionary

**from_config**

Create layer from configuration.

Args:
    config: Layer configuration dictionary

Returns:
    RowAttention instance

---

## SeasonLayer

### SeasonLayer

Layer for adding seasonal information based on month.

This layer adds seasonal information based on the month, encoding it as a one-hot vector
for the four seasons: Winter, Spring, Summer, and Fall.

Args:
    **kwargs: Additional layer arguments

Input shape:
    Tensor with shape: `(..., 4)` containing [year, month, day, day_of_week]

Output shape:
    Tensor with shape: `(..., 8)` containing the original 4 components plus
    4 one-hot encoded season values

#### Methods

**call**

Apply the layer to the inputs.

Args:
    inputs: Tensor with shape (..., 4) containing [year, month, day, day_of_week]

Returns:
    Tensor with shape (..., 8) containing the original 4 components plus
    4 one-hot encoded season values

**compute_output_shape**

Compute the output shape of the layer.

Args:
    input_shape: Shape of the input tensor

Returns:
    Output shape

**get_config**

Return the configuration of the layer.

Returns:
    Dictionary containing the layer configuration

---

## SlowNetwork

### SlowNetwork

A multi-layer network with configurable depth and width.

This layer processes input features through multiple dense layers with ReLU activations,
and projects the output back to the original feature dimension.

Args:
    input_dim: Dimension of the input features.
    num_layers: Number of hidden layers. Default is 3.
    units: Number of units per hidden layer. Default is 128.
    name: Optional name for the layer.
    
Input shape:
    2D tensor with shape: `(batch_size, input_dim)`
    
Output shape:
    2D tensor with shape: `(batch_size, input_dim)` (same as input)
    
Example:
    ```python
    import keras
    from kmr.layers import SlowNetwork
    
    # Create sample input data
    x = keras.random.normal((32, 16))  # 32 samples, 16 features
    
    # Create the layer
    slow_net = SlowNetwork(input_dim=16, num_layers=3, units=64)
    y = slow_net(x)
    print("Output shape:", y.shape)  # (32, 16)
    ```

#### Methods

**build**

Builds the layer with the given input shape.

Args:
    input_shape: Tuple of integers defining the input shape.

**call**

Forward pass of the layer.

Args:
    inputs: Input tensor with shape (batch_size, input_dim).
    training: Boolean indicating whether the layer should behave in
        training mode or inference mode.
        
Returns:
    Output tensor with the same shape as input.

**get_config**

Returns the config of the layer.

Returns:
    Python dictionary containing the layer configuration.

---

## SparseAttentionWeighting

### SparseAttentionWeighting

Sparse attention mechanism with temperature scaling for module outputs combination.

This layer implements a learnable attention mechanism that combines outputs from multiple
modules using temperature-scaled attention weights. The attention weights are learned
during training and can be made more or less sparse by adjusting the temperature parameter.
A higher temperature leads to more uniform weights, while a lower temperature makes the
weights more concentrated on specific modules.

Key features:
1. Learnable module importance weights
2. Temperature-controlled sparsity
3. Softmax-based attention mechanism
4. Support for variable number of input features per module

Example:
```python
import numpy as np
from keras import layers, Model
from kmr.layers import SparseAttentionWeighting

# Create sample module outputs
batch_size = 32
num_modules = 3
feature_dim = 64

# Create three different module outputs
module1 = layers.Dense(feature_dim)(inputs)
module2 = layers.Dense(feature_dim)(inputs)
module3 = layers.Dense(feature_dim)(inputs)

# Combine module outputs using sparse attention
attention = SparseAttentionWeighting(
    num_modules=num_modules,
    temperature=0.5  # Lower temperature for sharper attention
)
combined_output = attention([module1, module2, module3])

# The layer will learn which modules are most important
# and weight their outputs accordingly
```

Args:
    num_modules: Number of input modules whose outputs will be combined.
    temperature: Temperature parameter for softmax scaling. Default is 1.0.
        - temperature > 1.0: More uniform attention weights
        - temperature < 1.0: More sparse attention weights
        - temperature = 1.0: Standard softmax behavior

#### Methods

**call**

Apply sparse attention weighting to combine module outputs.

This method performs the following steps:
1. Applies temperature scaling to the attention weights
2. Computes softmax to get attention probabilities
3. Stacks all module outputs
4. Applies attention weights to combine outputs

Args:
    module_outputs: Sequence of module output tensors, each of shape
        (..., feature_dim). The feature dimension can vary between modules.

Returns:
    Combined tensor of shape (..., feature_dim) representing the
    attention-weighted sum of module outputs.

Raises:
    ValueError: If len(module_outputs) != num_modules

**get_config**

Get layer configuration for serialization.

Returns:
    Dictionary containing the layer configuration:
    - num_modules: Number of input modules
    - temperature: Temperature scaling parameter

**from_config**

Create layer from configuration.

Args:
    config: Layer configuration dictionary

Returns:
    SparseAttentionWeighting instance

---

## StochasticDepth

### StochasticDepth

Stochastic depth layer for regularization.

This layer randomly drops entire residual branches with a specified probability
during training. During inference, all branches are kept and scaled appropriately.
This technique helps reduce overfitting and training time in deep networks.

Reference:
    - [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382)

Example:
    ```python
    from keras import random, layers
    from kmr.layers import StochasticDepth
    
    # Create sample residual branch
    inputs = random.normal((32, 64, 64, 128))
    residual = layers.Conv2D(128, 3, padding="same")(inputs)
    residual = layers.BatchNormalization()(residual)
    residual = layers.ReLU()(residual)
    
    # Apply stochastic depth
    outputs = StochasticDepth(survival_prob=0.8)([inputs, residual])
    ```

#### Methods

**call**

Apply stochastic depth.

Args:
    inputs: List of [shortcut, residual] tensors
    training: Whether in training mode

Returns:
    Output tensor after applying stochastic depth

Raises:
    ValueError: If inputs is not a list of length 2

**compute_output_shape**

Compute output shape.

Args:
    input_shape: List of input shape tuples

Returns:
    Output shape tuple

**get_config**

Get layer configuration.

Returns:
    Layer configuration dictionary

**from_config**

Create layer from configuration.

Args:
    config: Layer configuration dictionary

Returns:
    StochasticDepth instance

---

## TabularAttention

### TabularAttention

Custom layer to apply inter-feature and inter-sample attention for tabular data.

This layer implements a dual attention mechanism:
1. Inter-feature attention: Captures dependencies between features for each sample
2. Inter-sample attention: Captures dependencies between samples for each feature

The layer uses MultiHeadAttention for both attention mechanisms and includes
layer normalization, dropout, and a feed-forward network.

Args:
    num_heads (int): Number of attention heads
    d_model (int): Dimensionality of the attention model
    dropout_rate (float): Dropout rate for regularization
    name (str, optional): Name for the layer
    
Input shape:
    Tensor with shape: `(batch_size, num_samples, num_features)`
    
Output shape:
    Tensor with shape: `(batch_size, num_samples, d_model)`
    
Example:
    ```python
    import keras
    from kmr.layers import TabularAttention
    
    # Create sample input data
    x = keras.random.normal((32, 100, 20))  # 32 batches, 100 samples, 20 features
    
    # Apply tabular attention
    attention = TabularAttention(num_heads=4, d_model=32, dropout_rate=0.1)
    y = attention(x)
    print("Output shape:", y.shape)  # (32, 100, 32)
    ```

#### Methods

**build**

Builds the layer with the given input shape.

Args:
    input_shape: Tuple of integers defining the input shape.

**call**

Forward pass of the layer.

Args:
    inputs: Input tensor of shape (batch_size, num_samples, num_features)
    training: Boolean indicating whether the layer should behave in
        training mode or inference mode.
        
Returns:
    Output tensor of shape (batch_size, num_samples, d_model)

**compute_output_shape**

Compute the output shape of the layer.

Args:
    input_shape: Shape of the input tensor.
    
Returns:
    Shape of the output tensor.

**get_config**

Get layer configuration.

Returns:
    Dictionary containing the layer configuration.

---

## TabularMoELayer

### TabularMoELayer

Mixture-of-Experts layer for tabular data.

This layer routes input features through multiple expert sub-networks and
aggregates their outputs via a learnable gating mechanism. Each expert is a small
MLP, and the gate learns to weight their contributions.

Args:
    num_experts: Number of expert networks. Default is 4.
    expert_units: Number of hidden units in each expert network. Default is 16.
    name: Optional name for the layer.

Input shape:
    2D tensor with shape: `(batch_size, num_features)`

Output shape:
    2D tensor with shape: `(batch_size, num_features)` (same as input)

Example:
    ```python
    import keras
    from kmr.layers import TabularMoELayer

    # Tabular data with 8 features
    x = keras.random.normal((32, 8))
    
    # Create the layer with 4 experts and 16 units per expert
    moe_layer = TabularMoELayer(num_experts=4, expert_units=16)
    y = moe_layer(x)
    print("MoE output shape:", y.shape)  # Expected: (32, 8)
    ```

#### Methods

**build**

Builds the layer with the given input shape.

Args:
    input_shape: Tuple of integers defining the input shape.

**call**

Forward pass of the layer.

Args:
    inputs: Input tensor with shape (batch_size, num_features).
    training: Boolean indicating whether the layer should behave in
        training mode or inference mode.

Returns:
    Output tensor with the same shape as input.

**get_config**

Returns the config of the layer.

Returns:
    Python dictionary containing the layer configuration.

---

## TransformerBlock

### TransformerBlock

Transformer block with multi-head attention and feed-forward layers.

This layer implements a standard transformer block with multi-head self-attention
followed by a feed-forward network, with residual connections and layer normalization.

Args:
    dim_model (int): Dimensionality of the model.
    num_heads (int): Number of attention heads.
    ff_units (int): Number of units in the feed-forward network.
    dropout_rate (float): Dropout rate for regularization.
    name (str, optional): Name for the layer.
    
Input shape:
    Tensor with shape: `(batch_size, sequence_length, dim_model)` or 
    `(batch_size, dim_model)` which will be automatically reshaped.
    
Output shape:
    Tensor with shape: `(batch_size, sequence_length, dim_model)` or
    `(batch_size, dim_model)` matching the input shape.
    
Example:
    ```python
    import keras
    from kmr.layers import TransformerBlock
    
    # Create sample input data
    x = keras.random.normal((32, 10, 64))  # 32 samples, 10 time steps, 64 features
    
    # Apply transformer block
    transformer = TransformerBlock(dim_model=64, num_heads=4, ff_units=128, dropout_rate=0.1)
    y = transformer(x)
    print("Output shape:", y.shape)  # (32, 10, 64)
    ```

#### Methods

**build**

Builds the layer with the given input shape.

Args:
    input_shape: Tuple of integers defining the input shape.

**call**

Forward pass of the layer.

Args:
    inputs: Input tensor.
    training: Boolean indicating whether the layer should behave in
        training mode or inference mode.
        
Returns:
    Output tensor after applying transformer block.

**compute_output_shape**

Compute the output shape of the layer.

Args:
    input_shape: Shape of the input tensor.
    
Returns:
    Shape of the output tensor.

**get_config**

Get layer configuration.

Returns:
    Dictionary containing the layer configuration.

---

## VariableSelection

### VariableSelection

Layer for dynamic feature selection using gated residual networks.

This layer applies a gated residual network to each feature independently and learns
feature weights through a softmax layer. It can optionally use a context vector to
condition the feature selection.

Args:
    nr_features (int): Number of input features
    units (int): Number of hidden units in the gated residual network
    dropout_rate (float): Dropout rate for regularization
    use_context (bool): Whether to use a context vector for conditioning
    name (str, optional): Name for the layer
    
Input shape:
    If use_context is False:
        - Single tensor with shape: `(batch_size, nr_features, feature_dim)`
    If use_context is True:
        - List of two tensors:
            - Features tensor with shape: `(batch_size, nr_features, feature_dim)`
            - Context tensor with shape: `(batch_size, context_dim)`
    
Output shape:
    Tuple of two tensors:
    - Selected features: `(batch_size, feature_dim)`
    - Feature weights: `(batch_size, nr_features)`
    
Example:
    ```python
    import keras
    from kmr.layers import VariableSelection
    
    # Create sample input data
    x = keras.random.normal((32, 10, 16))  # 32 batches, 10 features, 16 dims per feature
    
    # Without context
    vs = VariableSelection(nr_features=10, units=32, dropout_rate=0.1)
    selected, weights = vs(x)
    print("Selected features shape:", selected.shape)  # (32, 16)
    print("Feature weights shape:", weights.shape)  # (32, 10)
    
    # With context
    context = keras.random.normal((32, 64))  # 32 batches, 64-dim context
    vs_context = VariableSelection(nr_features=10, units=32, dropout_rate=0.1, use_context=True)
    selected, weights = vs_context([x, context])
    ```

#### Methods

**build**

Builds the layer with the given input shape.

Args:
    input_shape: Shape of the input tensor or list of shapes if using context.

**call**

Forward pass of the layer.

Args:
    inputs: Input tensor or list of tensors if using context
    training: Boolean indicating whether the layer should behave in
        training mode or inference mode.
        
Returns:
    Tuple of (selected_features, feature_weights)

**compute_output_shape**

Compute the output shape of the layer.

Args:
    input_shape: Shape of the input tensor or list of shapes if using context.
    
Returns:
    List of shapes for the output tensors.

**get_config**

Get layer configuration.

Returns:
    Dictionary containing the layer configuration.

---
