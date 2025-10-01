from typing import Any
from loguru import logger
from keras import layers, ops
from keras import KerasTensor
from kmr.layers._base_layer import BaseLayer
from keras.saving import register_keras_serializable


@register_keras_serializable(package="kmr.layers")
class DifferentialPreprocssingLayer(BaseLayer):
    """Differentiable preprocessing layer for numeric tabular data with multiple candidate transformations.

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
    """

    def __init__(self, num_features: int, mlp_hidden_units: int = 4, **kwargs):
        """Initialize the DifferentialPreprocssingLayer.

        Args:
            num_features: Number of numeric features in the input tensor. This must match
                the last dimension of the input tensor.
            mlp_hidden_units: Number of hidden units in the nonlinear transformation
                branch. A larger value allows for more complex nonlinear transformations
                but increases computational cost. Default is 4, which works well for
                most cases.
            **kwargs: Additional keyword arguments passed to the parent Layer class,
                such as name, dtype, or trainable.

        Raises:
            ValueError: If num_features <= 0 or mlp_hidden_units <= 0. Both parameters
                must be positive integers.

        Notes:
            The layer creates several trainable weights:
            - Imputation vector (shape: [num_features])
            - Affine transformation parameters (shape: [num_features] each)
            - MLP weights for nonlinear transformation
            - Combination weights for the transformations (shape: [4])

            The total number of trainable parameters is approximately:
            num_features * (3 + mlp_hidden_units) + mlp_hidden_units * num_features + 4
        """
        # Initialize attributes before super().__init__
        self.num_features = num_features
        self.mlp_hidden_units = mlp_hidden_units
        self.num_candidates = 4  # We have 4 candidate branches

        # Validate parameters
        if num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}")
        if mlp_hidden_units <= 0:
            raise ValueError(
                f"mlp_hidden_units must be positive, got {mlp_hidden_units}",
            )

        super().__init__(**kwargs)

    def build(self, input_shape: tuple) -> None:
        """Build the layer by creating trainable weights and sublayers.

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
        """
        if input_shape[1] != self.num_features:
            raise ValueError(
                f"Input shape {input_shape} does not match num_features {self.num_features}",
            )

        # Trainable imputation vector (shape: [num_features])
        self.impute = self.add_weight(
            name="impute",
            shape=(self.num_features,),
            initializer="zeros",
            trainable=True,
        )
        # Affine branch parameters: scale and bias.
        self.gamma = self.add_weight(
            name="gamma",
            shape=(self.num_features,),
            initializer="ones",
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(self.num_features,),
            initializer="zeros",
            trainable=True,
        )
        # Nonlinear branch: a small MLP.
        self.mlp_hidden = layers.Dense(
            units=self.mlp_hidden_units,
            activation="relu",
            name="mlp_hidden",
            kernel_initializer="glorot_uniform",
        )
        self.mlp_output = layers.Dense(
            units=self.num_features,
            activation=None,
            name="mlp_output",
            kernel_initializer="glorot_uniform",
        )
        # Combination weights for the 4 candidate transformations.
        self.alpha = self.add_weight(
            name="alpha",
            shape=(self.num_candidates,),
            initializer="zeros",
            trainable=True,
        )
        logger.debug("Layer built with {} features.", self.num_features)
        super().build(input_shape)

    def call(self, inputs: KerasTensor | Any, training: bool = False) -> KerasTensor:
        """Forward pass of the layer, applying the preprocessing pipeline.

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
        """
        # Input validation
        if not isinstance(inputs, (KerasTensor)):
            inputs = ops.convert_to_tensor(inputs)

        # Step 1: Impute missing values.
        is_nan = ops.isnan(inputs)
        imputed = ops.where(
            is_nan,
            ops.reshape(self.impute, (1, self.num_features)),
            inputs,
        )

        # Candidate 1: Identity.
        candidate_identity = imputed

        # Candidate 2: Affine transformation.
        candidate_affine = self.gamma * imputed + self.beta

        # Candidate 3: Nonlinear transformation (MLP).
        candidate_nonlinear = self.mlp_hidden(imputed)
        candidate_nonlinear = self.mlp_output(candidate_nonlinear)

        # Candidate 4: Log transformation.
        # Use softplus to ensure the argument is positive.
        epsilon = ops.cast(1e-6, dtype=inputs.dtype)
        candidate_log = ops.log(ops.nn.softplus(imputed) + epsilon)

        # Stack candidates: shape (batch, num_features, num_candidates)
        candidates = ops.stack(
            [candidate_identity, candidate_affine, candidate_nonlinear, candidate_log],
            axis=-1,
        )

        # Compute softmax weights.
        weights = ops.nn.softmax(self.alpha)
        weights = ops.reshape(weights, (1, 1, self.num_candidates))

        # Weighted sum.
        output = ops.sum(weights * candidates, axis=-1)
        return output

    def get_config(self) -> dict:
        """Return layer configuration for serialization.

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
        """
        config = super().get_config()
        config.update(
            {
                "num_features": self.num_features,
                "mlp_hidden_units": self.mlp_hidden_units,
            },
        )
        return config
