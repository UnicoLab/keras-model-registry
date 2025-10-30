"""This module implements a DifferentialPreprocessingLayer that applies multiple candidate transformations
to tabular data and learns to combine them optimally. It also handles missing values with learnable imputation.
This approach is useful for tabular data where the optimal preprocessing strategy is not known in advance.
"""

from typing import Any
from loguru import logger
from keras import layers, ops
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class DifferentialPreprocessingLayer(BaseLayer):
    """Differentiable preprocessing layer for numeric tabular data with multiple candidate transformations.

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
    """

    def __init__(
        self,
        num_features: int,
        mlp_hidden_units: int = 4,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the DifferentialPreprocessingLayer.

        Args:
            num_features: Number of input features.
            mlp_hidden_units: Number of hidden units in MLP.
            name: Name of the layer.
            **kwargs: Additional keyword arguments.
        """
        # Set public attributes
        self.num_features = num_features
        self.mlp_hidden_units = mlp_hidden_units
        self.num_candidates = 4  # We have 4 candidate branches

        # Initialize instance variables
        self.impute: layers.Embedding | None = None
        self.gamma: layers.Embedding | None = None
        self.beta: layers.Embedding | None = None
        self.mlp_hidden: layers.Dense | None = None
        self.mlp_output: layers.Dense | None = None
        self.alpha: layers.Embedding | None = None

        # Validate parameters during initialization
        self._validate_params()

        # Call parent's __init__
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if self.num_features <= 0:
            raise ValueError(f"num_features must be positive, got {self.num_features}")
        if self.mlp_hidden_units <= 0:
            raise ValueError(
                f"mlp_hidden_units must be positive, got {self.mlp_hidden_units}",
            )

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Builds the layer with the given input shape.

        Args:
            input_shape: Tuple of integers defining the input shape.
        """
        # Validate parameters again during build
        self._validate_params()

        # Trainable imputation vector (shape: [num_features])
        self.impute = self.add_weight(
            name="impute",
            shape=(self.num_features,),
            initializer="zeros",
            trainable=True,
        )

        # Affine branch parameters: scale and bias
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

        # Nonlinear branch: a small MLP
        self.mlp_hidden = layers.Dense(
            units=self.mlp_hidden_units,
            activation="relu",
            name="mlp_hidden",
        )

        self.mlp_output = layers.Dense(
            units=self.num_features,
            activation=None,
            name="mlp_output",
        )

        # Combination weights for the 4 candidate transformations
        self.alpha = self.add_weight(
            name="alpha",
            shape=(self.num_candidates,),
            initializer="zeros",
            trainable=True,
        )

        logger.debug(
            f"DifferentialPreprocessingLayer built with num_features={self.num_features}, "
            f"mlp_hidden_units={self.mlp_hidden_units}",
        )
        super().build(input_shape)

    def call(self, inputs: KerasTensor, _: bool | None = None) -> KerasTensor:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor with shape (batch_size, num_features).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Output tensor with the same shape as input.
        """
        # Ensure layer is built (Keras will auto-build on first call)
        if not self.built:
            self.build(inputs.shape)

        # Step 1: Impute missing values
        imputed = ops.where(
            ops.isnan(inputs),
            ops.reshape(self.impute, (1, self.num_features)),
            inputs,
        )

        # Candidate 1: Identity
        candidate_identity = imputed

        # Candidate 2: Affine transformation
        candidate_affine = self.gamma * imputed + self.beta

        # Candidate 3: Nonlinear transformation (MLP)
        candidate_nonlinear = self.mlp_hidden(imputed)
        candidate_nonlinear = self.mlp_output(candidate_nonlinear)

        # Candidate 4: Log transformation
        # Use softplus to ensure the argument is positive
        candidate_log = ops.log(ops.nn.softplus(imputed) + 1e-6)

        # Stack candidates: shape (batch, num_features, num_candidates)
        candidates = ops.stack(
            [candidate_identity, candidate_affine, candidate_nonlinear, candidate_log],
            axis=-1,
        )

        # Compute softmax weights
        weights = ops.nn.softmax(self.alpha)
        weights = ops.reshape(weights, (1, 1, self.num_candidates))

        # Weighted sum
        output = ops.sum(weights * candidates, axis=-1)

        return output

    def get_config(self) -> dict[str, Any]:
        """Returns the config of the layer.

        Returns:
            Python dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "num_features": self.num_features,
                "mlp_hidden_units": self.mlp_hidden_units,
            },
        )
        return config
