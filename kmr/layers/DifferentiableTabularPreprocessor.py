"""This module implements a DifferentiableTabularPreprocessor layer that integrates preprocessing
into the model so that the optimal imputation and normalization parameters are learned end-to-end.
This approach is useful for tabular data with missing values and features that need normalization.
"""

from typing import Any
from loguru import logger
from keras import ops
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class DifferentiableTabularPreprocessor(BaseLayer):
    """A differentiable preprocessing layer for numeric tabular data.

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
    """

    def __init__(
        self,
        num_features: int,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Set public attributes
        self.num_features = num_features

        # Initialize instance variables
        self.impute = None
        self.gamma = None
        self.beta = None

        # Validate parameters during initialization
        self._validate_params()

        # Call parent's __init__
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if self.num_features <= 0:
            raise ValueError(f"num_features must be positive, got {self.num_features}")

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Builds the layer with the given input shape.

        Args:
            input_shape: Tuple of integers defining the input shape.
        """
        # Validate parameters again during build
        self._validate_params()

        # Trainable imputation: one scalar per feature to replace missing values.
        self.impute = self.add_weight(
            name="impute",
            shape=(self.num_features,),
            initializer="zeros",
            trainable=True,
        )

        # Learnable normalization parameters (scale and shift).
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

        logger.debug(
            f"DifferentiableTabularPreprocessor built with num_features={self.num_features}",
        )
        super().build(input_shape)

    def call(self, inputs: KerasTensor, training: bool | None = None) -> KerasTensor:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor with shape (batch_size, num_features).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Output tensor with the same shape as input.
        """
        # Replace missing values with the learned imputation value.
        # Reshape impute to (1, num_features) for broadcasting.
        imputed = ops.where(
            ops.isnan(inputs),
            ops.reshape(self.impute, (1, self.num_features)),
            inputs,
        )

        # Apply a learnable affine transformation to normalize each feature.
        # This is similar in spirit to BatchNorm but without relying on batch statistics.
        normalized = self.gamma * imputed + self.beta

        return normalized

    def get_config(self) -> dict[str, Any]:
        """Returns the config of the layer.

        Returns:
            Python dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "num_features": self.num_features,
            },
        )
        return config
