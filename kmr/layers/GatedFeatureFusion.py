"""This module implements a GatedFeatureFusion layer that combines two feature representations
through a learned gating mechanism. It's particularly useful for tabular datasets with
multiple representations (e.g., raw numeric features alongside embeddings).
"""

from typing import Any
from loguru import logger
from keras import layers, ops
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class GatedFeatureFusion(BaseLayer):
    """Gated feature fusion layer for combining two feature representations.

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
    """

    def __init__(
        self,
        activation: str = "sigmoid",
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the GatedFeatureFusion layer.

        Args:
            activation: Activation function for the gate.
            name: Name of the layer.
            **kwargs: Additional keyword arguments.
        """
        # Set private attributes first
        self._activation = activation

        # No validation needed for activation as Keras will validate it

        # Set public attributes BEFORE calling parent's __init__
        self.activation = self._activation
        self.fusion_gate: layers.Dense | None = None

        # Call parent's __init__ after setting public attributes
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        # No specific validation needed for this layer
        pass

    def build(self, input_shape: list[tuple[int, ...]]) -> None:
        """Builds the layer with the given input shape.

        Args:
            input_shape: list of two input shapes, each a tuple of integers.
        """
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError(
                f"GatedFeatureFusion expects a list of 2 input shapes, got {input_shape}",
            )

        if input_shape[0][-1] != input_shape[1][-1]:
            raise ValueError(
                f"Both inputs must have the same feature dimension, "
                f"got {input_shape[0][-1]} and {input_shape[1][-1]}",
            )

        # Create the fusion gate layer
        self.fusion_gate = layers.Dense(
            input_shape[0][-1],
            activation=self.activation,
            name="fusion_gate",
        )

        logger.debug(f"GatedFeatureFusion built with activation={self.activation}")
        super().build(input_shape)

    def call(
        self,
        inputs: list[KerasTensor],
        _: bool | None = None,
    ) -> KerasTensor:
        """Forward pass of the layer.

        Args:
            inputs: list of two input tensors to be fused.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Fused output tensor with the same shape as each input.
        """
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError(
                f"GatedFeatureFusion expects a list of 2 inputs, got {inputs}",
            )

        feat1, feat2 = inputs

        # Concatenate the features along the last dimension
        concatenated = ops.concatenate([feat1, feat2], axis=-1)

        # Compute the gate values
        if self.fusion_gate is None:
            raise ValueError("Layer not built. Call build() first.")
        gate = self.fusion_gate(concatenated)

        # Fuse using the learned gate
        output = gate * feat1 + (1 - gate) * feat2

        return output

    def get_config(self) -> dict[str, Any]:
        """Returns the config of the layer.

        Returns:
            Python dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "activation": self.activation,
            },
        )
        return config
