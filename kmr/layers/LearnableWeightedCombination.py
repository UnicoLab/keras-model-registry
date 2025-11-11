"""Learnable weighted combination layer for score aggregation."""

from typing import Any
from keras import ops
from keras import KerasTensor
from keras.saving import register_keras_serializable

from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class LearnableWeightedCombination(BaseLayer):
    """Combines multiple scores with learnable weights.

    Uses trainable weights to combine multiple recommendation scores
    (e.g., collaborative filtering + content-based + ranking).

    Args:
        num_scores: Number of scores to combine (default=3).
        name: Optional name for the layer.
    """

    def __init__(
        self,
        num_scores: int = 3,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize layer."""
        self._num_scores = num_scores
        self._validate_params()

        self.num_scores = self._num_scores

        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate parameters."""
        if not isinstance(self._num_scores, int) or self._num_scores <= 0:
            raise ValueError(f"num_scores must be positive, got {self._num_scores}")

    def build(self, input_shape: tuple) -> None:
        """Build layer."""
        self.combination_weights = self.add_weight(
            name="combination_weights",
            shape=(self._num_scores,),
            initializer="ones",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: list[KerasTensor]) -> KerasTensor:
        """Combine scores with learnable weights.

        Args:
            inputs: List of score tensors, each (batch_size, ...) where ... is any shape.

        Returns:
            Combined scores with same shape as inputs except num_scores dimension.
        """
        # Stack scores along a new axis
        # If each input is (batch_size, n1, n2, ..., nk, 1), stack gives (num_scores, batch_size, n1, n2, ..., nk, 1)
        # We need to reorganize to (batch_size, num_scores, n1, n2, ..., nk, 1) then reduce
        stacked = ops.stack(inputs, axis=0)  # (num_scores, batch_size, ...)

        # Move num_scores to axis 1: (batch_size, num_scores, ...)
        stacked = ops.transpose(
            stacked,
            axes=[1, 0] + list(range(2, len(ops.shape(stacked)))),
        )

        # Squeeze the last dimension if it's 1: (batch_size, num_scores, ...)
        if ops.shape(stacked)[-1] == 1:
            stacked = ops.squeeze(stacked, axis=-1)  # (batch_size, num_scores, ...)

        # Apply weights: normalize and multiply
        normalized_weights = ops.softmax(self.combination_weights)

        # Reshape weights to broadcast correctly: (num_scores,) -> (1, num_scores, 1, 1, ...)
        weight_shape = [1, self._num_scores] + [1] * (len(ops.shape(stacked)) - 2)
        normalized_weights_reshaped = ops.reshape(normalized_weights, weight_shape)

        # Element-wise multiplication and sum across num_scores axis
        weighted = (
            stacked * normalized_weights_reshaped
        )  # (batch_size, num_scores, ...)
        combined = ops.sum(weighted, axis=1, keepdims=True)  # (batch_size, 1, ...)

        return combined

    def get_config(self) -> dict[str, Any]:
        """Get configuration."""
        config = super().get_config()
        config.update({"num_scores": self.num_scores})
        return config
