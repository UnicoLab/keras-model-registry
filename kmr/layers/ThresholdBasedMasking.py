"""Threshold-based masking layer for recommendation systems.

This layer applies threshold-based masking to filter features, setting values
below or above a threshold to zero. Useful for feature engineering and data filtering.
"""

from typing import Any
from keras import ops
from keras import KerasTensor
from keras.saving import register_keras_serializable

from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class ThresholdBasedMasking(BaseLayer):
    """Applies threshold-based masking to filter features by value.

    This layer creates a mask based on a threshold value and applies it to input
    tensors. Values above the threshold are preserved, values below are zeroed.
    Useful for filtering features in recommendation systems based on importance
    or activity levels.

    Args:
        threshold: Threshold value for masking (default=0.0).
            Values >= threshold are kept, others are zeroed.
        name: Optional name for the layer.

    Input shape:
        Tensor of any shape with numeric values.

    Output shape:
        Same as input shape.

    Example:
        ```python
        import keras
        from kmr.layers import ThresholdBasedMasking

        # Create sample input data
        x = keras.random.normal((32, 10))  # Random values around 0

        # Apply threshold masking (keep values >= 0.5)
        masking = ThresholdBasedMasking(threshold=0.5)
        masked_x = masking(x)
        print("Masked shape:", masked_x.shape)  # (32, 10)

        # All values < 0.5 are set to 0, values >= 0.5 are preserved
        ```
    """

    def __init__(
        self,
        threshold: float = 0.0,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the ThresholdBasedMasking layer.

        Args:
            threshold: Threshold value for masking.
            name: Name of the layer.
            **kwargs: Additional keyword arguments.
        """
        # Set private attributes first
        self._threshold = float(threshold)

        # Validate parameters
        self._validate_params()

        # Set public attributes BEFORE calling parent's __init__
        self.threshold = self._threshold

        # Call parent's __init__
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if not isinstance(self._threshold, int | float):
            raise ValueError(
                f"threshold must be numeric, got {type(self._threshold).__name__}",
            )

    def call(self, inputs: KerasTensor) -> KerasTensor:
        """Apply threshold-based masking.

        Args:
            inputs: Input tensor.

        Returns:
            Masked tensor with same shape as input.
        """
        # Create mask: True where values >= threshold
        mask = ops.cast(ops.greater_equal(inputs, self.threshold), dtype=inputs.dtype)
        # Apply mask: keep values >= threshold, zero out others
        masked = inputs * mask
        return masked

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "threshold": self.threshold,
            },
        )
        return config
