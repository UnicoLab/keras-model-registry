"""Tensor dimension expander for recommendation systems.

This layer expands tensor dimensions at a specified axis, enabling flexible
shape manipulation for layer composition and data flow control.
"""

from typing import Any
from keras import ops
from keras import KerasTensor
from keras.saving import register_keras_serializable

from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class TensorDimensionExpander(BaseLayer):
    """Expands tensor dimensions at specified axis for shape manipulation.

    This layer adds a new dimension to input tensors at a specified axis,
    enabling flexible shape transformations required for recommendation
    model composition and data pipeline control.

    Args:
        axis: Position at which to expand dimension (default=1).
            Negative indices count from the end.
        name: Optional name for the layer.

    Input shape:
        Tensor of any shape.

    Output shape:
        Same as input shape with expanded dimension at specified axis.

    Example:
        ```python
        import keras
        from kmr.layers import TensorDimensionExpander

        # Create sample input data with shape (32, 10)
        x = keras.random.normal((32, 10))

        # Expand at axis 1: (32, 10) -> (32, 1, 10)
        expander = TensorDimensionExpander(axis=1)
        y = expander(x)
        print("Output shape:", y.shape)  # (32, 1, 10)

        # Expand at axis -1: (32, 10) -> (32, 10, 1)
        expander2 = TensorDimensionExpander(axis=-1)
        y2 = expander2(x)
        print("Output shape:", y2.shape)  # (32, 10, 1)
        ```
    """

    def __init__(self, axis: int = 1, name: str | None = None, **kwargs: Any) -> None:
        """Initialize the TensorDimensionExpander layer.

        Args:
            axis: Position at which to expand dimension.
            name: Name of the layer.
            **kwargs: Additional keyword arguments.
        """
        # Set private attributes first
        self._axis = axis

        # Validate parameters
        self._validate_params()

        # Set public attributes BEFORE calling parent's __init__
        self.axis = self._axis

        # Call parent's __init__
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if not isinstance(self._axis, int):
            raise ValueError(
                f"axis must be an integer, got {type(self._axis).__name__}",
            )

    def call(self, inputs: KerasTensor) -> KerasTensor:
        """Expand tensor dimension.

        Args:
            inputs: Input tensor.

        Returns:
            Output tensor with expanded dimension at specified axis.
        """
        return ops.expand_dims(inputs, axis=self.axis)

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "axis": self.axis,
            },
        )
        return config
