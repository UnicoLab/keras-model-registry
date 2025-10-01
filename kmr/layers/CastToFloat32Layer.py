"""This module implements a CastToFloat32Layer that casts input tensors to float32 data type."""

from typing import Any

from keras import ops
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class CastToFloat32Layer(BaseLayer):
    """Layer that casts input tensors to float32 data type.

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
    """

    def __init__(self, name: str | None = None, **kwargs: Any) -> None:
        # No private attributes to set

        # No parameters to validate

        # Call parent's __init__
        super().__init__(name=name, **kwargs)

    def call(self, inputs: KerasTensor) -> KerasTensor:
        """Cast inputs to float32.

        Args:
            inputs: Input tensor of any numeric data type.

        Returns:
            Input tensor cast to float32.
        """
        return ops.cast(inputs, "float32")

    def compute_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Same shape as input.
        """
        return input_shape
