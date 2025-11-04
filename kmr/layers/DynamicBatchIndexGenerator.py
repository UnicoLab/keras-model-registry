"""Dynamic batch index generator for recommendation systems.

This layer generates sequential batch indices dynamically based on input batch size.
Useful for indexing operations in recommendation models where batch indices are needed.
"""

from typing import Any
from keras import ops
from keras import KerasTensor
from keras.saving import register_keras_serializable

from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class DynamicBatchIndexGenerator(BaseLayer):
    """Generates dynamic batch indices for recommendation batching.

    This layer creates a tensor of sequential indices from 0 to batch_size-1,
    enabling dynamic batch indexing operations in recommendation systems.
    The indices are generated dynamically based on the input batch size.

    Args:
        name: Optional name for the layer.

    Input shape:
        Any tensor with shape `(batch_size, ...)`

    Output shape:
        `(batch_size,)` - Array of indices [0, 1, 2, ..., batch_size-1]

    Example:
        ```python
        import keras
        from kmr.layers import DynamicBatchIndexGenerator

        # Create sample input data
        x = keras.random.normal((32, 10))  # 32 samples, 10 features

        # Create the layer
        index_gen = DynamicBatchIndexGenerator()
        indices = index_gen(x)
        print("Indices shape:", indices.shape)  # (32,)
        print("Indices:", indices)  # [0, 1, 2, ..., 31]
        ```
    """

    def __init__(self, name: str | None = None, **kwargs: Any) -> None:
        """Initialize the DynamicBatchIndexGenerator layer.

        Args:
            name: Name of the layer.
            **kwargs: Additional keyword arguments.
        """
        # No parameters to validate
        self._validate_params()

        # Call parent's __init__
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters.

        This layer has no parameters to validate.
        """
        pass

    def call(self, inputs: KerasTensor) -> KerasTensor:
        """Generate dynamic batch indices.

        Args:
            inputs: Input tensor of any shape.

        Returns:
            Batch indices tensor of shape (batch_size,).
        """
        batch_size = ops.shape(inputs)[0]
        indices = ops.arange(batch_size, dtype=inputs.dtype)
        return indices

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        return config
