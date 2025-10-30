"""This module implements a GatedLinearUnit layer that applies a gated linear transformation
to input tensors. It's particularly useful for controlling information flow in neural networks.
"""

from typing import Any
from keras import layers
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class GatedLinearUnit(BaseLayer):
    """GatedLinearUnit is a custom Keras layer that implements a gated linear unit.

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
    """

    def __init__(self, units: int, name: str | None = None, **kwargs: Any) -> None:
        """Initialize the GatedLinearUnit layer.

        Args:
            units: Number of units in the layer.
            name: Name of the layer.
            **kwargs: Additional keyword arguments.
        """
        # Set private attributes first
        self._units = units

        # Validate parameters
        self._validate_params()

        # Set public attributes BEFORE calling parent's __init__
        self.units = self._units
        self.linear: layers.Dense | None = None
        self.sigmoid: layers.Dense | None = None

        # Call parent's __init__ after setting public attributes
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if not isinstance(self._units, int) or self._units <= 0:
            raise ValueError(f"units must be a positive integer, got {self._units}")

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Builds the layer with the given input shape.

        Args:
            input_shape: Tuple of integers defining the input shape.
        """
        self.linear = layers.Dense(self.units)
        self.sigmoid = layers.Dense(self.units, activation="sigmoid")
        super().build(input_shape)

    def call(self, inputs: KerasTensor) -> KerasTensor:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor.

        Returns:
            Output tensor after applying gated linear transformation.
        """
        # Ensure layer is built (Keras will auto-build on first call)
        if not self.built:
            self.build(inputs.shape)

        return self.linear(inputs) * self.sigmoid(inputs)

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "units": self.units,
            },
        )
        return config
