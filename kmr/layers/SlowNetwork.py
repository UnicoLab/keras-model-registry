"""This module implements a SlowNetwork layer that processes features through multiple dense layers.
It's designed to be used as a component in more complex architectures.
"""

from typing import Any
from loguru import logger
from keras import layers
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class SlowNetwork(BaseLayer):
    """A multi-layer network with configurable depth and width.

    This layer processes input features through multiple dense layers with ReLU activations,
    and projects the output back to the original feature dimension.

    Args:
        input_dim: Dimension of the input features.
        num_layers: Number of hidden layers. Default is 3.
        units: Number of units per hidden layer. Default is 128.
        name: Optional name for the layer.

    Input shape:
        2D tensor with shape: `(batch_size, input_dim)`

    Output shape:
        2D tensor with shape: `(batch_size, input_dim)` (same as input)

    Example:
        ```python
        import keras
        from kmr.layers import SlowNetwork

        # Create sample input data
        x = keras.random.normal((32, 16))  # 32 samples, 16 features

        # Create the layer
        slow_net = SlowNetwork(input_dim=16, num_layers=3, units=64)
        y = slow_net(x)
        print("Output shape:", y.shape)  # (32, 16)
        ```
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int = 3,
        units: int = 128,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the SlowNetwork layer.

        Args:
            input_dim: Input dimension.
            num_layers: Number of hidden layers.
            units: Number of units in each layer.
            name: Name of the layer.
            **kwargs: Additional keyword arguments.
        """
        # Set public attributes
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.units = units

        # Initialize instance variables
        self.hidden_layers: list[Any] | None = None
        self.output_layer: Any | None = None

        # Validate parameters
        self._validate_params()

        # Call parent's __init__
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.units <= 0:
            raise ValueError(f"units must be positive, got {self.units}")

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Builds the layer with the given input shape.

        Args:
            input_shape: Tuple of integers defining the input shape.
        """
        # Validate parameters again during build
        self._validate_params()

        # Create hidden layers
        self.hidden_layers = [
            layers.Dense(self.units, activation="relu", name=f"hidden_{i}")
            for i in range(self.num_layers)
        ]

        # Create output layer
        self.output_layer = layers.Dense(self.input_dim, activation=None, name="output")

        logger.debug(
            f"SlowNetwork built with input_dim={self.input_dim}, num_layers={self.num_layers}, units={self.units}",
        )
        super().build(input_shape)

    def call(self, inputs: KerasTensor, _: bool | None = None) -> KerasTensor:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor with shape (batch_size, input_dim).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Output tensor with the same shape as input.
        """
        logger.debug(f"SlowNetwork input shape: {inputs.shape}")
        x = inputs
        for i, layer in enumerate(self.hidden_layers):  # type: ignore
            x = layer(x)
            logger.debug(f"SlowNetwork layer {i} output shape: {x.shape}")

        output = self.output_layer(x)
        logger.debug(f"SlowNetwork output shape: {output.shape}")
        return output

    def get_config(self) -> dict[str, Any]:
        """Returns the config of the layer.

        Returns:
            Python dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "num_layers": self.num_layers,
                "units": self.units,
            },
        )
        return config
