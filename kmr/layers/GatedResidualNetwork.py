"""This module implements a GatedResidualNetwork layer that combines residual connections
with gated linear units for improved gradient flow and feature transformation.
"""

from typing import Any
from keras import layers, ops
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kmr.layers._base_layer import BaseLayer
from kmr.layers.GatedLinearUnit import GatedLinearUnit


@register_keras_serializable(package="kmr.layers")
class GatedResidualNetwork(BaseLayer):
    """GatedResidualNetwork is a custom Keras layer that implements a gated residual network.

    This layer applies a series of transformations to the input tensor and combines the result with the input
    using a residual connection. The transformations include a dense layer with ELU activation, a dense linear
    layer, a dropout layer, a gated linear unit layer, layer normalization, and a final dense layer.

    Args:
        units (int): Positive integer, dimensionality of the output space.
        dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.2.
        name (str, optional): Name for the layer.

    Input shape:
        Tensor with shape: `(batch_size, ..., input_dim)`

    Output shape:
        Tensor with shape: `(batch_size, ..., units)`

    Example:
        ```python
        import keras
        from kmr.layers import GatedResidualNetwork

        # Create sample input data
        x = keras.random.normal((32, 16))  # 32 samples, 16 features

        # Create the layer
        grn = GatedResidualNetwork(units=16, dropout_rate=0.2)
        y = grn(x)
        print("Output shape:", y.shape)  # (32, 16)
        ```
    """

    def __init__(
        self,
        units: int,
        dropout_rate: float = 0.2,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the GatedResidualNetwork.

        Args:
            units: Number of units in the network.
            dropout_rate: Dropout rate.
            name: Name of the layer.
            **kwargs: Additional keyword arguments.
        """
        # Set private attributes first
        self._units = units
        self._dropout_rate = dropout_rate

        # Validate parameters
        self._validate_params()

        # Set public attributes BEFORE calling parent's __init__
        self.units = self._units
        self.dropout_rate = self._dropout_rate

        # Initialize instance variables
        self.elu_dense: layers.Dense | None = None
        self.linear_dense: layers.Dense | None = None
        self.dropout: layers.Dropout | None = None
        self.gated_linear_unit: GatedLinearUnit | None = None
        self.project: layers.Dense | None = None
        self.layer_norm: layers.LayerNormalization | None = None

        # Call parent's __init__ after setting public attributes
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if not isinstance(self._units, int) or self._units <= 0:
            raise ValueError(f"units must be a positive integer, got {self._units}")
        if (
            not isinstance(self._dropout_rate, int | float)
            or not 0 <= self._dropout_rate < 1
        ):
            raise ValueError(
                f"dropout_rate must be between 0 and 1, got {self._dropout_rate}",
            )

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Builds the layer with the given input shape.

        Args:
            input_shape: Tuple of integers defining the input shape.
        """
        # Create the layers
        self.elu_dense = layers.Dense(self.units, activation="elu")
        self.linear_dense = layers.Dense(self.units)
        self.dropout = layers.Dropout(self.dropout_rate)
        self.gated_linear_unit = GatedLinearUnit(units=self.units)

        # Create projection layer if input dimension doesn't match output dimension
        input_dim = input_shape[-1]
        if input_dim != self.units:
            self.project = layers.Dense(self.units)

        self.layer_norm = layers.LayerNormalization()

        super().build(input_shape)

    def call(self, inputs: KerasTensor, training: bool = False) -> KerasTensor:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Output tensor after applying gated residual transformations.
        """
        # Check if layer is built
        if any(
            layer is None
            for layer in [
                self.elu_dense,
                self.linear_dense,
                self.dropout,
                self.gated_linear_unit,
                self.project,
                self.layer_norm,
            ]
        ):
            raise ValueError("Layer not built. Call build() first.")

        # Cast inputs to float32 at the start
        inputs = ops.cast(inputs, "float32")

        # Apply transformations
        x = self.elu_dense(inputs)
        x = self.linear_dense(x)
        x = self.dropout(x, training=training)

        # Apply projection if needed
        if hasattr(self, "project") and self.project is not None:
            inputs = self.project(inputs)

        # Apply gated linear unit and add residual connection
        x = inputs + self.gated_linear_unit(x)

        # Apply layer normalization
        x = self.layer_norm(x)

        return x

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "dropout_rate": self.dropout_rate,
            },
        )
        return config
