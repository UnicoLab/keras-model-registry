"""Temporal Mixing layer for TSMixer model."""

from typing import Any

import keras
from keras import KerasTensor, layers, ops
from keras.saving import register_keras_serializable
from loguru import logger

from kerasfactory.layers._base_layer import BaseLayer


@register_keras_serializable(package="kerasfactory.layers")
class TemporalMixing(BaseLayer):
    """Temporal mixing layer using MLP on time dimension.

    Applies batch normalization and linear transformation across the time
    dimension to mix temporal information while preserving the multivariate
    structure.

    Args:
        n_series: Number of time series (channels/features).
        input_size: Length of the time series (sequence length).
        dropout: Dropout rate between 0 and 1.

    Input shape:
        (batch_size, input_size, n_series)

    Output shape:
        (batch_size, input_size, n_series)

    Example:
        >>> layer = TemporalMixing(n_series=7, input_size=96, dropout=0.1)
        >>> x = keras.random.normal((32, 96, 7))
        >>> output = layer(x)
        >>> output.shape
        (32, 96, 7)
    """

    def __init__(
        self,
        n_series: int,
        input_size: int,
        dropout: float,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the TemporalMixing layer.

        Args:
            n_series: Number of time series.
            input_size: Length of time series.
            dropout: Dropout rate.
            name: Optional layer name.
            **kwargs: Additional keyword arguments.
        """
        # Set private attributes
        self._n_series = n_series
        self._input_size = input_size
        self._dropout = dropout

        # Validate parameters
        self._validate_params()

        # Set public attributes BEFORE calling parent's __init__
        self.n_series = self._n_series
        self.input_size = self._input_size
        self.dropout_rate = self._dropout

        # Layer components
        self.temporal_norm: layers.BatchNormalization | None = None
        self.temporal_lin: layers.Dense | None = None
        self.dropout_layer: layers.Dropout | None = None

        # Call parent's __init__ after setting public attributes
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate input parameters."""
        if not isinstance(self._n_series, int) or self._n_series <= 0:
            raise ValueError(
                f"n_series must be a positive integer, got {self._n_series}",
            )
        if not isinstance(self._input_size, int) or self._input_size <= 0:
            raise ValueError(
                f"input_size must be a positive integer, got {self._input_size}",
            )
        if not isinstance(self._dropout, int | float) or not (0 <= self._dropout <= 1):
            raise ValueError(f"dropout must be between 0 and 1, got {self._dropout}")

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Build the layer.

        Args:
            input_shape: Shape of input tensor.
        """
        self.temporal_norm = layers.BatchNormalization(
            epsilon=0.001,
            momentum=0.01,
            name="temporal_norm",
        )
        self.temporal_lin = layers.Dense(
            self.input_size,
            activation=None,
            name="temporal_lin",
        )
        self.dropout_layer = layers.Dropout(self.dropout_rate, name="temporal_dropout")

        super().build(input_shape)

    def call(self, inputs: KerasTensor, training: bool | None = None) -> KerasTensor:
        """Apply temporal mixing.

        Args:
            inputs: Input tensor of shape (batch_size, input_size, n_series).
            training: Whether in training mode.

        Returns:
            Output tensor of shape (batch_size, input_size, n_series).
        """
        # Get shapes
        batch_size = ops.shape(inputs)[0]
        input_size = ops.shape(inputs)[1]
        n_series = ops.shape(inputs)[2]

        # Temporal MLP: mix across time dimension
        # [B, L, N] -> [B, N, L]
        x = ops.transpose(inputs, (0, 2, 1))

        # Reshape to apply batch norm across flattened dimensions
        # [B, N, L] -> [B, N * L]
        x = ops.reshape(x, (batch_size, n_series * input_size))

        # Apply batch normalization
        # [B, N * L] -> [B, N * L]
        if self.temporal_norm is None:
            raise RuntimeError("Layer must be built before calling")
        x = self.temporal_norm(x, training=training)

        # Reshape back
        # [B, N * L] -> [B, N, L]
        x = ops.reshape(x, (batch_size, n_series, input_size))

        # Apply linear transformation across time
        # [B, N, L] -> [B, N, L]
        if self.temporal_lin is None:
            raise RuntimeError("Layer must be built before calling")
        x = self.temporal_lin(x)
        x = ops.relu(x)

        # Transpose back to original format
        # [B, N, L] -> [B, L, N]
        x = ops.transpose(x, (0, 2, 1))

        # Apply dropout
        if self.dropout_layer is None:
            raise RuntimeError("Layer must be built before calling")
        x = self.dropout_layer(x, training=training)

        # Residual connection
        return x + inputs

    def get_config(self) -> dict[str, Any]:
        """Return layer configuration."""
        config = super().get_config()
        config.update(
            {
                "n_series": self.n_series,
                "input_size": self.input_size,
                "dropout": self.dropout_rate,
            },
        )
        return config


if __name__ == "__main__":
    # Simple test
    layer = TemporalMixing(n_series=7, input_size=96, dropout=0.1)
    x = keras.random.normal((32, 96, 7))
    output = layer(x)
    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Output shape: {output.shape}")
    logger.info("TemporalMixing test passed!")
