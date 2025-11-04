"""Feature Mixing layer for TSMixer model."""

from typing import Any

import keras
from keras import KerasTensor, layers, ops
from keras.saving import register_keras_serializable
from loguru import logger

from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class FeatureMixing(BaseLayer):
    """Feature mixing layer using MLP on channel dimension.

    Applies batch normalization and feed-forward network across the feature
    (channel) dimension to mix information between different time series
    while preserving temporal structure.

    Args:
        n_series: Number of time series (channels/features).
        input_size: Length of the time series (sequence length).
        dropout: Dropout rate between 0 and 1.
        ff_dim: Dimension of the hidden layer in the feed-forward network.

    Input shape:
        (batch_size, input_size, n_series)

    Output shape:
        (batch_size, input_size, n_series)

    Example:
        >>> layer = FeatureMixing(n_series=7, input_size=96, dropout=0.1, ff_dim=64)
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
        ff_dim: int,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the FeatureMixing layer.

        Args:
            n_series: Number of time series.
            input_size: Length of time series.
            dropout: Dropout rate.
            ff_dim: Feed-forward hidden dimension.
            name: Optional layer name.
            **kwargs: Additional keyword arguments.
        """
        # Set private attributes
        self._n_series = n_series
        self._input_size = input_size
        self._dropout = dropout
        self._ff_dim = ff_dim

        # Validate parameters
        self._validate_params()

        # Set public attributes BEFORE calling parent's __init__
        self.n_series = self._n_series
        self.input_size = self._input_size
        self.dropout_rate = self._dropout
        self.ff_dim = self._ff_dim

        # Layer components
        self.feature_norm = None
        self.feature_lin_1 = None
        self.feature_lin_2 = None
        self.dropout_layer_1 = None
        self.dropout_layer_2 = None

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
        if not isinstance(self._ff_dim, int) or self._ff_dim <= 0:
            raise ValueError(f"ff_dim must be a positive integer, got {self._ff_dim}")

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Build the layer.

        Args:
            input_shape: Shape of input tensor.
        """
        self.feature_norm = layers.BatchNormalization(
            epsilon=0.001,
            momentum=0.01,
            name="feature_norm",
        )
        self.feature_lin_1 = layers.Dense(
            self.ff_dim,
            activation=None,
            name="feature_lin_1",
        )
        self.feature_lin_2 = layers.Dense(
            self.n_series,
            activation=None,
            name="feature_lin_2",
        )
        self.dropout_layer_1 = layers.Dropout(
            self.dropout_rate,
            name="feature_dropout_1",
        )
        self.dropout_layer_2 = layers.Dropout(
            self.dropout_rate,
            name="feature_dropout_2",
        )

        super().build(input_shape)

    def call(self, inputs: KerasTensor, training: bool | None = None) -> KerasTensor:
        """Apply feature mixing.

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

        # Feature MLP: mix across channel dimension
        # [B, L, N] -> [B, L * N]
        x = ops.reshape(inputs, (batch_size, input_size * n_series))

        # Apply batch normalization
        # [B, L * N] -> [B, L * N]
        x = self.feature_norm(x, training=training)

        # Reshape back
        # [B, L * N] -> [B, L, N]
        x = ops.reshape(x, (batch_size, input_size, n_series))

        # First linear layer with ReLU
        # [B, L, N] -> [B, L, ff_dim]
        x = self.feature_lin_1(x)
        x = ops.relu(x)
        x = self.dropout_layer_1(x, training=training)

        # Second linear layer
        # [B, L, ff_dim] -> [B, L, N]
        x = self.feature_lin_2(x)
        x = self.dropout_layer_2(x, training=training)

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
                "ff_dim": self.ff_dim,
            },
        )
        return config


if __name__ == "__main__":
    # Simple test
    layer = FeatureMixing(n_series=7, input_size=96, dropout=0.1, ff_dim=64)
    x = keras.random.normal((32, 96, 7))
    output = layer(x)
    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Output shape: {output.shape}")
    logger.info("FeatureMixing test passed!")
