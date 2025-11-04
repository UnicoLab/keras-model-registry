"""Mixing Layer combining temporal and feature mixing for TSMixer."""

from typing import Any

import keras
from keras import KerasTensor
from keras.saving import register_keras_serializable
from loguru import logger

from kmr.layers._base_layer import BaseLayer
from kmr.layers.FeatureMixing import FeatureMixing
from kmr.layers.TemporalMixing import TemporalMixing


@register_keras_serializable(package="kmr.layers")
class MixingLayer(BaseLayer):
    """Mixing layer combining temporal and feature mixing.

    A mixing layer consists of sequential temporal and feature MLPs that
    jointly learn temporal and cross-sectional representations.

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
        >>> layer = MixingLayer(n_series=7, input_size=96, dropout=0.1, ff_dim=64)
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
        """Initialize the MixingLayer.

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
        self.temporal_mixer: TemporalMixing | None = None
        self.feature_mixer: FeatureMixing | None = None

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
        self.temporal_mixer = TemporalMixing(
            n_series=self.n_series,
            input_size=self.input_size,
            dropout=self.dropout_rate,
            name="temporal_mixer",
        )
        self.feature_mixer = FeatureMixing(
            n_series=self.n_series,
            input_size=self.input_size,
            dropout=self.dropout_rate,
            ff_dim=self.ff_dim,
            name="feature_mixer",
        )

        super().build(input_shape)

    def call(self, inputs: KerasTensor, training: bool | None = None) -> KerasTensor:
        """Apply mixing.

        Args:
            inputs: Input tensor of shape (batch_size, input_size, n_series).
            training: Whether in training mode.

        Returns:
            Output tensor of shape (batch_size, input_size, n_series).
        """
        # Apply temporal mixing first
        if self.temporal_mixer is None:
            raise RuntimeError("Layer must be built before calling")
        x = self.temporal_mixer(inputs, training=training)

        # Then apply feature mixing
        if self.feature_mixer is None:
            raise RuntimeError("Layer must be built before calling")
        x = self.feature_mixer(x, training=training)

        return x

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
    layer = MixingLayer(n_series=7, input_size=96, dropout=0.1, ff_dim=64)
    x = keras.random.normal((32, 96, 7))
    output = layer(x)
    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Output shape: {output.shape}")
    logger.info("MixingLayer test passed!")
