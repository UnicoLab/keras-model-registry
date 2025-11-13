"""TSMixer Model - MLP-based multivariate time series forecasting."""

from typing import Any

import keras
from keras import KerasTensor, layers, ops
from keras.saving import register_keras_serializable
from loguru import logger

from kerasfactory.layers.MixingLayer import MixingLayer
from kerasfactory.layers.ReversibleInstanceNormMultivariate import (
    ReversibleInstanceNormMultivariate,
)
from kerasfactory.models._base import BaseModel


@register_keras_serializable(package="kerasfactory.models")
class TSMixer(BaseModel):
    """TSMixer: MLP-based Multivariate Time Series Forecasting.

    Time-Series Mixer (TSMixer) is an MLP-based multivariate time-series
    forecasting model that jointly learns temporal and cross-sectional
    representations by repeatedly combining time- and feature information
    using stacked mixing layers.

    A mixing layer consists of sequential temporal and feature MLPs that
    process time series data in a straightforward manner without complex
    architectures like attention mechanisms.

    Args:
        seq_len: Sequence length (number of lookback steps).
        pred_len: Prediction length (forecast horizon).
        n_features: Number of features/time series.
        n_blocks: Number of mixing layers in the model.
        ff_dim: Hidden dimension for feed-forward networks in feature mixing.
        dropout: Dropout rate between 0 and 1.
        use_norm: If True, uses Reversible Instance Normalization.
        norm_affine: If True, uses learnable affine transformation in normalization.

    Input shape:
        (batch_size, seq_len, n_features)

    Output shape:
        (batch_size, pred_len, n_features)

    Example:
        >>> model = TSMixer(
        ...     seq_len=96,
        ...     pred_len=12,
        ...     n_features=7,
        ...     n_blocks=2,
        ...     ff_dim=64,
        ...     dropout=0.1
        ... )
        >>> model.compile(optimizer='adam', loss='mse')
        >>> x = keras.random.normal((32, 96, 7))
        >>> y = model(x)
        >>> y.shape
        (32, 12, 7)

    References:
        Chen, Si-An, Chun-Liang Li, Nate Yoder, Sercan O. Arik, and
        Tomas Pfister (2023). "TSMixer: An All-MLP Architecture for
        Time Series Forecasting." arXiv preprint arXiv:2303.06053.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_features: int,
        n_blocks: int = 2,
        ff_dim: int = 64,
        dropout: float = 0.1,
        use_norm: bool = True,
        norm_affine: bool = False,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the TSMixer model.

        Args:
            seq_len: Sequence length.
            pred_len: Prediction length.
            n_features: Number of features.
            n_blocks: Number of mixing layers.
            ff_dim: Feed-forward hidden dimension.
            dropout: Dropout rate.
            use_norm: Whether to use instance normalization.
            norm_affine: Whether to use learnable affine transformation in normalization.
            name: Optional model name.
            **kwargs: Additional keyword arguments.
        """
        # Set private attributes
        self._seq_len = seq_len
        self._pred_len = pred_len
        self._n_features = n_features
        self._n_blocks = n_blocks
        self._ff_dim = ff_dim
        self._dropout = dropout
        self._use_norm = use_norm
        self._norm_affine = norm_affine

        # Validate parameters
        self._validate_params()

        # Set public attributes BEFORE calling parent's __init__
        self.seq_len = self._seq_len
        self.pred_len = self._pred_len
        self.n_features = self._n_features
        self.n_blocks = self._n_blocks
        self.ff_dim = self._ff_dim
        self.dropout_rate = self._dropout
        self.use_norm = self._use_norm
        self.norm_affine = self._norm_affine

        # Model components
        self.norm_layer: ReversibleInstanceNormMultivariate | None = None
        self.mixing_layers: list[MixingLayer] | None = None
        self.output_layer: layers.Dense | None = None

        # Call parent's __init__ after setting public attributes
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate input parameters."""
        if not isinstance(self._seq_len, int) or self._seq_len <= 0:
            raise ValueError(f"seq_len must be a positive integer, got {self._seq_len}")
        if not isinstance(self._pred_len, int) or self._pred_len <= 0:
            raise ValueError(
                f"pred_len must be a positive integer, got {self._pred_len}",
            )
        if not isinstance(self._n_features, int) or self._n_features <= 0:
            raise ValueError(
                f"n_features must be a positive integer, got {self._n_features}",
            )
        if not isinstance(self._n_blocks, int) or self._n_blocks <= 0:
            raise ValueError(
                f"n_blocks must be a positive integer, got {self._n_blocks}",
            )
        if not isinstance(self._ff_dim, int) or self._ff_dim <= 0:
            raise ValueError(f"ff_dim must be a positive integer, got {self._ff_dim}")
        if not isinstance(self._dropout, int | float) or not (0 <= self._dropout <= 1):
            raise ValueError(f"dropout must be between 0 and 1, got {self._dropout}")
        if not isinstance(self._use_norm, bool):
            raise ValueError(f"use_norm must be a boolean, got {self._use_norm}")
        if not isinstance(self._norm_affine, bool):
            raise ValueError(f"norm_affine must be a boolean, got {self._norm_affine}")

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Build the model.

        Args:
            input_shape: Shape of input tensor.
        """
        # Instance normalization layer
        if self.use_norm:
            self.norm_layer = ReversibleInstanceNormMultivariate(
                num_features=self.n_features,
                affine=self.norm_affine,
                name="instance_norm",
            )

        # Stacked mixing layers
        self.mixing_layers = []
        for i in range(self.n_blocks):
            self.mixing_layers.append(
                MixingLayer(
                    n_series=self.n_features,
                    input_size=self.seq_len,
                    dropout=self.dropout_rate,
                    ff_dim=self.ff_dim,
                    name=f"mixing_layer_{i}",
                ),
            )

        # Output projection layer
        self.output_layer = layers.Dense(
            self.pred_len,
            activation=None,
            name="output_projection",
        )

        super().build(input_shape)

    def call(self, inputs: KerasTensor, training: bool | None = None) -> KerasTensor:
        """Forward pass of the model.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len, n_features).
            training: Whether in training mode.

        Returns:
            Output tensor of shape (batch_size, pred_len, n_features).
        """
        # inputs shape: [B, L, N]
        batch_size = ops.shape(inputs)[0]

        # Apply instance normalization if enabled
        x = inputs
        if self.use_norm:
            if self.norm_layer is None:
                raise RuntimeError("Layer must be built before calling")
            x = self.norm_layer(x, training=training, mode="norm")

        # Apply stacked mixing layers
        if self.mixing_layers is None:
            raise RuntimeError("Layer must be built before calling")
        for mixing_layer in self.mixing_layers:
            x = mixing_layer(x, training=training)

        # Project temporal dimension to prediction length
        # [B, L, N] -> [B, N, L]
        x = ops.transpose(x, (0, 2, 1))

        # Apply output layer: [B, N, L] -> [B, N, pred_len]
        if self.output_layer is None:
            raise RuntimeError("Layer must be built before calling")
        x = self.output_layer(x)

        # [B, N, pred_len] -> [B, pred_len, N]
        x = ops.transpose(x, (0, 2, 1))

        # Reverse instance normalization if enabled
        if self.use_norm:
            if self.norm_layer is None:
                raise RuntimeError("Layer must be built before calling")
            x = self.norm_layer(x, training=training, mode="denorm")

        return x

    def get_config(self) -> dict[str, Any]:
        """Return model configuration."""
        config = super().get_config()
        config.update(
            {
                "seq_len": self.seq_len,
                "pred_len": self.pred_len,
                "n_features": self.n_features,
                "n_blocks": self.n_blocks,
                "ff_dim": self.ff_dim,
                "dropout": self.dropout_rate,
                "use_norm": self.use_norm,
                "norm_affine": self.norm_affine,
            },
        )
        return config

    def summary_info(self) -> dict[str, Any]:
        """Get model summary information, automatically building if needed.

        This method ensures the model is built before accessing parameter counts.

        Returns:
            A dictionary containing model information:
            - total_params: Total number of parameters
            - trainable_params: Number of trainable parameters
            - non_trainable_params: Number of non-trainable parameters
            - config: Model configuration dictionary

        Example:
            >>> model = TSMixer(seq_len=96, pred_len=12, n_features=5)
            >>> info = model.summary_info()
            >>> print(f"Total params: {info['total_params']:,}")
        """
        # Build the model if not already built
        if not self.built:
            self.build((None, self.seq_len, self.n_features))

        return {
            "total_params": self.count_params(),
            "trainable_params": sum(keras.ops.size(w) for w in self.trainable_weights),
            "non_trainable_params": sum(
                keras.ops.size(w) for w in self.non_trainable_weights
            ),
            "config": self.get_config(),
        }


if __name__ == "__main__":
    # Simple test
    model = TSMixer(
        seq_len=96,
        pred_len=12,
        n_features=7,
        n_blocks=2,
        ff_dim=64,
        dropout=0.1,
    )
    model.compile(optimizer="adam", loss="mse")

    # Get model info (automatically builds the model)
    info = model.summary_info()
    logger.info(f"✅ Model built with {info['total_params']:,} parameters")
    logger.info(f"   - Trainable: {info['trainable_params']:,}")
    logger.info(f"   - Non-trainable: {info['non_trainable_params']:,}")

    x = keras.random.normal((32, 96, 7))
    y = model(x, training=True)
    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Output shape: {y.shape}")
    logger.info("✅ TSMixer model test passed!")
