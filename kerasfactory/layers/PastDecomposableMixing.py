"""Past Decomposable Mixing layer for time series encoder blocks."""

from typing import Any
from keras import layers, ops
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kerasfactory.layers._base_layer import BaseLayer
from kerasfactory.layers.SeriesDecomposition import SeriesDecomposition
from kerasfactory.layers.DFTSeriesDecomposition import DFTSeriesDecomposition
from kerasfactory.layers.MultiScaleSeasonMixing import MultiScaleSeasonMixing
from kerasfactory.layers.MultiScaleTrendMixing import MultiScaleTrendMixing


@register_keras_serializable(package="kerasfactory.layers")
class PastDecomposableMixing(BaseLayer):
    """Past Decomposable Mixing block for TimeMixer encoder.

    Decomposes time series, applies multi-scale mixing to trend and seasonal
    components, then reconstructs the signal.

    Args:
        seq_len: Sequence length.
        pred_len: Prediction length.
        down_sampling_window: Downsampling window size.
        down_sampling_layers: Number of downsampling layers.
        d_model: Model dimension.
        dropout: Dropout rate.
        channel_independence: Whether to use channel-independent processing.
        decomp_method: Decomposition method ('moving_avg' or 'dft_decomp').
        d_ff: Feed-forward dimension.
        moving_avg: Window size for moving average.
        top_k: Top-k frequencies for DFT.
        name: Optional name for the layer.

    Example:
        ```python
        import keras
        from kerasfactory.layers import PastDecomposableMixing

        # Create PDM block
        pdm = PastDecomposableMixing(seq_len=100, pred_len=12,
                                     down_sampling_window=2,
                                     down_sampling_layers=1)

        # Process multi-scale inputs
        x_list = [keras.random.normal((32, 100, 8))]
        output = pdm(x_list)
        ```
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        down_sampling_window: int,
        down_sampling_layers: int,
        d_model: int,
        dropout: float,
        channel_independence: int,
        decomp_method: str,
        d_ff: int,
        moving_avg: int,
        top_k: int,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the PastDecomposableMixing layer."""
        # Set private attributes
        self._seq_len = seq_len
        self._pred_len = pred_len
        self._down_sampling_window = down_sampling_window
        self._down_sampling_layers = down_sampling_layers
        self._d_model = d_model
        self._dropout = dropout
        self._channel_independence = channel_independence
        self._decomp_method = decomp_method
        self._d_ff = d_ff
        self._moving_avg = moving_avg
        self._top_k = top_k

        # Validate parameters
        self._validate_params()

        # Set public attributes BEFORE calling parent's __init__
        self.seq_len = self._seq_len
        self.pred_len = self._pred_len
        self.down_sampling_window = self._down_sampling_window
        self.down_sampling_layers = self._down_sampling_layers
        self.d_model = self._d_model
        self.dropout_rate = self._dropout
        self.channel_independence = self._channel_independence
        self.decomp_method = self._decomp_method
        self.d_ff = self._d_ff
        self.moving_avg_kernel = self._moving_avg
        self.top_k = self._top_k

        # Components
        self.decomposition: SeriesDecomposition | DFTSeriesDecomposition | None = None
        self.season_mixing: MultiScaleSeasonMixing | None = None
        self.trend_mixing: MultiScaleTrendMixing | None = None
        self.dense1: layers.Dense | None = None
        self.dense2: layers.Dense | None = None

        # Call parent's __init__ after setting public attributes
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if self._decomp_method not in ["moving_avg", "dft_decomp"]:
            raise ValueError(
                f"decomp_method must be 'moving_avg' or 'dft_decomp', got {self._decomp_method}",
            )
        if self._channel_independence not in [0, 1]:
            raise ValueError(
                f"channel_independence must be 0 or 1, got {self._channel_independence}",
            )

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Build the layer."""
        # Create decomposition layer
        if self.decomp_method == "moving_avg":
            self.decomposition = SeriesDecomposition(kernel_size=self.moving_avg_kernel)
        else:
            self.decomposition = DFTSeriesDecomposition(top_k=self.top_k)

        # Create multi-scale mixing layers
        self.season_mixing = MultiScaleSeasonMixing(
            seq_len=self.seq_len,
            down_sampling_window=self.down_sampling_window,
            down_sampling_layers=self.down_sampling_layers,
        )
        self.trend_mixing = MultiScaleTrendMixing(
            seq_len=self.seq_len,
            down_sampling_window=self.down_sampling_window,
            down_sampling_layers=self.down_sampling_layers,
        )

        # Cross-layer for channel dependence (using individual Dense layers)
        if self.channel_independence == 0:
            self.dense1 = layers.Dense(self.d_ff, activation="gelu")
            self.dense2 = layers.Dense(self.d_model)

        super().build(input_shape)

    def call(self, inputs: list[KerasTensor]) -> list[KerasTensor]:
        """Process multi-scale inputs through decomposition and mixing.

        Args:
            inputs: List of tensors at different scales.

        Returns:
            List of processed tensors.
        """
        season_list = []
        trend_list = []

        # Decompose each scale
        if self.decomposition is None:
            raise RuntimeError("Layer must be built before calling")
        for x in inputs:
            season, trend = self.decomposition(x)

            if self.channel_independence == 0:
                # Apply cross-layer (dense1 -> dense2)
                if self.dense1 is None:
                    raise RuntimeError("Layer must be built before calling")
                if self.dense2 is None:
                    raise RuntimeError("Layer must be built before calling")
                season = self.dense1(season)
                season = self.dense2(season)
                trend = self.dense1(trend)
                trend = self.dense2(trend)

            # Transpose for mixing: (batch, time, features) -> (batch, features, time)
            season_list.append(ops.transpose(season, (0, 2, 1)))
            trend_list.append(ops.transpose(trend, (0, 2, 1)))

        # Apply multi-scale mixing
        if self.season_mixing is None:
            raise RuntimeError("Layer must be built before calling")
        if self.trend_mixing is None:
            raise RuntimeError("Layer must be built before calling")
        out_season_list = self.season_mixing(season_list)
        out_trend_list = self.trend_mixing(trend_list)

        # Reconstruct and collect outputs
        out_list = []
        for season, trend, original_x in zip(
            out_season_list,
            out_trend_list,
            inputs,
            strict=True,
        ):
            out = season + trend

            if self.channel_independence == 1:
                # For channel independence, combine with original residual
                out = original_x + out

            out_list.append(out)

        return out_list

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "seq_len": self.seq_len,
                "pred_len": self.pred_len,
                "down_sampling_window": self.down_sampling_window,
                "down_sampling_layers": self.down_sampling_layers,
                "d_model": self.d_model,
                "dropout": self.dropout_rate,
                "channel_independence": self.channel_independence,
                "decomp_method": self.decomp_method,
                "d_ff": self.d_ff,
                "moving_avg": self.moving_avg_kernel,
                "top_k": self.top_k,
            },
        )
        return config
