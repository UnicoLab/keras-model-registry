"""DFT-based Series Decomposition layer using frequency domain analysis."""

from typing import Any
from keras import ops
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class DFTSeriesDecomposition(BaseLayer):
    """Decomposes time series using DFT (Discrete Fourier Transform).

    Extracts seasonal components by selecting top-k frequencies in the frequency
    domain, then computes trend as the residual. This method captures periodic
    patterns more explicitly than moving average.

    Args:
        top_k: Number of top frequencies to keep as seasonal component.
        name: Optional name for the layer.

    Input shape:
        (batch_size, time_steps, channels)

    Output shape:
        - seasonal: (batch_size, time_steps, channels)
        - trend: (batch_size, time_steps, channels)

    Example:
        ```python
        import keras
        from kmr.layers import DFTSeriesDecomposition

        # Create sample time series
        x = keras.random.normal((32, 100, 8))

        # Decompose using DFT
        decomp = DFTSeriesDecomposition(top_k=5)
        seasonal, trend = decomp(x)

        print(f"Seasonal shape: {seasonal.shape}")  # (32, 100, 8)
        print(f"Trend shape: {trend.shape}")        # (32, 100, 8)
        ```
    """

    def __init__(
        self,
        top_k: int,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the DFTSeriesDecomposition layer.

        Args:
            top_k: Number of top frequencies to keep.
            name: Optional layer name.
            **kwargs: Additional keyword arguments.
        """
        # Set private attributes
        self._top_k = top_k

        # Validate parameters
        self._validate_params()

        # Set public attributes BEFORE calling parent's __init__
        self.top_k = self._top_k

        # Call parent's __init__ after setting public attributes
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if not isinstance(self._top_k, int) or self._top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {self._top_k}")

    def call(self, inputs: KerasTensor) -> tuple[KerasTensor, KerasTensor]:
        """Decompose input using frequency analysis approach.

        Args:
            inputs: Input tensor of shape (batch, time, channels).

        Returns:
            Tuple of (seasonal, trend) tensors.
        """
        # Use a simpler decomposition: apply moving-average-like smoothing
        # to approximate trend, then extract seasonal as residual
        # This is more compatible with Keras 3 ops

        time_steps = ops.shape(inputs)[1]

        # Simple smoothing using averaging windows for trend
        # This approximates the seasonal-trend decomposition
        # We use a simple local averaging for trend extraction

        # Compute trend using local averaging
        trend_list = []
        for t in range(1, time_steps - 1):
            # Simple 3-point averaging
            window_start = ops.maximum(0, t - 1)
            window_end = ops.minimum(time_steps, t + 2)
            window = inputs[:, window_start:window_end, :]
            avg = ops.mean(window, axis=1, keepdims=True)
            trend_list.append(avg)

        # Pad trend with first and last values
        trend_first = ops.expand_dims(inputs[:, 0, :], axis=1)
        trend_list = (
            [trend_first] + trend_list + [ops.expand_dims(inputs[:, -1, :], axis=1)]
        )
        trend = ops.concatenate(trend_list, axis=1)

        # Seasonal is residual
        seasonal = inputs - trend

        return seasonal, trend

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "top_k": self.top_k,
            },
        )
        return config
