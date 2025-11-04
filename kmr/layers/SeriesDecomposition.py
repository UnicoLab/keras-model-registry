"""Series Decomposition layer for time series trend-seasonal separation."""

from typing import Any
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kmr.layers._base_layer import BaseLayer
from kmr.layers.MovingAverage import MovingAverage


@register_keras_serializable(package="kmr.layers")
class SeriesDecomposition(BaseLayer):
    """Decomposes time series into trend and seasonal components.

    Uses moving average to extract the trend component, then computes
    seasonal as the residual (input - trend).

    Args:
        kernel_size: Size of the moving average window.
        name: Optional name for the layer.

    Input shape:
        (batch_size, time_steps, channels)

    Output shape:
        - seasonal: (batch_size, time_steps, channels)
        - trend: (batch_size, time_steps, channels)

    Example:
        ```python
        import keras
        from kmr.layers import SeriesDecomposition

        # Create sample time series
        x = keras.random.normal((32, 100, 8))

        # Decompose into trend and seasonal
        decomp = SeriesDecomposition(kernel_size=25)
        seasonal, trend = decomp(x)

        print(f"Seasonal shape: {seasonal.shape}")  # (32, 100, 8)
        print(f"Trend shape: {trend.shape}")        # (32, 100, 8)
        ```
    """

    def __init__(
        self,
        kernel_size: int,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the SeriesDecomposition layer.

        Args:
            kernel_size: Size of the moving average window.
            name: Optional layer name.
            **kwargs: Additional keyword arguments.
        """
        # Set private attributes
        self._kernel_size = kernel_size

        # Validate parameters
        self._validate_params()

        # Set public attributes BEFORE calling parent's __init__
        self.kernel_size = self._kernel_size
        self.moving_avg = None

        # Call parent's __init__ after setting public attributes
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if not isinstance(self._kernel_size, int) or self._kernel_size <= 0:
            raise ValueError(
                f"kernel_size must be a positive integer, got {self._kernel_size}",
            )

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Build the layer.

        Args:
            input_shape: Shape of input tensor.
        """
        self.moving_avg = MovingAverage(kernel_size=self.kernel_size)
        super().build(input_shape)

    def call(self, inputs: KerasTensor) -> tuple[KerasTensor, KerasTensor]:
        """Decompose input into seasonal and trend components.

        Args:
            inputs: Input tensor of shape (batch, time, channels).

        Returns:
            Tuple of (seasonal, trend) tensors.
        """
        # Extract trend using moving average
        trend = self.moving_avg(inputs)

        # Seasonal is the residual
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
                "kernel_size": self.kernel_size,
            },
        )
        return config
