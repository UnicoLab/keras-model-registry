"""Moving Average layer for time series trend extraction."""

from typing import Any
from keras import ops
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kerasfactory.layers._base_layer import BaseLayer


@register_keras_serializable(package="kerasfactory.layers")
class MovingAverage(BaseLayer):
    """Extracts the trend component using moving average.

    This layer computes a moving average over time series to extract the trend
    component. It applies padding at both ends to maintain the temporal dimension.

    Args:
        kernel_size: Size of the moving average window.
        name: Optional name for the layer.

    Input shape:
        (batch_size, time_steps, channels)

    Output shape:
        (batch_size, time_steps, channels)

    Example:
        ```python
        import keras
        from kerasfactory.layers import MovingAverage

        # Create sample time series data
        x = keras.random.normal((32, 100, 8))  # 32 samples, 100 time steps, 8 features

        # Apply moving average
        moving_avg = MovingAverage(kernel_size=25)
        trend = moving_avg(x)
        print("Trend shape:", trend.shape)  # (32, 100, 8)
        ```
    """

    def __init__(
        self,
        kernel_size: int,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the MovingAverage layer.

        Args:
            kernel_size: Size of the moving average kernel.
            name: Optional layer name.
            **kwargs: Additional keyword arguments.
        """
        # Set private attributes first
        self._kernel_size = kernel_size

        # Validate parameters
        self._validate_params()

        # Set public attributes BEFORE calling parent's __init__
        self.kernel_size = self._kernel_size

        # Call parent's __init__ after setting public attributes
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if not isinstance(self._kernel_size, int) or self._kernel_size <= 0:
            raise ValueError(
                f"kernel_size must be a positive integer, got {self._kernel_size}",
            )

    def call(self, inputs: KerasTensor) -> KerasTensor:
        """Compute moving average of the input.

        Args:
            inputs: Input tensor of shape (batch, time, channels).

        Returns:
            Moving average tensor of same shape as input.
        """
        # inputs shape: (batch, time, channels)
        time_steps = ops.shape(inputs)[1]

        # Compute padding size
        padding_size = (self.kernel_size - 1) // 2

        # Pad the front and end with the first and last values
        # Repeat the first value padding_size times
        front = ops.repeat(inputs[:, 0:1, :], padding_size, axis=1)

        # Repeat the last value padding_size times
        end = ops.repeat(inputs[:, -1:, :], padding_size, axis=1)

        # Concatenate: front + input + end
        padded = ops.concatenate([front, inputs, end], axis=1)

        # Apply moving average using window operations
        result = []
        for t in range(time_steps):
            window_start = t
            window_end = t + self.kernel_size
            window = padded[
                :,
                window_start:window_end,
                :,
            ]  # (batch, kernel_size, channels)
            avg = ops.mean(window, axis=1)  # (batch, channels)
            result.append(avg)

        output = ops.stack(result, axis=1)  # (batch, time, channels)

        return output

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
