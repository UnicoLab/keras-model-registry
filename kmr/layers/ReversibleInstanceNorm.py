"""Reversible Instance Normalization layer for time series."""

from typing import Any
from keras import ops
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class ReversibleInstanceNorm(BaseLayer):
    """Reversible Instance Normalization (RevIN) for time series.

    Normalizes each series independently and enables reversible denormalization.
    This is useful for improving model performance by removing distributional shifts.

    Args:
        num_features: Number of features/channels.
        eps: Small value for numerical stability (default: 1e-5).
        affine: Whether to use learnable scale and shift (default: False).
        subtract_last: If True, normalize by last value instead of mean (default: False).
        non_norm: If True, no normalization is applied (default: False).
        name: Optional name for the layer.

    Example:
        ```python
        import keras
        from kmr.layers import ReversibleInstanceNorm

        # Create normalization layer
        revin = ReversibleInstanceNorm(num_features=8)

        # Normalize
        x = keras.random.normal((32, 100, 8))
        x_norm = revin(x, training=True)

        # Denormalize
        x_denorm = revin(x_norm, mode='denorm')
        ```
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = False,
        subtract_last: bool = False,
        non_norm: bool = False,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the ReversibleInstanceNorm layer.

        Args:
            num_features: Number of features.
            eps: Epsilon for numerical stability.
            affine: Whether to use learnable affine transformation.
            subtract_last: Whether to normalize by last value.
            non_norm: Whether to skip normalization.
            name: Optional layer name.
            **kwargs: Additional keyword arguments.
        """
        # Set private attributes
        self._num_features = num_features
        self._eps = eps
        self._affine = affine
        self._subtract_last = subtract_last
        self._non_norm = non_norm

        # Validate parameters
        self._validate_params()

        # Set public attributes BEFORE calling parent's __init__
        self.num_features = self._num_features
        self.eps = self._eps
        self.affine = self._affine
        self.subtract_last = self._subtract_last
        self.non_norm = self._non_norm

        # Learnable parameters
        self.affine_weight = None
        self.affine_bias = None

        # Call parent's __init__ after setting public attributes
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if not isinstance(self._num_features, int) or self._num_features <= 0:
            raise ValueError(
                f"num_features must be a positive integer, got {self._num_features}",
            )
        if not isinstance(self._eps, int | float) or self._eps < 0:
            raise ValueError(f"eps must be non-negative, got {self._eps}")

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Build the layer.

        Args:
            input_shape: Shape of input tensor.
        """
        if self.affine:
            self.affine_weight = self.add_weight(
                name="affine_weight",
                shape=(self.num_features,),
                initializer="ones",
                trainable=True,
            )
            self.affine_bias = self.add_weight(
                name="affine_bias",
                shape=(self.num_features,),
                initializer="zeros",
                trainable=True,
            )

        super().build(input_shape)

    def _normalize(
        self,
        x: KerasTensor,
    ) -> tuple[KerasTensor, KerasTensor, KerasTensor]:
        """Normalize the input and return statistics for denormalization.

        Args:
            x: Input tensor of shape (batch, time, features).

        Returns:
            Tuple of (normalized_tensor, mean, stdev) or (tensor, last_value, stdev) depending on mode.
        """
        if self.non_norm:
            return x, None, None

        # Get statistics
        dim2reduce = tuple(range(1, len(x.shape) - 1))  # Reduce over time dimension

        if self.subtract_last:
            last = ops.expand_dims(x[:, -1, :], axis=1)
            x_norm = x - last
        else:
            mean = ops.mean(x, axis=dim2reduce, keepdims=True)
            x_norm = x - mean

        stdev = ops.sqrt(ops.var(x_norm, axis=dim2reduce, keepdims=True) + self.eps)
        x_norm = x_norm / stdev

        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias

        if self.subtract_last:
            return x_norm, last, stdev
        else:
            return x_norm, mean, stdev

    def _denormalize(
        self,
        x: KerasTensor,
        stat1: KerasTensor,
        stat2: KerasTensor,
    ) -> KerasTensor:
        """Denormalize the input.

        Args:
            x: Normalized tensor.
            stat1: Mean or last value (depending on subtract_last mode).
            stat2: Standard deviation.

        Returns:
            Denormalized tensor.
        """
        if self.non_norm:
            return x

        if self.affine:
            x = (x - self.affine_bias) / self.affine_weight

        x = x * stat2

        # Add stat1 regardless of subtract_last setting
        x = x + stat1

        return x

    def call(
        self,
        inputs: KerasTensor,
        mode: str = "norm",
    ) -> KerasTensor:
        """Normalize or denormalize the input.

        Args:
            inputs: Input tensor of shape (batch, time, features).
            training: Whether in training mode (for normalization).
            mode: Either 'norm' for normalization or 'denorm' for denormalization.

        Returns:
            Normalized or denormalized tensor.
        """
        if mode == "norm":
            normalized, stat1, stat2 = self._normalize(inputs)
            # Store statistics for later denormalization if needed
            self._stat1 = stat1
            self._stat2 = stat2
            return normalized
        elif mode == "denorm":
            return self._denormalize(inputs, self._stat1, self._stat2)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'norm' or 'denorm'.")

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "num_features": self.num_features,
                "eps": self.eps,
                "affine": self.affine,
                "subtract_last": self.subtract_last,
                "non_norm": self.non_norm,
            },
        )
        return config
