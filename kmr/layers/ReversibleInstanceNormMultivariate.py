"""Multivariate Reversible Instance Normalization layer."""

from typing import Any
from keras import ops
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class ReversibleInstanceNormMultivariate(BaseLayer):
    """Reversible Instance Normalization for multivariate time series.

    Normalizes each series independently across the time dimension,
    enabling reversible denormalization. Designed for multivariate data.

    Args:
        num_features: Number of features/channels.
        eps: Small value for numerical stability (default: 1e-5).
        affine: Whether to use learnable scale and shift (default: False).
        name: Optional name for the layer.

    Example:
        ```python
        import keras
        from kmr.layers import ReversibleInstanceNormMultivariate

        # Create normalization layer
        revin = ReversibleInstanceNormMultivariate(num_features=8)

        # Normalize
        x = keras.random.normal((32, 100, 8))
        x_norm = revin(x)

        # Denormalize
        x_denorm = revin(x_norm, mode='denorm')
        ```
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = False,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the ReversibleInstanceNormMultivariate layer.

        Args:
            num_features: Number of features.
            eps: Epsilon for numerical stability.
            affine: Whether to use learnable affine transformation.
            name: Optional layer name.
            **kwargs: Additional keyword arguments.
        """
        # Set private attributes
        self._num_features = num_features
        self._eps = eps
        self._affine = affine

        # Validate parameters
        self._validate_params()

        # Set public attributes BEFORE calling parent's __init__
        self.num_features = self._num_features
        self.eps = self._eps
        self.affine = self._affine

        # State for normalization
        self.batch_mean = None
        self.batch_std = None

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
                shape=(1, 1, self.num_features),
                initializer="ones",
                trainable=True,
            )
            self.affine_bias = self.add_weight(
                name="affine_bias",
                shape=(1, 1, self.num_features),
                initializer="zeros",
                trainable=True,
            )

        super().build(input_shape)

    def call(
        self,
        inputs: KerasTensor,
        mode: str = "norm",
    ) -> KerasTensor:
        """Normalize or denormalize the input.

        Args:
            inputs: Input tensor of shape (batch, time, features).
            mode: Either 'norm' for normalization or 'denorm' for denormalization.

        Returns:
            Normalized or denormalized tensor.
        """
        if mode == "norm":
            return self._normalize(inputs)
        elif mode == "denorm":
            return self._denormalize(inputs)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'norm' or 'denorm'.")

    def _normalize(self, x: KerasTensor) -> KerasTensor:
        """Normalize the input.

        Args:
            x: Input tensor of shape (batch, time, features).

        Returns:
            Normalized tensor.
        """
        # Compute batch statistics across time
        self.batch_mean = ops.mean(x, axis=1, keepdims=True)
        self.batch_std = ops.sqrt(ops.var(x, axis=1, keepdims=True) + self.eps)

        # Instance normalization
        x = x - self.batch_mean
        x = x / self.batch_std

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x

    def _denormalize(self, x: KerasTensor) -> KerasTensor:
        """Denormalize the input.

        Args:
            x: Normalized tensor.

        Returns:
            Denormalized tensor.
        """
        if self.affine:
            x = (x - self.affine_bias) / self.affine_weight

        x = x * self.batch_std + self.batch_mean

        return x

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
            },
        )
        return config
