"""Stochastic depth layer for neural networks."""
from keras import layers, ops, random
from keras import KerasTensor
from typing import Any
from keras.saving import register_keras_serializable


@register_keras_serializable(package="kmr.layers")
class StochasticDepth(layers.Layer):
    """Stochastic depth layer for regularization.

    This layer randomly drops entire residual branches with a specified probability
    during training. During inference, all branches are kept and scaled appropriately.
    This technique helps reduce overfitting and training time in deep networks.

    Reference:
        - [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382)

    Example:
        ```python
        from keras import random, layers
        from kmr.layers import StochasticDepth

        # Create sample residual branch
        inputs = random.normal((32, 64, 64, 128))
        residual = layers.Conv2D(128, 3, padding="same")(inputs)
        residual = layers.BatchNormalization()(residual)
        residual = layers.ReLU()(residual)

        # Apply stochastic depth
        outputs = StochasticDepth(survival_prob=0.8)([inputs, residual])
        ```
    """

    def __init__(
        self,
        survival_prob: float = 0.5,
        seed: int | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize stochastic depth.

        Args:
            survival_prob: Probability of keeping the residual branch (default: 0.5)
            seed: Random seed for reproducibility
            **kwargs: Additional layer arguments

        Raises:
            ValueError: If survival_prob is not in [0, 1]
        """
        super().__init__(**kwargs)

        if not 0 <= survival_prob <= 1:
            raise ValueError(f"survival_prob must be in [0, 1], got {survival_prob}")

        self.survival_prob = survival_prob
        self.seed = seed

        # Create random generator with fixed seed
        self._rng = random.SeedGenerator(seed) if seed else None

    def call(
        self,
        inputs: list[KerasTensor],
        training: bool = False,
    ) -> KerasTensor:
        """Apply stochastic depth.

        Args:
            inputs: List of [shortcut, residual] tensors
            training: Whether in training mode

        Returns:
            Output tensor after applying stochastic depth

        Raises:
            ValueError: If inputs is not a list of length 2
        """
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError("inputs must be a list of [shortcut, residual]")

        shortcut, residual = inputs

        # During inference, scale residual by survival probability
        if not training or self.survival_prob == 1.0:
            return shortcut + self.survival_prob * residual

        # During training, randomly drop residual branch
        batch_size = ops.shape(shortcut)[0]
        random_tensor = random.uniform(
            (batch_size, 1, 1, 1) if len(shortcut.shape) == 4 else (batch_size, 1),
            seed=self._rng,
            minval=0,
            maxval=1,
        )
        binary_tensor = ops.cast(random_tensor < self.survival_prob, shortcut.dtype)

        return shortcut + binary_tensor * residual

    def compute_output_shape(
        self,
        input_shape: list[tuple[int, ...]],
    ) -> tuple[int, ...]:
        """Compute output shape.

        Args:
            input_shape: List of input shape tuples

        Returns:
            Output shape tuple
        """
        return input_shape[0]

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Layer configuration dictionary
        """
        config = super().get_config()
        config.update(
            {
                "survival_prob": self.survival_prob,
                "seed": self.seed,
            },
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "StochasticDepth":
        """Create layer from configuration.

        Args:
            config: Layer configuration dictionary

        Returns:
            StochasticDepth instance
        """
        return cls(**config)
