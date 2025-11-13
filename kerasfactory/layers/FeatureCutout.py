"""Feature cutout regularization layer for neural networks."""
from keras import layers, ops, random
from keras import KerasTensor
from typing import Any
from keras.saving import register_keras_serializable


@register_keras_serializable(package="kerasfactory.layers")
class FeatureCutout(layers.Layer):
    """Feature cutout regularization layer.

    This layer randomly masks out (sets to zero) a specified fraction of features
    during training to improve model robustness and prevent overfitting. During
    inference, all features are kept intact.

    Example:
        ```python
        from keras import random
        from kerasfactory.layers import FeatureCutout

        # Create sample data
        batch_size = 32
        feature_dim = 10
        inputs = random.normal((batch_size, feature_dim))

        # Apply feature cutout
        cutout = FeatureCutout(cutout_prob=0.2)
        masked_outputs = cutout(inputs, training=True)
        ```
    """

    def __init__(
        self,
        cutout_prob: float = 0.1,
        noise_value: float = 0.0,
        seed: int | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize feature cutout.

        Args:
            cutout_prob: Probability of masking each feature
            noise_value: Value to use for masked features (default: 0.0)
            seed: Random seed for reproducibility
            **kwargs: Additional layer arguments

        Raises:
            ValueError: If cutout_prob is not in [0, 1]
        """
        super().__init__(**kwargs)

        if not 0 <= cutout_prob <= 1:
            raise ValueError(f"cutout_prob must be in [0, 1], got {cutout_prob}")

        self.cutout_prob = cutout_prob
        self.noise_value = noise_value
        self.seed = seed

        # Create random generator with fixed seed
        self._rng = random.SeedGenerator(seed) if seed else None

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Build the layer.

        Args:
            input_shape: Shape of input tensor
        """
        if len(input_shape) != 2:
            raise ValueError(f"Expected input shape of rank 2, got shape {input_shape}")
        super().build(input_shape)

    def call(
        self,
        inputs: KerasTensor,
        training: bool = False,
    ) -> KerasTensor:
        """Apply feature cutout.

        Args:
            inputs: Input tensor of shape [batch_size, feature_dim]
            training: Whether in training mode

        Returns:
            Masked tensor of shape [batch_size, feature_dim]
        """
        if not training or self.cutout_prob == 0:
            return inputs

        # Generate random mask
        mask_shape = ops.shape(inputs)
        if self._rng is not None:
            random_values = random.uniform(
                mask_shape,
                seed=self._rng,
                minval=0,
                maxval=1,
            )
        else:
            random_values = random.uniform(mask_shape, minval=0, maxval=1)

        mask = ops.cast(random_values > self.cutout_prob, inputs.dtype)

        # Apply mask with noise value
        return inputs * mask + self.noise_value * (1 - mask)

    def compute_output_shape(
        self,
        input_shape: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Compute output shape.

        Args:
            input_shape: Input shape tuple

        Returns:
            Output shape tuple
        """
        return input_shape

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Layer configuration dictionary
        """
        config = super().get_config()
        config.update(
            {
                "cutout_prob": self.cutout_prob,
                "noise_value": self.noise_value,
                "seed": self.seed,
            },
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FeatureCutout":
        """Create layer from configuration.

        Args:
            config: Layer configuration dictionary

        Returns:
            FeatureCutout instance
        """
        return cls(**config)
