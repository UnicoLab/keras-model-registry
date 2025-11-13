from keras import layers, ops
from keras import KerasTensor
from typing import Any
from collections.abc import Sequence
from keras.saving import register_keras_serializable


@register_keras_serializable(package="kerasfactory.layers")
class SparseAttentionWeighting(layers.Layer):
    """Sparse attention mechanism with temperature scaling for module outputs combination.

    This layer implements a learnable attention mechanism that combines outputs from multiple
    modules using temperature-scaled attention weights. The attention weights are learned
    during training and can be made more or less sparse by adjusting the temperature parameter.
    A higher temperature leads to more uniform weights, while a lower temperature makes the
    weights more concentrated on specific modules.

    Key features:
    1. Learnable module importance weights
    2. Temperature-controlled sparsity
    3. Softmax-based attention mechanism
    4. Support for variable number of input features per module

    Example:
    ```python
    import numpy as np
    from keras import layers, Model
    from kerasfactory.layers import SparseAttentionWeighting

    # Create sample module outputs
    batch_size = 32
    num_modules = 3
    feature_dim = 64

    # Create three different module outputs
    module1 = layers.Dense(feature_dim)(inputs)
    module2 = layers.Dense(feature_dim)(inputs)
    module3 = layers.Dense(feature_dim)(inputs)

    # Combine module outputs using sparse attention
    attention = SparseAttentionWeighting(
        num_modules=num_modules,
        temperature=0.5  # Lower temperature for sharper attention
    )
    combined_output = attention([module1, module2, module3])

    # The layer will learn which modules are most important
    # and weight their outputs accordingly
    ```

    Args:
        num_modules: Number of input modules whose outputs will be combined.
        temperature: Temperature parameter for softmax scaling. Default is 1.0.
            - temperature > 1.0: More uniform attention weights
            - temperature < 1.0: More sparse attention weights
            - temperature = 1.0: Standard softmax behavior
    """

    def __init__(
        self,
        num_modules: int,
        temperature: float = 1.0,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize sparse attention weighting layer.

        Args:
            num_modules: Number of input modules to weight. Must be positive.
            temperature: Temperature parameter for softmax scaling. Must be positive.
                Controls the sparsity of attention weights:
                - Higher values (>1.0) lead to more uniform weights
                - Lower values (<1.0) lead to more concentrated weights
            **kwargs: Additional layer arguments passed to the parent Layer class.

        Raises:
            ValueError: If num_modules <= 0 or temperature <= 0
        """
        if num_modules <= 0:
            raise ValueError(f"num_modules must be positive, got {num_modules}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        super().__init__(**kwargs)
        self.num_modules = num_modules
        self.temperature = temperature

        # Learnable attention weights
        self.attention_weights = self.add_weight(
            shape=(num_modules,),
            initializer="ones",
            trainable=True,
            name="attention_weights",
        )

    def call(self, module_outputs: Sequence[KerasTensor]) -> KerasTensor:
        """Apply sparse attention weighting to combine module outputs.

        This method performs the following steps:
        1. Applies temperature scaling to the attention weights
        2. Computes softmax to get attention probabilities
        3. Stacks all module outputs
        4. Applies attention weights to combine outputs

        Args:
            module_outputs: Sequence of module output tensors, each of shape
                (..., feature_dim). The feature dimension can vary between modules.

        Returns:
            Combined tensor of shape (..., feature_dim) representing the
            attention-weighted sum of module outputs.

        Raises:
            ValueError: If len(module_outputs) != num_modules
        """
        if len(module_outputs) != self.num_modules:
            raise ValueError(
                f"Expected {self.num_modules} module outputs, but got {len(module_outputs)}",
            )

        # Temperature-scaled softmax for sharper attention
        attention_probs = ops.softmax(self.attention_weights / self.temperature)

        # Stack and weight module outputs
        stacked_outputs = ops.stack(module_outputs, axis=1)
        attention_weights = ops.expand_dims(ops.expand_dims(attention_probs, 0), -1)

        # Weighted combination of outputs
        weighted_sum = ops.sum(stacked_outputs * attention_weights, axis=1)
        return weighted_sum

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration:
            - num_modules: Number of input modules
            - temperature: Temperature scaling parameter
        """
        config = super().get_config()
        config.update(
            {
                "num_modules": self.num_modules,
                "temperature": self.temperature,
            },
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SparseAttentionWeighting":
        """Create layer from configuration.

        Args:
            config: Layer configuration dictionary

        Returns:
            SparseAttentionWeighting instance
        """
        return cls(**config)
