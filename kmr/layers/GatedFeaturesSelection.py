from keras import layers, models
from keras import KerasTensor
from typing import Any
from keras.saving import register_keras_serializable
from keras.models import Sequential


@register_keras_serializable(package="kmr.layers")
class GatedFeatureSelection(layers.Layer):
    """Gated feature selection layer with residual connection.

    This layer implements a learnable feature selection mechanism using a gating network.
    Each feature is assigned a dynamic importance weight between 0 and 1 through a multi-layer
    gating network. The gating network includes batch normalization and ReLU activations
    for stable training. A small residual connection (0.1) is added to maintain gradient flow.

    The layer is particularly useful for:
    1. Dynamic feature importance learning
    2. Feature selection in time-series data
    3. Attention-like mechanisms for tabular data
    4. Reducing noise in input features

    Example:
    ```python
    import numpy as np
    from keras import layers, Model
    from kmr.layers import GatedFeatureSelection

    # Create sample input data
    input_dim = 20
    x = np.random.normal(size=(100, input_dim))

    # Build model with gated feature selection
    inputs = layers.Input(shape=(input_dim,))
    x = GatedFeatureSelection(input_dim=input_dim, reduction_ratio=4)(inputs)
    outputs = layers.Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)

    # The layer will learn which features are most important
    # and dynamically adjust their contribution to the output
    ```

    Args:
        input_dim: Dimension of the input features
        reduction_ratio: Ratio to reduce the hidden dimension of the gating network.
            A higher ratio means fewer parameters but potentially less expressive gates.
            Default is 4, meaning the hidden dimension will be input_dim // 4.
    """

    def __init__(
        self,
        input_dim: int,
        reduction_ratio: int = 4,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the gated feature selection layer.

        Args:
            input_dim: Dimension of the input features. Must match the last dimension
                of the input tensor.
            reduction_ratio: Ratio to reduce the hidden dimension of the gating network.
                The hidden dimension will be max(input_dim // reduction_ratio, 1).
                Default is 4.
            **kwargs: Additional layer arguments passed to the parent Layer class.
        """
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.reduction_ratio = reduction_ratio
        self.gate_net: Sequential | None = None

    def build(self, input_shape: tuple) -> None:
        """Build the gating network.

        Creates a multi-layer gating network with batch normalization and ReLU
        activations. The network architecture is:
        1. Dense(hidden_dim) -> ReLU -> BatchNorm
        2. Dense(hidden_dim) -> ReLU -> BatchNorm
        3. Dense(input_dim) -> Sigmoid

        Args:
            input_shape: Shape of input tensor, expected to be (..., input_dim)

        Raises:
            ValueError: If the last dimension of input_shape doesn't match input_dim
        """
        if input_shape[-1] != self.input_dim:
            raise ValueError(
                f"Last dimension of input shape {input_shape[-1]} does not match input_dim {self.input_dim}",
            )

        # More powerful gate network with skip connection
        hidden_dim = max(self.input_dim // self.reduction_ratio, 1)
        self.gate_net = models.Sequential(
            [
                layers.Dense(hidden_dim, activation="relu"),
                layers.BatchNormalization(),
                layers.Dense(hidden_dim, activation="relu"),
                layers.BatchNormalization(),
                layers.Dense(self.input_dim, activation="sigmoid"),
            ],
        )
        self.built = True

    def call(self, inputs: KerasTensor) -> KerasTensor:
        """Apply gated feature selection with residual connection.

        The layer computes feature importance gates using the gating network
        and applies them to the input features. A small residual connection (0.1)
        is added to maintain gradient flow and prevent features from being
        completely masked.

        Args:
            inputs: Input tensor of shape (..., input_dim)

        Returns:
            Tensor of same shape as input with gated features.
            The output is computed as: inputs * gates + 0.1 * inputs
        """
        # Ensure layer is built (Keras will auto-build on first call)
        if not self.built:
            self.build(inputs.shape)

        # Compute feature gates
        gates = self.gate_net(inputs)

        # Residual connection with gating
        return inputs * gates + 0.1 * inputs  # Small residual to maintain gradient flow

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration:
            - input_dim: Dimension of input features
            - reduction_ratio: Ratio for hidden dimension reduction
        """
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "reduction_ratio": self.reduction_ratio,
            },
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GatedFeatureSelection":
        """Create layer from configuration.

        Args:
            config: Layer configuration dictionary

        Returns:
            GatedFeatureSelection instance
        """
        return cls(**config)
