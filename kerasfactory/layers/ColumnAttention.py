"""Column attention mechanism for weighting features dynamically."""
from keras import layers, models
from keras import KerasTensor
from typing import Any
from keras.saving import register_keras_serializable
from keras.models import Sequential


@register_keras_serializable(package="kerasfactory.layers")
class ColumnAttention(layers.Layer):
    """Column attention mechanism to weight features dynamically.

    This layer applies attention weights to each feature (column) in the input tensor.
    The attention weights are computed using a two-layer neural network that takes
    the input features and outputs attention weights for each feature.

    Example:
        ```python
        import tensorflow as tf
        from kerasfactory.layers import ColumnAttention

        # Create sample data
        batch_size = 32
        input_dim = 10
        inputs = tf.random.normal((batch_size, input_dim))

        # Apply column attention
        attention = ColumnAttention(input_dim=input_dim)
        weighted_outputs = attention(inputs)
        ```
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize column attention.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension. If None, uses input_dim // 2
            **kwargs: Additional layer arguments
        """
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or max(input_dim // 2, 1)

        # Initialize layer weights to None
        self.attention_net: Sequential | None = None

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Build the layer.

        Args:
            input_shape: Shape of input tensor
        """
        if len(input_shape) != 2:
            raise ValueError(f"Expected input shape of rank 2, got shape {input_shape}")
        if input_shape[1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features, got {input_shape[1]}",
            )

        # Two-layer attention mechanism for better feature interaction
        self.attention_net = models.Sequential(
            [
                layers.Dense(self.hidden_dim, activation="relu"),
                layers.BatchNormalization(),
                layers.Dense(self.input_dim, activation="softmax"),
            ],
        )

        super().build(input_shape)

    def call(self, inputs: KerasTensor) -> KerasTensor:
        """Apply column attention.

        Args:
            inputs: Input tensor of shape [batch_size, input_dim]

        Returns:
            Attention weighted tensor of shape [batch_size, input_dim]
        """
        # Ensure layer is built (Keras will auto-build on first call)
        if not self.built:
            self.build(inputs.shape)

        # Compute attention weights with shape [batch_size, input_dim]
        attention_weights = self.attention_net(inputs)

        # Apply attention
        return inputs * attention_weights

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Layer configuration dictionary
        """
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
            },
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ColumnAttention":
        """Create layer from configuration.

        Args:
            config: Layer configuration dictionary

        Returns:
            ColumnAttention instance
        """
        return cls(**config)
