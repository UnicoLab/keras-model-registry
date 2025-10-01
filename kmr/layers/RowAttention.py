"""Row attention mechanism for weighting samples in a batch."""
from keras import layers, models, ops
from keras import KerasTensor
from typing import Any
from keras.saving import register_keras_serializable


@register_keras_serializable(package="kmr.layers")
class RowAttention(layers.Layer):
    """Row attention mechanism to weight samples dynamically.

    This layer applies attention weights to each sample (row) in the input tensor.
    The attention weights are computed using a two-layer neural network that takes
    each sample as input and outputs a scalar attention weight.

    Example:
        ```python
        import tensorflow as tf
        from kmr.layers import RowAttention

        # Create sample data
        batch_size = 32
        feature_dim = 10
        inputs = tf.random.normal((batch_size, feature_dim))

        # Apply row attention
        attention = RowAttention(feature_dim=feature_dim)
        weighted_outputs = attention(inputs)
        ```
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize row attention.

        Args:
            feature_dim: Number of input features
            hidden_dim: Hidden layer dimension. If None, uses feature_dim // 2
            **kwargs: Additional layer arguments
        """
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim or max(feature_dim // 2, 1)

        # Two-layer attention mechanism
        self.attention_net = models.Sequential(
            [
                layers.Dense(self.hidden_dim, activation="relu"),
                layers.BatchNormalization(),
                layers.Dense(1, activation="sigmoid"),
            ],
        )

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Build the layer.

        Args:
            input_shape: Shape of input tensor
        """
        if len(input_shape) != 2:
            raise ValueError(f"Expected input shape of rank 2, got shape {input_shape}")
        if input_shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected {self.feature_dim} features, got {input_shape[1]}",
            )
        super().build(input_shape)

    def call(self, inputs: KerasTensor) -> KerasTensor:
        """Apply row attention.

        Args:
            inputs: Input tensor of shape [batch_size, feature_dim]

        Returns:
            Attention weighted tensor of shape [batch_size, feature_dim]
        """
        # Compute attention weights [batch_size, 1]
        attention_weights = self.attention_net(inputs)

        # Use softmax to normalize weights across batch dimension
        # This will give more emphasis to higher values (dominant samples)
        # First, we need to reshape to remove the last dimension for softmax
        attention_flat = ops.reshape(attention_weights, (-1,))
        normalized_attention = ops.softmax(
            attention_flat * 5.0,
        )  # Scale factor to make softmax more peaked
        attention_weights = ops.reshape(
            normalized_attention,
            ops.shape(attention_weights),
        )

        # Apply attention weights
        return inputs * attention_weights

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Layer configuration dictionary
        """
        config = super().get_config()
        config.update(
            {
                "feature_dim": self.feature_dim,
                "hidden_dim": self.hidden_dim,
            },
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RowAttention":
        """Create layer from configuration.

        Args:
            config: Layer configuration dictionary

        Returns:
            RowAttention instance
        """
        return cls(**config)
