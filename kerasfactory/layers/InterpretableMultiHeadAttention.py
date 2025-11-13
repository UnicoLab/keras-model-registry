"""Interpretable Multi-Head Attention layer implementation."""
from typing import Any, ClassVar
from keras import KerasTensor
from keras import layers
from keras.saving import register_keras_serializable


@register_keras_serializable(package="kerasfactory.layers")
class InterpretableMultiHeadAttention(layers.Layer):
    """Interpretable Multi-Head Attention layer.

    This layer wraps Keras MultiHeadAttention and stores the attention scores
    for interpretability purposes. The attention scores can be accessed via
    the `attention_scores` attribute after calling the layer.

    Args:
        d_model: Size of each attention head for query, key, value.
        n_head: Number of attention heads.
        dropout_rate: Dropout probability. Default: 0.1.
        **kwargs: Additional arguments passed to MultiHeadAttention.
            Supported arguments:
            - value_dim: Size of each attention head for value.
            - use_bias: Whether to use bias. Default: True.
            - output_shape: Expected output shape. Default: None.
            - attention_axes: Axes for attention. Default: None.
            - kernel_initializer: Initializer for kernels. Default: 'glorot_uniform'.
            - bias_initializer: Initializer for biases. Default: 'zeros'.
            - kernel_regularizer: Regularizer for kernels. Default: None.
            - bias_regularizer: Regularizer for biases. Default: None.
            - activity_regularizer: Regularizer for activity. Default: None.
            - kernel_constraint: Constraint for kernels. Default: None.
            - bias_constraint: Constraint for biases. Default: None.
            - seed: Random seed for dropout. Default: None.

    Call Args:
        query: Query tensor of shape `(B, S, E)` where B is batch size,
            S is sequence length, and E is the feature dimension.
        key: Key tensor of shape `(B, S, E)`.
        value: Value tensor of shape `(B, S, E)`.
        training: Python boolean indicating whether the layer should behave in
            training mode (applying dropout) or in inference mode (no dropout).

    Returns:
        output: Attention output of shape `(B, S, E)`.

    Example:
        ```python
        d_model = 64
        n_head = 4
        seq_len = 10
        batch_size = 32

        layer = InterpretableMultiHeadAttention(
            d_model=d_model,
            n_head=n_head,
            kernel_initializer='he_normal',
            use_bias=False
        )
        query = tf.random.normal((batch_size, seq_len, d_model))
        output = layer(query, query, query)
        attention_scores = layer.attention_scores  # Access attention weights
        ```
    """

    # Valid kwargs for MultiHeadAttention
    _valid_mha_kwargs: ClassVar[set[str]] = {
        "value_dim",
        "use_bias",
        "output_shape",
        "attention_axes",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
        "seed",
    }

    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout_rate: float = 0.1,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the layer."""
        # Extract MHA-specific kwargs
        mha_kwargs = {k: v for k, v in kwargs.items() if k in self._valid_mha_kwargs}
        # Remove MHA kwargs from the kwargs passed to parent
        layer_kwargs = {
            k: v for k, v in kwargs.items() if k not in self._valid_mha_kwargs
        }

        super().__init__(**layer_kwargs)
        self.d_model = d_model
        self.n_head = n_head
        self.dropout_rate = dropout_rate
        self.mha_kwargs = mha_kwargs

        # Initialize multihead attention
        self.mha = layers.MultiHeadAttention(
            num_heads=n_head,
            key_dim=d_model,
            dropout=dropout_rate,
            **mha_kwargs,
        )
        self.attention_scores: Any | None = None

    def call(
        self,
        query: KerasTensor,
        key: KerasTensor,
        value: KerasTensor,
        training: bool = False,
    ) -> KerasTensor:
        """Forward pass of the layer.

        Args:
            query: Query tensor of shape (B, S, E).
            key: Key tensor of shape (B, S, E).
            value: Value tensor of shape (B, S, E).
            training: Whether the model is in training mode.

        Returns:
            Attention output tensor of shape (B, S, E).
        """
        attn_output, attn_scores = self.mha(
            query=query,
            key=key,
            value=value,
            return_attention_scores=True,
            training=training,
        )
        self.attention_scores = attn_scores
        return attn_output

    def get_config(self) -> dict[str, Any]:
        """Return the config dictionary for serialization."""
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "n_head": self.n_head,
                "dropout_rate": self.dropout_rate,
                **self.mha_kwargs,
            },
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "InterpretableMultiHeadAttention":
        """Create layer from configuration.

        Args:
            config: Layer configuration dictionary

        Returns:
            Layer instance
        """
        return cls(**config)
