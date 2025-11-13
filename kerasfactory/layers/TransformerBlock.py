"""This module implements a TransformerBlock layer that applies transformer-style self-attention
and feed-forward processing to input tensors. It's particularly useful for capturing complex
relationships in tabular data.
"""

from typing import Any
from loguru import logger
from keras import layers, ops
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kerasfactory.layers._base_layer import BaseLayer


@register_keras_serializable(package="kerasfactory.layers")
class TransformerBlock(BaseLayer):
    """Transformer block with multi-head attention and feed-forward layers.

    This layer implements a standard transformer block with multi-head self-attention
    followed by a feed-forward network, with residual connections and layer normalization.

    Args:
        dim_model (int): Dimensionality of the model.
        num_heads (int): Number of attention heads.
        ff_units (int): Number of units in the feed-forward network.
        dropout_rate (float): Dropout rate for regularization.
        name (str, optional): Name for the layer.

    Input shape:
        Tensor with shape: `(batch_size, sequence_length, dim_model)` or
        `(batch_size, dim_model)` which will be automatically reshaped.

    Output shape:
        Tensor with shape: `(batch_size, sequence_length, dim_model)` or
        `(batch_size, dim_model)` matching the input shape.

    Example:
        ```python
        import keras
        from kerasfactory.layers import TransformerBlock

        # Create sample input data
        x = keras.random.normal((32, 10, 64))  # 32 samples, 10 time steps, 64 features

        # Apply transformer block
        transformer = TransformerBlock(dim_model=64, num_heads=4, ff_units=128, dropout_rate=0.1)
        y = transformer(x)
        print("Output shape:", y.shape)  # (32, 10, 64)
        ```
    """

    def __init__(
        self,
        dim_model: int = 32,
        num_heads: int = 3,
        ff_units: int = 16,
        dropout_rate: float = 0.2,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the TransformerBlock layer.

        Args:
            dim_model: Model dimension.
            num_heads: Number of attention heads.
            ff_units: Feed-forward units.
            dropout_rate: Dropout rate.
            name: Name of the layer.
            **kwargs: Additional keyword arguments.
        """
        # Set private attributes first
        self._dim_model = dim_model
        self._num_heads = num_heads
        self._ff_units = ff_units
        self._dropout_rate = dropout_rate

        # Validate parameters
        self._validate_params()

        # Set public attributes BEFORE calling parent's __init__
        self.dim_model = self._dim_model
        self.num_heads = self._num_heads
        self.ff_units = self._ff_units
        self.dropout_rate = self._dropout_rate

        # Initialize layers
        self.multihead_attention: layers.MultiHeadAttention | None = None
        self.dropout1: layers.Dropout | None = None
        self.add1: layers.Add | None = None
        self.layer_norm1: layers.LayerNormalization | None = None
        self.ff1: layers.Dense | None = None
        self.dropout2: layers.Dropout | None = None
        self.ff2: layers.Dense | None = None
        self.add2: layers.Add | None = None
        self.layer_norm2: layers.LayerNormalization | None = None

        # Call parent's __init__ after setting public attributes
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if not isinstance(self._dim_model, int) or self._dim_model <= 0:
            raise ValueError(
                f"dim_model must be a positive integer, got {self._dim_model}",
            )

        if not isinstance(self._num_heads, int) or self._num_heads <= 0:
            raise ValueError(
                f"num_heads must be a positive integer, got {self._num_heads}",
            )

        if not isinstance(self._ff_units, int) or self._ff_units <= 0:
            raise ValueError(
                f"ff_units must be a positive integer, got {self._ff_units}",
            )

        if not isinstance(self._dropout_rate, float) or not 0 <= self._dropout_rate < 1:
            raise ValueError(
                f"dropout_rate must be a float between 0 and 1, got {self._dropout_rate}",
            )

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Builds the layer with the given input shape.

        Args:
            input_shape: Tuple of integers defining the input shape.
        """
        # Define layers
        self.multihead_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.dim_model // self.num_heads,
            dropout=self.dropout_rate,
        )
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.add1 = layers.Add()
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)

        self.ff1 = layers.Dense(self.ff_units, activation="relu")
        self.dropout2 = layers.Dropout(self.dropout_rate)
        self.ff2 = layers.Dense(self.dim_model)
        self.add2 = layers.Add()
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)

        logger.debug(
            f"TransformerBlock built with dim_model={self.dim_model}, num_heads={self.num_heads}",
        )
        super().build(input_shape)

    def call(self, inputs: KerasTensor, training: bool = False) -> KerasTensor:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Output tensor after applying transformer block.
        """
        # Ensure layer is built (Keras will auto-build on first call)
        if not self.built:
            self.build(inputs.shape)

        # Store original shape and dimensions
        ops.shape(inputs)
        original_rank = len(inputs.shape)

        # Reshape if needed (handle 2D inputs)
        if original_rank == 2:
            inputs = ops.expand_dims(inputs, axis=1)

        # Apply layer normalization first (pre-norm transformer architecture)
        # This helps with training stability and allows better gradient flow
        x = self.layer_norm1(inputs)

        # Multi-head attention
        # Explicitly provide query, key, value to ensure proper attention
        attention_output = self.multihead_attention(query=x, key=x, value=x, training=training)  # type: ignore
        attention_output = self.dropout1(attention_output, training=training)

        # First residual connection
        x = self.add1([inputs, attention_output])

        # Feed-forward network with pre-norm
        y = self.layer_norm2(x)
        ff_output = self.ff1(y)
        ff_output = self.dropout2(ff_output, training=training)
        ff_output = self.ff2(ff_output)

        # Second residual connection
        output = self.add2([x, ff_output])

        # Restore original shape if needed
        if original_rank == 2:
            output = ops.squeeze(output, axis=1)

        return output

    def compute_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Shape of the output tensor.
        """
        return input_shape

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "dim_model": self.dim_model,
                "num_heads": self.num_heads,
                "ff_units": self.ff_units,
                "dropout_rate": self.dropout_rate,
            },
        )
        return config
