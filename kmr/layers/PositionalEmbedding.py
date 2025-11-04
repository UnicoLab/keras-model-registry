"""Positional Embedding layer for transformer-based models."""

import math
from typing import Any
from keras import ops
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class PositionalEmbedding(BaseLayer):
    """Sinusoidal positional encoding layer.

    Generates fixed positional encodings using sine and cosine functions with
    different frequencies. These are added to input embeddings to provide
    positional information to the model.

    Args:
        d_model: Dimension of the positional embeddings.
        max_len: Maximum length of sequences (default: 5000).
        name: Optional name for the layer.

    Input shape:
        (batch_size, seq_len, ...)

    Output shape:
        (1, seq_len, d_model)

    Example:
        ```python
        import keras
        from kmr.layers import PositionalEmbedding

        # Create positional embeddings
        pos_emb = PositionalEmbedding(d_model=64, max_len=512)
        positions = pos_emb(keras.random.normal((32, 100, 64)))
        print("Positional embeddings shape:", positions.shape)  # (1, 100, 64)
        ```
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the PositionalEmbedding layer.

        Args:
            d_model: Dimension of positional embeddings.
            max_len: Maximum sequence length.
            name: Optional layer name.
            **kwargs: Additional keyword arguments.
        """
        # Set private attributes
        self._d_model = d_model
        self._max_len = max_len

        # Validate parameters
        self._validate_params()

        # Set public attributes BEFORE calling parent's __init__
        self.d_model = self._d_model
        self.max_len = self._max_len
        self.pe = None

        # Call parent's __init__ after setting public attributes
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if not isinstance(self._d_model, int) or self._d_model <= 0:
            raise ValueError(f"d_model must be a positive integer, got {self._d_model}")
        if not isinstance(self._max_len, int) or self._max_len <= 0:
            raise ValueError(f"max_len must be a positive integer, got {self._max_len}")

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Build the layer by precomputing positional encodings.

        Args:
            input_shape: Shape of input tensor.
        """
        # Compute positional encodings: (max_len, d_model)
        position = ops.arange(0, self.max_len, dtype="float32")
        position = ops.expand_dims(position, axis=1)  # (max_len, 1)

        div_term = ops.exp(
            ops.arange(0, self.d_model, 2, dtype="float32")
            * (-math.log(10000.0) / self.d_model),
        )  # (d_model // 2,)

        # Compute sin and cos
        sin_vals = ops.sin(position * div_term)  # (max_len, d_model // 2)
        cos_vals = ops.cos(position * div_term)  # (max_len, d_model // 2)

        # Interleave sin and cos
        pe_list = []
        for i in range(self.max_len):
            row = ops.concatenate(
                [
                    ops.expand_dims(sin_vals[i, :], 0),
                    ops.expand_dims(cos_vals[i, :], 0),
                ],
                axis=1,
            )
            row = ops.reshape(row, (self.d_model,))
            pe_list.append(ops.expand_dims(row, 0))

        pe = ops.concatenate(pe_list, axis=0)  # (max_len, d_model)
        self.pe = ops.expand_dims(pe, axis=0)  # (1, max_len, d_model)

        super().build(input_shape)

    def call(self, inputs: KerasTensor) -> KerasTensor:
        """Return positional encodings for the sequence length.

        Args:
            inputs: Input tensor of shape (batch, seq_len, ...).

        Returns:
            Positional encodings of shape (1, seq_len, d_model).
        """
        seq_len = ops.shape(inputs)[1]
        return self.pe[:, :seq_len, :]

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "max_len": self.max_len,
            },
        )
        return config
