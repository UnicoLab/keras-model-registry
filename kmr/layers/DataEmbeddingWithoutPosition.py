"""Data Embedding layer combining value and temporal embeddings."""

from typing import Any
from keras import layers
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kmr.layers._base_layer import BaseLayer
from kmr.layers.TokenEmbedding import TokenEmbedding
from kmr.layers.TemporalEmbedding import TemporalEmbedding


@register_keras_serializable(package="kmr.layers")
class DataEmbeddingWithoutPosition(BaseLayer):
    """Combines token (value) and temporal embeddings.

    Embeds time series values using token embedding and optionally adds
    temporal features. Applies dropout after combining embeddings.

    Args:
        c_in: Number of input channels.
        d_model: Dimension of embeddings.
        embed_type: Type of temporal embedding ('fixed' or 'learned').
        freq: Frequency for temporal features ('t' or 'h').
        dropout: Dropout rate (default: 0.1).
        name: Optional name for the layer.

    Example:
        ```python
        import keras
        from kmr.layers import DataEmbeddingWithoutPosition

        # Create data embedding
        data_emb = DataEmbeddingWithoutPosition(c_in=1, d_model=64)

        # Apply to time series values
        x = keras.random.normal((32, 100, 1))
        x_mark = keras.random.uniform((32, 100, 5), minval=0, maxval=13, dtype='int32')

        embeddings = data_emb([x, x_mark])
        print("Embeddings shape:", embeddings.shape)  # (32, 100, 64)
        ```
    """

    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = "fixed",
        freq: str = "h",
        dropout: float = 0.1,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the DataEmbeddingWithoutPosition layer.

        Args:
            c_in: Number of input channels.
            d_model: Dimension of embeddings.
            embed_type: Type of temporal embedding.
            freq: Frequency for temporal embedding.
            dropout: Dropout rate.
            name: Optional layer name.
            **kwargs: Additional keyword arguments.
        """
        # Set private attributes
        self._c_in = c_in
        self._d_model = d_model
        self._embed_type = embed_type
        self._freq = freq
        self._dropout = dropout

        # Validate parameters
        self._validate_params()

        # Set public attributes BEFORE calling parent's __init__
        self.c_in = self._c_in
        self.d_model = self._d_model
        self.embed_type = self._embed_type
        self.freq = self._freq
        self.dropout_rate = self._dropout

        # Embedding layers
        self.value_embedding = None
        self.temporal_embedding = None
        self.dropout = None

        # Call parent's __init__ after setting public attributes
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if not isinstance(self._c_in, int) or self._c_in <= 0:
            raise ValueError(f"c_in must be a positive integer, got {self._c_in}")
        if not isinstance(self._d_model, int) or self._d_model <= 0:
            raise ValueError(f"d_model must be a positive integer, got {self._d_model}")
        if not isinstance(self._dropout, int | float) or not (0 <= self._dropout <= 1):
            raise ValueError(f"dropout must be between 0 and 1, got {self._dropout}")

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Build the layer.

        Args:
            input_shape: Shape of input tensor.
        """
        self.value_embedding = TokenEmbedding(c_in=self.c_in, d_model=self.d_model)
        self.temporal_embedding = TemporalEmbedding(
            d_model=self.d_model,
            embed_type=self.embed_type,
            freq=self.freq,
        )
        self.dropout = layers.Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(
        self,
        inputs: list[KerasTensor] | tuple[KerasTensor, ...],
        training: bool | None = None,
    ) -> KerasTensor:
        """Combine value and temporal embeddings.

        Args:
            inputs: List/tuple of [x, x_mark] where:
                x: Value tensor of shape (batch, time, channels)
                x_mark: Optional temporal indices (batch, time, 5)
            training: Whether in training mode.

        Returns:
            Combined embeddings of shape (batch, time, d_model).
        """
        if isinstance(inputs, list | tuple):
            if len(inputs) == 2:
                x, x_mark = inputs
            else:
                x = inputs[0]
                x_mark = None
        else:
            x = inputs
            x_mark = None

        # Embed values
        x_emb = self.value_embedding(x)

        # Add temporal embedding if provided
        if x_mark is not None:
            x_mark_emb = self.temporal_embedding(x_mark)
            x_emb = x_emb + x_mark_emb

        # Apply dropout
        x_emb = self.dropout(x_emb, training=training)

        return x_emb

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "c_in": self.c_in,
                "d_model": self.d_model,
                "embed_type": self.embed_type,
                "freq": self.freq,
                "dropout": self.dropout_rate,
            },
        )
        return config
