"""Temporal Embedding layer for time feature encoding."""

from typing import Any
from keras import layers, ops
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kerasfactory.layers._base_layer import BaseLayer
from kerasfactory.layers.FixedEmbedding import FixedEmbedding


@register_keras_serializable(package="kerasfactory.layers")
class TemporalEmbedding(BaseLayer):
    """Embeds temporal features (month, day, weekday, hour, minute).

    Creates embeddings for calendar features to capture temporal patterns.
    Supports both fixed and trainable embedding modes.

    Args:
        d_model: Dimension of embeddings.
        embed_type: Type of embedding - 'fixed' or 'learned' (default: 'fixed').
        freq: Frequency - 't' (minute level) or 'h' (hour level) (default: 'h').
        name: Optional name for the layer.

    Input shape:
        (batch_size, seq_len, 5) - with encoded [month, day, weekday, hour, minute]

    Output shape:
        (batch_size, seq_len, d_model)

    Example:
        ```python
        import keras
        from kerasfactory.layers import TemporalEmbedding

        # Create temporal embedding
        temp_emb = TemporalEmbedding(d_model=64)

        # Apply to temporal features
        x = keras.random.uniform((32, 100, 5), minval=0, maxval=13, dtype='int32')
        embeddings = temp_emb(x)
        print("Embeddings shape:", embeddings.shape)  # (32, 100, 64)
        ```
    """

    def __init__(
        self,
        d_model: int,
        embed_type: str = "fixed",
        freq: str = "h",
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the TemporalEmbedding layer.

        Args:
            d_model: Dimension of embeddings.
            embed_type: Type of embedding ('fixed' or 'learned').
            freq: Frequency ('t' or 'h').
            name: Optional layer name.
            **kwargs: Additional keyword arguments.
        """
        # Set private attributes
        self._d_model = d_model
        self._embed_type = embed_type
        self._freq = freq

        # Validate parameters
        self._validate_params()

        # Set public attributes BEFORE calling parent's __init__
        self.d_model = self._d_model
        self.embed_type = self._embed_type
        self.freq = self._freq

        # Embedding layers
        self.minute_embed: FixedEmbedding | layers.Embedding | None = None
        self.hour_embed: FixedEmbedding | layers.Embedding | None = None
        self.weekday_embed: FixedEmbedding | layers.Embedding | None = None
        self.day_embed: FixedEmbedding | layers.Embedding | None = None
        self.month_embed: FixedEmbedding | layers.Embedding | None = None

        # Call parent's __init__ after setting public attributes
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if not isinstance(self._d_model, int) or self._d_model <= 0:
            raise ValueError(f"d_model must be a positive integer, got {self._d_model}")
        if self._embed_type not in ["fixed", "learned"]:
            raise ValueError(
                f"embed_type must be 'fixed' or 'learned', got {self._embed_type}",
            )
        if self._freq not in ["t", "h"]:
            raise ValueError(f"freq must be 't' or 'h', got {self._freq}")

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Build the layer.

        Args:
            input_shape: Shape of input tensor.
        """
        # Define feature sizes
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        # Select embedding class
        embed_class = FixedEmbedding if self.embed_type == "fixed" else layers.Embedding

        # Create embeddings for each temporal component
        if self.freq == "t":
            self.minute_embed = embed_class(
                minute_size if self.embed_type == "fixed" else minute_size,
                self.d_model,
            )

        self.hour_embed = embed_class(
            hour_size if self.embed_type == "fixed" else hour_size,
            self.d_model,
        )
        self.weekday_embed = embed_class(
            weekday_size if self.embed_type == "fixed" else weekday_size,
            self.d_model,
        )
        self.day_embed = embed_class(
            day_size if self.embed_type == "fixed" else day_size,
            self.d_model,
        )
        self.month_embed = embed_class(
            month_size if self.embed_type == "fixed" else month_size,
            self.d_model,
        )

        super().build(input_shape)

    def call(self, inputs: KerasTensor) -> KerasTensor:
        """Embed temporal features.

        Args:
            inputs: Temporal indices of shape (batch, seq_len, 5).

        Returns:
            Embeddings of shape (batch, seq_len, d_model).
        """
        # Convert to long/int for indexing
        inputs = ops.cast(inputs, "int32")

        # Extract temporal components
        # inputs expected to be (..., [month, day, weekday, hour, minute])
        if self.month_embed is None:
            raise RuntimeError("Layer must be built before calling")
        if self.day_embed is None:
            raise RuntimeError("Layer must be built before calling")
        if self.weekday_embed is None:
            raise RuntimeError("Layer must be built before calling")
        if self.hour_embed is None:
            raise RuntimeError("Layer must be built before calling")
        month_x = self.month_embed(inputs[..., 0])
        day_x = self.day_embed(inputs[..., 1])
        weekday_x = self.weekday_embed(inputs[..., 2])
        hour_x = self.hour_embed(inputs[..., 3])

        # Add minute if frequency is minute-level
        if self.freq == "t" and self.minute_embed is not None:
            minute_x = self.minute_embed(inputs[..., 4])
            return hour_x + weekday_x + day_x + month_x + minute_x
        else:
            return hour_x + weekday_x + day_x + month_x

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "embed_type": self.embed_type,
                "freq": self.freq,
            },
        )
        return config
