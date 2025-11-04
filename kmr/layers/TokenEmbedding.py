"""Token Embedding layer for time series using 1D convolution."""

from typing import Any
from keras import layers
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class TokenEmbedding(BaseLayer):
    """Embeds time series values using 1D convolution.

    Uses a conv1d layer with circular padding to create embeddings from raw values.
    Kaiming normal initialization is applied for proper training dynamics.

    Args:
        c_in: Number of input channels.
        d_model: Dimension of output embeddings.
        name: Optional name for the layer.

    Input shape:
        (batch_size, time_steps, channels)

    Output shape:
        (batch_size, time_steps, d_model)

    Example:
        ```python
        import keras
        from kmr.layers import TokenEmbedding

        # Create token embedding
        token_emb = TokenEmbedding(c_in=1, d_model=64)

        # Apply to time series
        x = keras.random.normal((32, 100, 1))
        embeddings = token_emb(x)
        print("Embeddings shape:", embeddings.shape)  # (32, 100, 64)
        ```
    """

    def __init__(
        self,
        c_in: int,
        d_model: int,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the TokenEmbedding layer.

        Args:
            c_in: Number of input channels.
            d_model: Dimension of output embeddings.
            name: Optional layer name.
            **kwargs: Additional keyword arguments.
        """
        # Set private attributes
        self._c_in = c_in
        self._d_model = d_model

        # Validate parameters
        self._validate_params()

        # Set public attributes BEFORE calling parent's __init__
        self.c_in = self._c_in
        self.d_model = self._d_model
        self.conv: layers.Conv1D | None = None

        # Call parent's __init__ after setting public attributes
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if not isinstance(self._c_in, int) or self._c_in <= 0:
            raise ValueError(f"c_in must be a positive integer, got {self._c_in}")
        if not isinstance(self._d_model, int) or self._d_model <= 0:
            raise ValueError(f"d_model must be a positive integer, got {self._d_model}")

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Build the layer.

        Args:
            input_shape: Shape of input tensor (batch, time, c_in).
        """
        # Create 1D convolution with same padding
        # Conv1D expects (batch, time, channels) format by default
        self.conv = layers.Conv1D(
            filters=self.d_model,
            kernel_size=3,
            padding="same",
            use_bias=False,
            input_shape=input_shape[1:],  # Exclude batch dimension
        )

        super().build(input_shape)

    def call(self, inputs: KerasTensor) -> KerasTensor:
        """Apply token embedding.

        Args:
            inputs: Input tensor of shape (batch, time, channels).

        Returns:
            Embedded tensor of shape (batch, time, d_model).
        """
        # Keras Conv1D expects (batch, time, channels) format (channels_last)
        # So we can apply directly without transposition
        if self.conv is None:
            raise RuntimeError("Layer must be built before calling")
        output = self.conv(inputs)

        return output

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
            },
        )
        return config
