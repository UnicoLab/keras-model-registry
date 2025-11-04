"""Fixed Embedding layer for temporal position encoding."""

import math
from typing import Any
from keras import layers, ops
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class FixedEmbedding(BaseLayer):
    """Fixed sinusoidal embedding layer.

    Provides fixed (non-trainable) sinusoidal embeddings for discrete indices,
    commonly used for encoding temporal features or positions.

    Args:
        n_features: Number of features/vocabulary size.
        d_model: Dimension of the embedding vectors.
        name: Optional name for the layer.

    Input shape:
        (batch_size, seq_len) - integer indices

    Output shape:
        (batch_size, seq_len, d_model)

    Example:
        ```python
        import keras
        from kmr.layers import FixedEmbedding

        # Create fixed embedding
        emb = FixedEmbedding(n_features=32, d_model=64)
        indices = keras.random.uniform((16, 100), minval=0, maxval=32, dtype='int32')
        embeddings = emb(indices)
        print("Embeddings shape:", embeddings.shape)  # (16, 100, 64)
        ```
    """

    def __init__(
        self,
        n_features: int,
        d_model: int,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the FixedEmbedding layer.

        Args:
            n_features: Number of discrete features/positions.
            d_model: Dimension of embedding vectors.
            name: Optional layer name.
            **kwargs: Additional keyword arguments.
        """
        # Set private attributes
        self._n_features = n_features
        self._d_model = d_model

        # Validate parameters
        self._validate_params()

        # Set public attributes BEFORE calling parent's __init__
        self.n_features = self._n_features
        self.d_model = self._d_model
        self.embedding_layer = None

        # Call parent's __init__ after setting public attributes
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if not isinstance(self._n_features, int) or self._n_features <= 0:
            raise ValueError(
                f"n_features must be a positive integer, got {self._n_features}",
            )
        if not isinstance(self._d_model, int) or self._d_model <= 0:
            raise ValueError(f"d_model must be a positive integer, got {self._d_model}")

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Build the layer by creating fixed embeddings.

        Args:
            input_shape: Shape of input tensor.
        """
        # Create fixed sinusoidal embeddings
        weights = ops.zeros((self.n_features, self.d_model))

        position = ops.arange(0, self.n_features, dtype="float32")
        position = ops.expand_dims(position, axis=1)  # (n_features, 1)

        div_term = ops.exp(
            ops.arange(0, self.d_model, 2, dtype="float32")
            * (-math.log(10000.0) / self.d_model),
        )  # (d_model // 2,)

        # Compute sin and cos
        sin_vals = ops.sin(position * div_term)  # (n_features, d_model // 2)
        cos_vals = ops.cos(position * div_term)  # (n_features, d_model // 2)

        # Interleave sin and cos
        pe_list = []
        for i in range(self.n_features):
            row = ops.concatenate(
                [
                    ops.expand_dims(sin_vals[i, :], 0),
                    ops.expand_dims(cos_vals[i, :], 0),
                ],
                axis=1,
            )
            row = ops.reshape(row, (self.d_model,))
            pe_list.append(ops.expand_dims(row, 0))

        weights = ops.concatenate(pe_list, axis=0)  # (n_features, d_model)

        # Create embedding layer
        self.embedding_layer = layers.Embedding(
            input_dim=self.n_features,
            output_dim=self.d_model,
            trainable=False,
        )

        # Set the weights
        self.embedding_layer.build(input_shape)
        self.embedding_layer.set_weights([weights])

        super().build(input_shape)

    def call(self, inputs: KerasTensor) -> KerasTensor:
        """Look up embeddings for the input indices.

        Args:
            inputs: Integer tensor of shape (batch, seq_len).

        Returns:
            Embedding tensor of shape (batch, seq_len, d_model).
        """
        return self.embedding_layer(inputs)

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "n_features": self.n_features,
                "d_model": self.d_model,
            },
        )
        return config
