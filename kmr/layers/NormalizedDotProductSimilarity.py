"""Normalized dot product similarity layer for collaborative filtering."""

from typing import Any
from keras import ops
from keras import KerasTensor
from keras.saving import register_keras_serializable

from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class NormalizedDotProductSimilarity(BaseLayer):
    """Computes normalized dot product similarity between embeddings.

    Calculates dot product between two embedding vectors and normalizes
    the result for stable training in recommendation systems.

    Input: Tuple of (embedding1, embedding2), each (batch_size, embedding_dim)
    Output: Similarity scores (batch_size, 1) normalized by embedding dimension
    """

    def __init__(self, name: str | None = None, **kwargs: Any) -> None:
        """Initialize layer."""
        self._validate_params()
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate parameters (no-op for this layer)."""
        pass

    def call(self, inputs: tuple[KerasTensor, KerasTensor]) -> KerasTensor:
        """Calculate similarity.

        Args:
            inputs: Tuple of (embedding1, embedding2).

        Returns:
            Similarity scores (batch_size, 1).
        """
        emb1, emb2 = inputs
        dot_product = ops.sum(emb1 * emb2, axis=1, keepdims=True)
        embedding_dim = ops.cast(ops.shape(emb1)[-1], dtype=dot_product.dtype)
        normalized = dot_product / ops.sqrt(embedding_dim)
        return normalized

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration."""
        return super().get_config()
