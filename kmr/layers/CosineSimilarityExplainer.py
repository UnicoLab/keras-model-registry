"""Cosine similarity explainer layer for recommendation systems."""

from typing import Any
from keras import ops
from keras import KerasTensor
from keras.saving import register_keras_serializable

from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class CosineSimilarityExplainer(BaseLayer):
    """Analyzes cosine similarity between user and item embeddings.

    Computes cosine similarity for explainability, showing which items
    are most similar to given user embeddings.

    Input: Tuple of (user_embeddings, all_item_embeddings)
    Output: Similarity matrix (batch_size, num_items) with values in [-1, 1]
    """

    def __init__(self, name: str | None = None, **kwargs: Any) -> None:
        """Initialize layer."""
        self._validate_params()
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate parameters (no-op)."""
        pass

    def call(self, inputs: tuple[KerasTensor, KerasTensor]) -> KerasTensor:
        """Calculate cosine similarity.

        Args:
            inputs: Tuple of (user_emb, item_embeddings).
                - user_emb: (batch_size, embedding_dim)
                - item_embeddings: (batch_size, num_items, embedding_dim)

        Returns:
            Similarity scores (batch_size, num_items).
        """
        user_emb, item_emb = inputs

        # Normalize user embeddings (batch_size, embedding_dim)
        user_norm = user_emb / (ops.norm(user_emb, axis=-1, keepdims=True) + 1e-10)

        # Normalize item embeddings (batch_size, num_items, embedding_dim)
        # Normalize along last axis (embedding_dim)
        item_norm = item_emb / (ops.norm(item_emb, axis=-1, keepdims=True) + 1e-10)

        # Compute cosine similarity via batched matrix multiplication
        # user_norm: (batch_size, 1, embedding_dim) after expand_dims
        # item_norm: (batch_size, num_items, embedding_dim)
        # Result: (batch_size, 1, num_items) -> reshape to (batch_size, num_items)
        user_norm_exp = ops.expand_dims(
            user_norm,
            axis=1,
        )  # (batch_size, 1, embedding_dim)
        similarity = ops.matmul(user_norm_exp, ops.transpose(item_norm, axes=(0, 2, 1)))
        similarity = ops.squeeze(similarity, axis=1)  # (batch_size, num_items)
        return similarity

    def get_config(self) -> dict[str, Any]:
        """Get configuration."""
        return super().get_config()
