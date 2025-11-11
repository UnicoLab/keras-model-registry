"""Collaborative embedding layer for recommendation systems.

Provides dual embedding lookups for users and items with configurable
L2 regularization for improved generalization.
"""

from typing import Any
from keras import layers
from keras import KerasTensor
from keras.saving import register_keras_serializable
from keras.regularizers import l2

from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class CollaborativeUserItemEmbedding(BaseLayer):
    """Dual user and item embedding lookup with L2 regularization.

    This layer provides embedding lookups for both users and items in a
    collaborative filtering context. Each embedding is regularized with L2
    to prevent overfitting and improve generalization to unseen items/users.

    Args:
        num_users: Number of unique users (vocabulary size for user embeddings).
        num_items: Number of unique items (vocabulary size for item embeddings).
        embedding_dim: Dimension of embedding vectors (default=32).
        l2_reg: L2 regularization coefficient (default=1e-6).
        name: Optional name for the layer.

    Input:
        Tuple of (user_ids, item_ids) where:
        - user_ids: shape (batch_size,), integer IDs of users
        - item_ids: shape (batch_size,), integer IDs of items

    Output:
        Tuple of (user_embeddings, item_embeddings) where:
        - user_embeddings: shape (batch_size, embedding_dim)
        - item_embeddings: shape (batch_size, embedding_dim)

    Example:
        ```python
        import keras
        from kmr.layers import CollaborativeUserItemEmbedding

        # Create embedding layer for 1000 users and 500 items
        embedding_layer = CollaborativeUserItemEmbedding(
            num_users=1000, num_items=500, embedding_dim=32, l2_reg=1e-6
        )

        # Create sample user and item IDs
        user_ids = keras.constant([1, 5, 10, 3])
        item_ids = keras.constant([2, 8, 15, 7])

        # Get embeddings
        user_emb, item_emb = embedding_layer([user_ids, item_ids])
        print("User embeddings shape:", user_emb.shape)  # (4, 32)
        print("Item embeddings shape:", item_emb.shape)  # (4, 32)
        ```
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        l2_reg: float = 1e-6,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the CollaborativeUserItemEmbedding layer.

        Args:
            num_users: Number of unique users.
            num_items: Number of unique items.
            embedding_dim: Embedding vector dimension.
            l2_reg: L2 regularization coefficient.
            name: Name of the layer.
            **kwargs: Additional keyword arguments.
        """
        # Set private attributes first
        self._num_users = num_users
        self._num_items = num_items
        self._embedding_dim = embedding_dim
        self._l2_reg = float(l2_reg)

        # Validate parameters
        self._validate_params()

        # Set public attributes BEFORE calling parent's __init__
        self.num_users = self._num_users
        self.num_items = self._num_items
        self.embedding_dim = self._embedding_dim
        self.l2_reg = self._l2_reg
        self.user_embedding = None
        self.item_embedding = None

        # Call parent's __init__
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if not isinstance(self._num_users, int) or self._num_users <= 0:
            raise ValueError(
                f"num_users must be positive integer, got {self._num_users}",
            )
        if not isinstance(self._num_items, int) or self._num_items <= 0:
            raise ValueError(
                f"num_items must be positive integer, got {self._num_items}",
            )
        if not isinstance(self._embedding_dim, int) or self._embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be positive integer, got {self._embedding_dim}",
            )
        if not isinstance(self._l2_reg, int | float) or self._l2_reg < 0:
            raise ValueError(f"l2_reg must be non-negative, got {self._l2_reg}")

    def build(self, input_shape: tuple) -> None:
        """Build layer with given input shape.

        Args:
            input_shape: Input shape tuple.
        """
        # Create user embedding layer
        self.user_embedding = layers.Embedding(
            input_dim=self.num_users,
            output_dim=self.embedding_dim,
            embeddings_regularizer=l2(self.l2_reg),
            name="user_embedding",
        )

        # Create item embedding layer
        self.item_embedding = layers.Embedding(
            input_dim=self.num_items,
            output_dim=self.embedding_dim,
            embeddings_regularizer=l2(self.l2_reg),
            name="item_embedding",
        )

        super().build(input_shape)

    def call(
        self,
        inputs: tuple[KerasTensor, KerasTensor],
    ) -> tuple[KerasTensor, KerasTensor]:
        """Lookup user and item embeddings.

        Args:
            inputs: Tuple of (user_ids, item_ids).

        Returns:
            Tuple of (user_embeddings, item_embeddings).
        """
        user_ids, item_ids = inputs

        # Lookup embeddings
        user_vecs = self.user_embedding(user_ids)
        item_vecs = self.item_embedding(item_ids)

        return user_vecs, item_vecs

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "num_users": self.num_users,
                "num_items": self.num_items,
                "embedding_dim": self.embedding_dim,
                "l2_reg": self.l2_reg,
            },
        )
        if isinstance(self.user_embedding, layers.Embedding):
            config["user_embedding_weights"] = self.user_embedding.get_weights()
        if isinstance(self.item_embedding, layers.Embedding):
            config["item_embedding_weights"] = self.item_embedding.get_weights()
        return config
