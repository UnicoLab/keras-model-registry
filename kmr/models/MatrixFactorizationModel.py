"""Matrix Factorization recommendation model.

This module implements a matrix factorization-based recommendation system using
user and item embeddings with dot product similarity for ranking.
"""

from typing import Any, Optional
from keras import ops, Model
from keras.saving import register_keras_serializable
from loguru import logger

from kmr.models._base import BaseModel
from kmr.layers import (
    CollaborativeUserItemEmbedding,
    NormalizedDotProductSimilarity,
    TopKRecommendationSelector,
)


@register_keras_serializable(package="kmr.models")
class MatrixFactorizationModel(BaseModel):
    """Matrix Factorization recommendation model with full Keras compatibility.

    Uses user and item embeddings with dot product similarity for collaborative
    filtering. Embeddings are learned to minimize ranking loss using standard
    Keras compile() and fit() methods.

    This model implements the standard Keras API:
    - compile(): Use standard Keras optimizer and custom ImprovedMarginRankingLoss
    - fit(): Use standard Keras training loop with recommendation metrics
    - predict(): Generate recommendations for inference

    Architecture:
        - User and item embeddings with L2 regularization
        - Normalized dot product similarity computation
        - Top-K recommendation selection via TopKRecommendationSelector

    Args:
        num_users: Number of unique users.
        num_items: Number of unique items.
        embedding_dim: Dimension of user/item embeddings (default=32).
        top_k: Number of top recommendations to return (default=10).
        l2_reg: L2 regularization factor for embeddings (default=1e-4).
        preprocessing_model: Optional preprocessing model for input features.
        name: Optional name for the model.

    Inputs:
        - user_ids: User identifiers (batch_size,)
        - item_ids: Item identifiers (batch_size, num_items)

    Outputs:
        Tuple of:
        - similarities: All item similarity scores (batch_size, num_items)
        - recommendation_indices: Top-K item indices (batch_size, top_k)
        - recommendation_scores: Top-K similarity scores (batch_size, top_k)

    Example:
        ```python
        import keras
        import numpy as np
        from kmr.models import MatrixFactorizationModel
        from kmr.losses import ImprovedMarginRankingLoss
        from kmr.metrics import AccuracyAtK, PrecisionAtK, RecallAtK

        # Create model
        model = MatrixFactorizationModel(
            num_users=1000,
            num_items=500,
            embedding_dim=32,
            top_k=10
        )

        # Compile with custom loss and metrics
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=ImprovedMarginRankingLoss(margin=1.0),
            metrics=[
                AccuracyAtK(k=5, name='acc@5'),
                AccuracyAtK(k=10, name='acc@10'),
                PrecisionAtK(k=10, name='prec@10'),
                RecallAtK(k=10, name='recall@10'),
            ]
        )

        # Train with binary labels (1=positive, 0=negative)
        user_ids = np.random.randint(0, 1000, (32,))
        item_ids = np.random.randint(0, 500, (32, 500))
        labels = np.random.randint(0, 2, (32, 500)).astype(np.float32)

        history = model.fit(
            x=[user_ids, item_ids],
            y=labels,
            epochs=10,
            batch_size=32
        )

        # Generate recommendations for inference
        similarities, indices, scores = model.predict([user_ids, item_ids])
        print("Similarities:", similarities.shape)       # (32, 500)
        print("Recommendation indices:", indices.shape)  # (32, 10)
        print("Recommendation scores:", scores.shape)    # (32, 10)
        ```

    Keras Compatibility:
        ✅ Standard compile() - Works with standard optimizers and loss functions
        ✅ Standard fit() - Uses default Keras training loop
        ✅ Standard predict() - Generates predictions without custom code
        ✅ Serializable - Full save/load support via get_config()
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        top_k: int = 10,
        l2_reg: float = 1e-4,
        preprocessing_model: Optional[Model] = None,
        name: str = "matrix_factorization_model",
        **kwargs: Any,
    ) -> None:
        """Initialize MatrixFactorizationModel."""
        super().__init__(name=name, preprocessing_model=preprocessing_model, **kwargs)

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self.l2_reg = l2_reg

        self._validate_params()

        # User and item embedding layer
        self.embedding_layer = CollaborativeUserItemEmbedding(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            l2_reg=l2_reg,
        )

        # Similarity computation
        self.similarity_layer = NormalizedDotProductSimilarity()

        # Top-K selector
        self.selector_layer = TopKRecommendationSelector(k=top_k)

        logger.debug(
            f"Initialized {name} with num_users={num_users}, "
            f"num_items={num_items}, embedding_dim={embedding_dim}, top_k={top_k}",
        )

    def _validate_params(self) -> None:
        """Validate model parameters."""
        if self.num_users <= 0:
            raise ValueError(f"num_users must be positive, got {self.num_users}")
        if self.num_items <= 0:
            raise ValueError(f"num_items must be positive, got {self.num_items}")
        if self.embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be positive, got {self.embedding_dim}",
            )
        if self.top_k <= 0 or self.top_k > self.num_items:
            raise ValueError(
                f"top_k must be between 1 and {self.num_items}, got {self.top_k}",
            )
        if self.l2_reg < 0:
            raise ValueError(f"l2_reg must be non-negative, got {self.l2_reg}")

    def call(
        self,
        inputs: tuple,
        training: bool | None = None,
    ) -> tuple:
        """Forward pass for recommendation generation.

        Args:
            inputs: Tuple of (user_ids, item_ids)
            training: Whether in training mode.

        Returns:
            Tuple of (similarities, recommendation_indices, recommendation_scores)
            where:
            - similarities: All item scores (batch_size, num_items) for loss computation
            - recommendation_indices: Top-K item indices (batch_size, top_k)
            - recommendation_scores: Top-K scores (batch_size, top_k)

            This tuple is returned consistently for both training and inference modes,
            following Keras 3 best practices for pure functional architecture.
        """
        user_ids, item_ids = inputs

        # Get user and item embeddings
        user_emb, item_emb = self.embedding_layer(
            [user_ids, item_ids],
            training=training,
        )

        # Compute similarities using dot product
        user_emb_exp = ops.expand_dims(
            user_emb,
            axis=1,
        )  # (batch_size, 1, embedding_dim)
        similarities = ops.sum(
            user_emb_exp * item_emb,
            axis=-1,
        )  # (batch_size, num_items)

        # Normalize similarities
        user_norm = ops.sqrt(ops.sum(user_emb_exp**2, axis=-1, keepdims=True) + 1e-8)
        item_norm = ops.sqrt(ops.sum(item_emb**2, axis=-1, keepdims=True) + 1e-8)
        similarities = similarities / (user_norm[:, 0, :] * item_norm[:, :, 0] + 1e-8)

        # Select top-K
        rec_indices, rec_scores = self.selector_layer(similarities)

        # Return tuple: (similarities, rec_indices, rec_scores)
        # Keras handles tuples natively for both training and inference
        return (similarities, rec_indices, rec_scores)

    def get_config(self) -> dict:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "num_users": self.num_users,
                "num_items": self.num_items,
                "embedding_dim": self.embedding_dim,
                "top_k": self.top_k,
                "l2_reg": self.l2_reg,
            },
        )
        return config
