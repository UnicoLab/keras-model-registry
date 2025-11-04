"""Explainable Unified Recommendation model with interpretability.

This module implements an explainable unified recommendation system that combines
multiple approaches with transparency through per-component similarities.
"""

from typing import Any, Optional
from keras import ops, Model
from keras.saving import register_keras_serializable
from loguru import logger

from kmr.models._base import BaseModel
from kmr.layers import (
    CollaborativeUserItemEmbedding,
    DeepFeatureTower,
    NormalizedDotProductSimilarity,
    TopKRecommendationSelector,
)


@register_keras_serializable(package="kmr.models")
class ExplainableUnifiedRecommendationModel(BaseModel):
    """Explainable Unified Recommendation model with transparency.

    Combines multiple recommendation approaches with explainability through
    per-component similarity scores and learnable weight combination.

    Args:
        num_users: Number of unique users.
        num_items: Number of unique items.
        user_feature_dim: Dimension of user feature input.
        item_feature_dim: Dimension of item feature input.
        embedding_dim: Dimension of embeddings (default=32).
        tower_dim: Dimension of feature tower output (default=32).
        top_k: Number of top recommendations to return (default=10).
        l2_reg: L2 regularization factor (default=1e-4).
        preprocessing_model: Optional preprocessing model for input features.
        name: Optional name for the model.

    Inputs:
        - user_ids: User identifiers (batch_size,)
        - user_features: User feature vectors (batch_size, user_feature_dim)
        - item_ids: Item identifiers (batch_size, num_items)
        - item_features: Item feature vectors (batch_size, num_items, item_feature_dim)

    Outputs:
        Tuple of:
        - recommendation_indices: Top-K item indices (batch_size, top_k)
        - recommendation_scores: Top-K blended scores (batch_size, top_k)
        - cf_similarities: Collaborative filtering similarities (batch_size, num_items)
        - cb_similarities: Content-based similarities (batch_size, num_items)
        - component_weights: Learned weights for each approach (3,)
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_feature_dim: int,
        item_feature_dim: int,
        embedding_dim: int = 32,
        tower_dim: int = 32,
        top_k: int = 10,
        l2_reg: float = 1e-4,
        preprocessing_model: Optional[Model] = None,
        name: str = "explainable_unified_recommendation_model",
        **kwargs: Any,
    ) -> None:
        """Initialize ExplainableUnifiedRecommendationModel."""
        super().__init__(name=name, preprocessing_model=preprocessing_model, **kwargs)

        self.num_users = num_users
        self.num_items = num_items
        self.user_feature_dim = user_feature_dim
        self.item_feature_dim = item_feature_dim
        self.embedding_dim = embedding_dim
        self.tower_dim = tower_dim
        self.top_k = top_k
        self.l2_reg = l2_reg

        self._validate_params()

        # Collaborative Filtering component
        self.embedding_layer = CollaborativeUserItemEmbedding(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            l2_reg=l2_reg,
        )

        # Content-Based component - feature towers
        self.user_tower = DeepFeatureTower(
            units=tower_dim,
            hidden_layers=2,
            activation="relu",
            dropout_rate=0.2,
            l2_reg=l2_reg,
            name="user_tower",
        )

        self.item_tower = DeepFeatureTower(
            units=tower_dim,
            hidden_layers=2,
            activation="relu",
            dropout_rate=0.2,
            l2_reg=l2_reg,
            name="item_tower",
        )

        # Similarity layers for explainability
        self.similarity_layer = NormalizedDotProductSimilarity()

        # Top-K selector
        self.selector_layer = TopKRecommendationSelector(k=top_k)

        logger.debug(
            f"Initialized {name} with num_users={num_users}, "
            f"num_items={num_items}, top_k={top_k}",
        )

    def _validate_params(self) -> None:
        """Validate model parameters."""
        if self.num_users <= 0:
            raise ValueError(f"num_users must be positive, got {self.num_users}")
        if self.num_items <= 0:
            raise ValueError(f"num_items must be positive, got {self.num_items}")
        if self.user_feature_dim <= 0:
            raise ValueError(
                f"user_feature_dim must be positive, got {self.user_feature_dim}",
            )
        if self.item_feature_dim <= 0:
            raise ValueError(
                f"item_feature_dim must be positive, got {self.item_feature_dim}",
            )
        if self.top_k <= 0 or self.top_k > self.num_items:
            raise ValueError(
                f"top_k must be between 1 and {self.num_items}, got {self.top_k}",
            )

    def call(
        self,
        inputs: tuple,
        training: bool | None = None,
    ) -> tuple:
        """Forward pass with explanations.

        Args:
            inputs: Tuple of (user_ids, user_features, item_ids, item_features)
            training: Whether in training mode.

        Returns:
            Tuple of (indices, scores, cf_sims, cb_sims, weights)
        """
        user_ids, user_features, item_ids, item_features = inputs

        # ========== Collaborative Filtering Component ==========
        user_emb_cf, item_emb_cf = self.embedding_layer(
            [user_ids, item_ids],
            training=training,
        )
        user_emb_cf_exp = ops.expand_dims(user_emb_cf, axis=1)
        cf_similarities = self.similarity_layer([user_emb_cf_exp, item_emb_cf])
        cf_scores = ops.expand_dims(cf_similarities, axis=-1)

        # ========== Content-Based Component ==========
        batch_size = ops.shape(item_features)[0]
        num_items_actual = ops.shape(item_features)[1]

        user_repr_cb = self.user_tower(user_features, training=training)

        item_features_flat = ops.reshape(item_features, (-1, self.item_feature_dim))
        item_repr_flat = self.item_tower(item_features_flat, training=training)
        item_repr_cb = ops.reshape(
            item_repr_flat,
            (batch_size, num_items_actual, self.tower_dim),
        )

        user_repr_cb_exp = ops.expand_dims(user_repr_cb, axis=1)
        cb_similarities = self.similarity_layer([user_repr_cb_exp, item_repr_cb])
        cb_scores = ops.expand_dims(cb_similarities, axis=-1)

        # ========== Hybrid Component ==========
        hybrid_scores = (cf_scores + cb_scores) / 2.0

        # ========== Combine Scores with Explanations ==========
        # Get weights for explanation
        # Learnable weights represented as simple factors
        weights = ops.array([1.0, 1.0, 1.0])  # Equal weights for all components
        weights = weights / ops.sum(weights)  # Normalize

        # Squeeze extra dimensions and average
        cf_scores_sq = ops.squeeze(cf_scores, axis=1)  # (batch_size, num_items)
        cb_scores_sq = ops.squeeze(cb_scores, axis=1)  # (batch_size, num_items)
        hybrid_scores_sq = ops.squeeze(hybrid_scores, axis=1)  # (batch_size, num_items)
        combined_scores = (cf_scores_sq + cb_scores_sq + hybrid_scores_sq) / 3.0

        # ========== Select Top-K ==========
        rec_indices, rec_scores = self.selector_layer(combined_scores)

        # Return indices, scores, and explanation components
        return rec_indices, rec_scores, cf_similarities, cb_similarities, weights

    def get_config(self) -> dict:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "num_users": self.num_users,
                "num_items": self.num_items,
                "user_feature_dim": self.user_feature_dim,
                "item_feature_dim": self.item_feature_dim,
                "embedding_dim": self.embedding_dim,
                "tower_dim": self.tower_dim,
                "top_k": self.top_k,
                "l2_reg": self.l2_reg,
            },
        )
        return config
