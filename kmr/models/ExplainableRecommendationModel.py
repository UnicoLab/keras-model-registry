"""Explainable Recommendation model with interpretability.

This module implements an explainable recommendation system that provides
cosine similarity scores between user and item embeddings for interpretability,
along with feedback-based adjustments for personalization.
"""

from typing import Any, Optional
from keras import ops, Model
from keras.saving import register_keras_serializable
from loguru import logger

from kmr.models._base import BaseModel
from kmr.layers import (
    CollaborativeUserItemEmbedding,
    CosineSimilarityExplainer,
    FeedbackAdjustmentLayer,
    TopKRecommendationSelector,
)


@register_keras_serializable(package="kmr.models")
class ExplainableRecommendationModel(BaseModel):
    """Explainable Recommendation model with interpretability features.

    Provides transparent recommendation generation with cosine similarity-based
    explanations. Users can understand recommendations through similarity scores
    and feedback-based adjustments.

    Args:
        num_users: Number of unique users.
        num_items: Number of unique items.
        embedding_dim: Dimension of user/item embeddings (default=32).
        top_k: Number of top recommendations to return (default=10).
        l2_reg: L2 regularization factor for embeddings (default=1e-4).
        feedback_weight: Weight for feedback adjustment (default=0.5).
        preprocessing_model: Optional preprocessing model for input features.
        name: Optional name for the model.

    Inputs:
        - user_ids: User identifiers (batch_size,)
        - item_ids: Item identifiers (batch_size, num_items)
        - user_feedback: Optional user feedback signals (batch_size, num_items)

    Outputs:
        Tuple of:
        - recommendation_indices: Top-K item indices (batch_size, top_k)
        - recommendation_scores: Top-K scores with explanations (batch_size, top_k)
        - similarity_matrix: User-item similarity matrix for explanation (batch_size, num_items)

    Example:
        ```python
        import keras
        import numpy as np
        from kmr.models import ExplainableRecommendationModel

        model = ExplainableRecommendationModel(
            num_users=1000,
            num_items=500,
            embedding_dim=32,
            top_k=10
        )

        # Sample data
        user_ids = np.random.randint(0, 1000, (32,))
        item_ids = np.random.randint(0, 500, (32, 500))
        user_feedback = np.random.uniform(0, 1, (32, 500))

        # Get recommendations with explanations
        indices, scores, similarities = model([user_ids, item_ids, user_feedback])
        print("Recommendation indices:", indices.shape)  # (32, 10)
        print("Recommendation scores:", scores.shape)    # (32, 10)
        print("Similarity matrix:", similarities.shape)  # (32, 500)
        ```
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        top_k: int = 10,
        l2_reg: float = 1e-4,
        feedback_weight: float = 0.5,
        preprocessing_model: Optional[Model] = None,
        name: str = "explainable_recommendation_model",
        **kwargs: Any,
    ) -> None:
        """Initialize ExplainableRecommendationModel."""
        super().__init__(name=name, preprocessing_model=preprocessing_model, **kwargs)

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self.l2_reg = l2_reg
        self.feedback_weight = feedback_weight

        self._validate_params()

        # User and item embedding layer
        self.embedding_layer = CollaborativeUserItemEmbedding(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            l2_reg=l2_reg,
        )

        # Cosine similarity explainer for interpretation
        self.explainer = CosineSimilarityExplainer()

        # Feedback adjustment layer
        self.feedback_adjuster = FeedbackAdjustmentLayer()

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
        if not (0 <= self.feedback_weight <= 1):
            raise ValueError(
                f"feedback_weight must be in [0, 1], got {self.feedback_weight}",
            )

    def call(
        self,
        inputs: tuple,
        training: bool | None = None,
    ) -> tuple:
        """Forward pass for recommendation generation with explanations.

        Args:
            inputs: Tuple of (user_ids, item_ids, user_feedback) or (user_ids, item_ids)
            training: Whether in training mode.

        Returns:
            Tuple of (recommendation_indices, recommendation_scores, similarity_matrix)
        """
        # Handle optional feedback
        if len(inputs) == 3:
            user_ids, item_ids, user_feedback = inputs
        else:
            user_ids, item_ids = inputs
            user_feedback = None

        # Get embeddings
        user_emb, item_emb = self.embedding_layer(
            [user_ids, item_ids],
            training=training,
        )

        # Compute cosine similarity for explainability
        # user_emb: (batch_size, embedding_dim)
        # item_emb: (batch_size, num_items, embedding_dim)
        similarity_matrix = self.explainer([user_emb, item_emb])

        # Apply feedback adjustment if provided
        if user_feedback is not None:
            # Reshape feedback for adjustment layer
            # user_feedback: (batch_size, num_items) -> (batch_size, num_items, 1)
            feedback_exp = ops.expand_dims(user_feedback, axis=-1)
            similarity_adj = self.feedback_adjuster(feedback_exp)

            # Combine original similarity with feedback-adjusted similarity
            # feedback_adj: (batch_size, num_items, 1) -> (batch_size, num_items)
            feedback_adj = ops.squeeze(similarity_adj, axis=-1)

            # Weight-adjusted combination
            scores = (
                1 - self.feedback_weight
            ) * similarity_matrix + self.feedback_weight * feedback_adj
        else:
            scores = similarity_matrix

        # Select top-K
        rec_indices, rec_scores = self.selector_layer(scores)

        return rec_indices, rec_scores, similarity_matrix

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
                "feedback_weight": self.feedback_weight,
            },
        )
        return config
