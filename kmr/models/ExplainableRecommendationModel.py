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
    """Explainable Recommendation model with full Keras compatibility.

    Provides transparent recommendation generation with cosine similarity-based
    explanations. Users can understand recommendations through similarity scores
    and feedback-based adjustments.

    This model implements the standard Keras API:
    - compile(): Use standard Keras optimizer and custom ImprovedMarginRankingLoss
    - fit(): Use standard Keras training loop with recommendation metrics
    - predict(): Generate recommendations for inference

    Architecture:
        - Collaborative user and item embeddings with cosine similarity
        - Cosine similarity explainer for interpretable explanations
        - Feedback adjustment layer for personalization
        - Top-K recommendation selection via TopKRecommendationSelector

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
        - During training: Ranking scores (batch_size, num_items) for loss computation
        - During inference: Tuple of (recommendation_indices, recommendation_scores, similarity_matrix)
            - recommendation_indices: Top-K item indices (batch_size, top_k)
            - recommendation_scores: Top-K scores with explanations (batch_size, top_k)
            - similarity_matrix: User-item similarity matrix for explanation (batch_size, num_items)

    Example:
        ```python
        import keras
        import numpy as np
        from kmr.models import ExplainableRecommendationModel
        from kmr.losses import ImprovedMarginRankingLoss
        from kmr.metrics import AccuracyAtK, PrecisionAtK, RecallAtK

        # Create model
        model = ExplainableRecommendationModel(
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
        user_feedback = np.random.uniform(0, 1, (32, 500))
        labels = np.random.randint(0, 2, (32, 500)).astype(np.float32)

        history = model.fit(
            x=[user_ids, item_ids, user_feedback],
            y=labels,
            epochs=10,
            batch_size=32
        )

        # Generate recommendations for inference
        indices, scores, similarities = model.predict([user_ids, item_ids, user_feedback])
        print("Recommendation indices:", indices.shape)  # (32, 10)
        print("Recommendation scores:", scores.shape)    # (32, 10)
        print("Similarity matrix:", similarities.shape)  # (32, 500)
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
            Tuple of (scores, rec_indices, rec_scores, similarity_matrix, feedback_adjusted_scores)
            where:
            - scores: Ranking scores (batch_size, num_items) for loss computation
            - rec_indices: Top-K item indices (batch_size, top_k)
            - rec_scores: Top-K ranking scores (batch_size, top_k)
            - similarity_matrix: User-item similarity matrix for explanations (batch_size, num_items)
            - feedback_adjusted_scores: Scores adjusted by user feedback (batch_size, num_items)

            This tuple is returned consistently for both training and inference modes,
            following Keras 3 best practices for pure functional architecture.
        """
        # Handle variable input formats
        if len(inputs) == 3:
            user_ids, item_ids, user_feedback = inputs
        else:
            user_ids, item_ids = inputs
            user_feedback = None

        # Compute base ranking scores
        user_emb, item_emb = self.embedding_layer(
            [user_ids, item_ids],
            training=training,
        )
        similarity_matrix = self.explainer([user_emb, item_emb])
        scores = (
            ops.squeeze(similarity_matrix, axis=-1)
            if len(similarity_matrix.shape) > 2
            else similarity_matrix
        )

        # Apply feedback adjustment if provided
        if user_feedback is not None:
            feedback_adjusted = self.feedback_layer(
                [scores, user_feedback],
                training=training,
            )
        else:
            feedback_adjusted = scores

        # Select top-K
        rec_indices, rec_scores = self.selector_layer(scores)

        # Return tuple - all components available for both training and inference
        return (scores, rec_indices, rec_scores, similarity_matrix, feedback_adjusted)

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
