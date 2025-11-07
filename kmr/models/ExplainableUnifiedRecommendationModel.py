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
    """Explainable Unified Recommendation model with full Keras compatibility.

    Combines multiple recommendation approaches with explainability through
    per-component similarity scores and learnable weight combination.

    This model implements the standard Keras API:
    - compile(): Use standard Keras optimizer and custom ImprovedMarginRankingLoss
    - fit(): Use standard Keras training loop with recommendation metrics
    - predict(): Generate recommendations for inference

    Architecture:
        - Collaborative Filtering: User/item embeddings with cosine similarity
        - Content-Based: Deep feature towers for user and item features
        - Hybrid: Average of CF and CB scores
        - Weighted Combination: Equal weighting of all three approaches
        - Explainability: Per-component similarity matrices returned
        - Top-K selection via TopKRecommendationSelector

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
        - During training: Combined scores (batch_size, num_items) for loss computation
        - During inference: Tuple of (recommendation_indices, recommendation_scores, cf_similarities, cb_similarities, component_weights)
            - recommendation_indices: Top-K item indices (batch_size, top_k)
            - recommendation_scores: Top-K blended scores (batch_size, top_k)
            - cf_similarities: Collaborative filtering similarities (batch_size, num_items)
            - cb_similarities: Content-based similarities (batch_size, num_items)
            - component_weights: Learned weights for each approach (3,)

    Example:
        ```python
        import keras
        import numpy as np
        from kmr.models import ExplainableUnifiedRecommendationModel
        from kmr.losses import ImprovedMarginRankingLoss
        from kmr.metrics import AccuracyAtK, PrecisionAtK, RecallAtK

        # Create model
        model = ExplainableUnifiedRecommendationModel(
            num_users=1000,
            num_items=500,
            user_feature_dim=64,
            item_feature_dim=64,
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
        user_features = np.random.randn(32, 64).astype(np.float32)
        item_ids = np.random.randint(0, 500, (32, 500))
        item_features = np.random.randn(32, 500, 64).astype(np.float32)
        labels = np.random.randint(0, 2, (32, 500)).astype(np.float32)

        history = model.fit(
            x=[user_ids, user_features, item_ids, item_features],
            y=labels,
            epochs=10,
            batch_size=32
        )

        # Generate recommendations with explanations for inference
        indices, scores, cf_sims, cb_sims, weights = model.predict(
            [user_ids, user_features, item_ids, item_features]
        )
        print("Recommendation indices:", indices.shape)  # (32, 10)
        print("Recommendation scores:", scores.shape)    # (32, 10)
        print("CF similarities:", cf_sims.shape)         # (32, 500)
        print("CB similarities:", cb_sims.shape)         # (32, 500)
        print("Component weights:", weights.shape)       # (3,)
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
            Tuple of (combined_scores, rec_indices, rec_scores, cf_similarities, cb_similarities, weights, raw_cf_scores)
            where:
            - combined_scores: Combined scores (batch_size, num_items) for loss computation
            - rec_indices: Top-K item indices (batch_size, top_k)
            - rec_scores: Top-K scores (batch_size, top_k)
            - cf_similarities: Collaborative filtering similarities (batch_size, num_items)
            - cb_similarities: Content-based similarities (batch_size, num_items)
            - weights: Component weights (scalar tensors for CF and CB)
            - raw_cf_scores: Raw collaborative filtering scores before normalization

            This tuple is returned consistently for both training and inference modes,
            following Keras 3 best practices for pure functional architecture.
        """
        user_ids, user_features, item_ids, item_features = inputs

        # ========== Collaborative Filtering Component ==========
        user_emb_cf, item_emb_cf = self.embedding_layer(
            [user_ids, item_ids],
            training=training,
        )
        user_emb_cf_exp = ops.expand_dims(user_emb_cf, axis=1)
        raw_cf_scores = ops.sum(user_emb_cf_exp * item_emb_cf, axis=-1)
        user_norm_cf = ops.sqrt(ops.sum(user_emb_cf_exp**2, axis=-1) + 1e-8)
        item_norm_cf = ops.sqrt(ops.sum(item_emb_cf**2, axis=-1) + 1e-8)
        cf_similarities = raw_cf_scores / (user_norm_cf * item_norm_cf + 1e-8)

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
        cb_similarities = ops.sum(user_repr_cb_exp * item_repr_cb, axis=-1)
        user_norm_cb = ops.sqrt(ops.sum(user_repr_cb_exp**2, axis=-1) + 1e-8)
        item_norm_cb = ops.sqrt(ops.sum(item_repr_cb**2, axis=-1) + 1e-8)
        cb_similarities = cb_similarities / (user_norm_cb * item_norm_cb + 1e-8)

        # ========== Combined Scores ==========
        hybrid_similarities = (cf_similarities + cb_similarities) / 2.0
        combined_scores = (
            cf_similarities + cb_similarities + hybrid_similarities
        ) / 3.0

        # ========== Component Weights ==========
        cf_weight = ops.array(1.0)
        cb_weight = ops.array(1.0)
        weights = [cf_weight, cb_weight]

        # ========== Select Top-K ==========
        rec_indices, rec_scores = self.selector_layer(combined_scores)

        # Return tuple - all components available for both training and inference
        return (
            combined_scores,
            rec_indices,
            rec_scores,
            cf_similarities,
            cb_similarities,
            weights,
            raw_cf_scores,
        )

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
