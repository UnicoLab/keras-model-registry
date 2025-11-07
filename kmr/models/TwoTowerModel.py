"""Two-Tower recommendation model.

This module implements a two-tower architecture with separate user and item
feature processing towers, combining their representations for similarity-based
recommendation.
"""

from typing import Any, Optional
from keras import ops, Model
from keras.saving import register_keras_serializable
from loguru import logger

from kmr.models._base import BaseModel
from kmr.layers import (
    DeepFeatureTower,
    NormalizedDotProductSimilarity,
    TopKRecommendationSelector,
)


@register_keras_serializable(package="kmr.models")
class TwoTowerModel(BaseModel):
    """Two-Tower recommendation model.

    Implements a two-tower architecture with separate neural network towers for
    processing user and item features. The towers process their respective inputs
    independently and the representations are combined using normalized dot product
    similarity for ranking.

    Args:
        user_feature_dim: Dimension of user feature input.
        item_feature_dim: Dimension of item feature input.
        num_items: Number of items to rank.
        hidden_units: Hidden units for each dense layer in towers (default=[64, 32]).
        output_dim: Output dimension of towers (default=32).
        activation: Activation function for hidden layers (default='relu').
        dropout_rate: Dropout rate for regularization (default=0.2).
        l2_reg: L2 regularization factor (default=1e-4).
        top_k: Number of top recommendations to return (default=10).
        preprocessing_model: Optional preprocessing model for input features.
        name: Optional name for the model.

    Inputs:
        - user_features: User feature vectors (batch_size, user_feature_dim)
        - item_features: Item feature vectors (batch_size, num_items, item_feature_dim)

    Outputs:
        Tuple of:
        - similarities: All item similarity scores (batch_size, num_items)
        - recommendation_indices: Top-K item indices (batch_size, top_k)
        - recommendation_scores: Top-K similarity scores (batch_size, top_k)

    Example:
        ```python
        import keras
        import numpy as np
        from kmr.models import TwoTowerModel

        model = TwoTowerModel(
            user_feature_dim=64,
            item_feature_dim=64,
            num_items=500,
            hidden_units=[64, 32],
            output_dim=32,
            top_k=10
        )

        # Sample data
        user_features = np.random.randn(32, 64)
        item_features = np.random.randn(32, 500, 64)

        # Get recommendations
        similarities, indices, scores = model([user_features, item_features])
        print("Similarities:", similarities.shape)       # (32, 500)
        print("Recommendation indices:", indices.shape)  # (32, 10)
        print("Recommendation scores:", scores.shape)    # (32, 10)
        ```
    """

    def __init__(
        self,
        user_feature_dim: int,
        item_feature_dim: int,
        num_items: int,
        hidden_units: list | None = None,
        output_dim: int = 32,
        activation: str = "relu",
        dropout_rate: float = 0.2,
        l2_reg: float = 1e-4,
        top_k: int = 10,
        preprocessing_model: Optional[Model] = None,
        name: str = "two_tower_model",
        **kwargs: Any,
    ) -> None:
        """Initialize TwoTowerModel."""
        super().__init__(name=name, preprocessing_model=preprocessing_model, **kwargs)

        self.user_feature_dim = user_feature_dim
        self.item_feature_dim = item_feature_dim
        self.num_items = num_items
        self.hidden_units = hidden_units or [64, 32]
        self.output_dim = output_dim
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.top_k = top_k

        self._validate_params()

        # User tower
        self.user_tower = DeepFeatureTower(
            units=output_dim,
            hidden_layers=len(self.hidden_units),
            activation=activation,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            name="user_tower",
        )

        # Item tower
        self.item_tower = DeepFeatureTower(
            units=output_dim,
            hidden_layers=len(self.hidden_units),
            activation=activation,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            name="item_tower",
        )

        # Similarity computation
        self.similarity_layer = NormalizedDotProductSimilarity()

        # Top-K selector
        self.selector_layer = TopKRecommendationSelector(k=top_k)

        logger.debug(
            f"Initialized {name} with user_dim={user_feature_dim}, "
            f"item_dim={item_feature_dim}, output_dim={output_dim}, top_k={top_k}",
        )

    def _validate_params(self) -> None:
        """Validate model parameters."""
        if self.user_feature_dim <= 0:
            raise ValueError(
                f"user_feature_dim must be positive, got {self.user_feature_dim}",
            )
        if self.item_feature_dim <= 0:
            raise ValueError(
                f"item_feature_dim must be positive, got {self.item_feature_dim}",
            )
        if self.num_items <= 0:
            raise ValueError(f"num_items must be positive, got {self.num_items}")
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {self.output_dim}")
        if not (0 <= self.dropout_rate <= 1):
            raise ValueError(f"dropout_rate must be in [0, 1], got {self.dropout_rate}")
        if self.l2_reg < 0:
            raise ValueError(f"l2_reg must be non-negative, got {self.l2_reg}")
        if self.top_k <= 0 or self.top_k > self.num_items:
            raise ValueError(
                f"top_k must be between 1 and {self.num_items}, got {self.top_k}",
            )

    def call(
        self,
        inputs: tuple,
        training: bool | None = None,
    ) -> dict:
        """Forward pass for recommendation generation.

        Args:
            inputs: Tuple of (user_features, item_features)
            training: Whether in training mode (not used, always returns full dict for Keras compatibility)

        Returns:
            Dictionary with keys:
            - 'similarities': All item scores (batch_size, num_items) for loss computation
            - 'rec_indices': Top-K item indices (batch_size, top_k)
            - 'rec_scores': Top-K scores (batch_size, top_k)

            Returns a consistent dictionary for both training and inference modes,
            following Keras 3 best practices. Keras automatically uses 'similarities'
            for loss computation when configured.
        """
        user_features, item_features = inputs

        # Process through towers
        # user_features: (batch_size, user_feature_dim) -> (batch_size, output_dim)
        user_repr = self.user_tower(user_features, training=training)

        # item_features: (batch_size, num_items, item_feature_dim) ->
        #                (batch_size, num_items, output_dim)
        batch_size = ops.shape(item_features)[0]
        num_items_actual = ops.shape(item_features)[1]

        # Reshape items for tower processing
        item_features_flat = ops.reshape(
            item_features,
            (-1, self.item_feature_dim),
        )  # (batch_size*num_items, item_feature_dim)

        item_repr_flat = self.item_tower(item_features_flat, training=training)

        # Reshape back
        item_repr = ops.reshape(
            item_repr_flat,
            (batch_size, num_items_actual, self.output_dim),
        )

        # Expand user representation for broadcasting
        user_repr_exp = ops.expand_dims(
            user_repr,
            axis=1,
        )  # (batch_size, 1, output_dim)

        # Compute similarities using dot product
        similarities = ops.sum(
            user_repr_exp * item_repr,
            axis=-1,
        )  # (batch_size, num_items)

        # Normalize
        user_norm = ops.sqrt(ops.sum(user_repr_exp**2, axis=-1, keepdims=True) + 1e-8)
        item_norm = ops.sqrt(ops.sum(item_repr**2, axis=-1, keepdims=True) + 1e-8)
        similarities = similarities / (user_norm[:, 0, :] * item_norm[:, :, 0] + 1e-8)

        # Select top-K recommendations
        rec_indices, rec_scores = self.selector_layer(similarities)

        # Return based on mode for Keras compatibility
        return (similarities, rec_indices, rec_scores)

    def get_config(self) -> dict:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "user_feature_dim": self.user_feature_dim,
                "item_feature_dim": self.item_feature_dim,
                "num_items": self.num_items,
                "hidden_units": self.hidden_units,
                "output_dim": self.output_dim,
                "activation": self.activation,
                "dropout_rate": self.dropout_rate,
                "l2_reg": self.l2_reg,
                "top_k": self.top_k,
            },
        )
        return config
