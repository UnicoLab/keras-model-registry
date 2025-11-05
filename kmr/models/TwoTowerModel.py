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
        indices, scores = model([user_features, item_features])
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
    ) -> tuple:
        """Forward pass for recommendation generation.

        Args:
            inputs: Tuple of (user_features, item_features)
            training: Whether in training mode.

        Returns:
            Tuple of (recommendation_indices, recommendation_scores)
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

        # Select top-K
        rec_indices, rec_scores = self.selector_layer(similarities)

        return rec_indices, rec_scores

    def compute_similarities(
        self,
        inputs: tuple,
        training: bool | None = None,
    ) -> Any:
        """Compute similarity scores for all items (before top-K selection).

        Args:
            inputs: Tuple of (user_features, item_features)
            training: Whether in training mode.

        Returns:
            Similarity scores for all items (batch_size, num_items)
        """
        user_features, item_features = inputs

        # Process through towers
        user_repr = self.user_tower(user_features, training=training)

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

        return similarities

    def compile(  # noqa: A003
        self,
        metrics=None,
        **kwargs,
    ) -> None:
        """Compile the model and store custom recommendation metrics.

        Args:
            metrics: List of metrics. Custom recommendation metrics (AccuracyAtK, etc.)
                    will be stored separately for train_step and tracked manually.
            **kwargs: Additional arguments passed to super().compile()
        """
        # Extract custom recommendation metrics (those with 'k' attribute)
        if metrics:
            custom_metrics_list = [
                m for m in metrics if hasattr(m, "k") and hasattr(m, "update_state")
            ]
            standard_metrics = [
                m
                for m in metrics
                if not (hasattr(m, "k") and hasattr(m, "update_state"))
            ]
            self._custom_metrics = custom_metrics_list
            # Include all metrics in compile to ensure compiled_metrics is built
            # Custom metrics will be updated manually in train_step with correct format
            metrics_to_compile = metrics
        else:
            self._custom_metrics = []
            metrics_to_compile = metrics

        # Call parent compile (all metrics included so compiled_metrics is built)
        super().compile(metrics=metrics_to_compile, **kwargs)

        # Ensure compiled_metrics is built by updating it once with dummy data
        # This prevents "metric has not yet been built" errors during fit()
        # Keras tries to get results from compiled_metrics before train_step runs
        if hasattr(self, "compiled_metrics") and self.compiled_metrics:
            try:
                # Build compiled_metrics with dummy data to mark it as built
                # Always update once to ensure it's built, regardless of built attribute
                dummy_y_true = ops.zeros((1, 1), dtype="float32")
                dummy_y_pred = ops.zeros((1, 1), dtype="float32")
                self.compiled_metrics.update_state(dummy_y_true, dummy_y_pred)
            except Exception:
                # If building fails, it will be handled during training
                pass

    def train_step(self, data: tuple) -> dict:
        """Custom training step for recommendation learning with ranking loss.

        Args:
            data: Tuple of (inputs, targets) where:
                - inputs: (user_features, item_features)
                - targets: Binary labels (batch_size, num_items) indicating positive items

        Returns:
            Dictionary of loss and metrics
        """
        inputs, targets = data
        user_features, item_features = inputs

        # Compute similarities for all items
        similarities = self.compute_similarities(inputs, training=True)
        # similarities shape: (batch_size, num_items)

        # Compute loss
        if targets is not None:
            # Supervised learning: margin ranking loss
            # Positive items should have higher scores than negative items
            positive_mask = targets > 0.5  # (batch_size, num_items)
            negative_mask = targets < 0.5  # (batch_size, num_items)

            n_positive = ops.sum(
                ops.cast(positive_mask, dtype="float32"),
                axis=-1,
                keepdims=True,
            )  # (batch_size, 1)
            n_negative = ops.sum(
                ops.cast(negative_mask, dtype="float32"),
                axis=-1,
                keepdims=True,
            )  # (batch_size, 1)

            positive_scores = ops.where(
                positive_mask,
                similarities,
                ops.zeros_like(similarities),
            )
            negative_scores = ops.where(
                negative_mask,
                similarities,
                ops.zeros_like(similarities),
            )

            avg_positive = ops.sum(positive_scores, axis=-1, keepdims=True) / (
                n_positive + 1e-8
            )  # (batch_size, 1)
            avg_negative = ops.sum(negative_scores, axis=-1, keepdims=True) / (
                n_negative + 1e-8
            )  # (batch_size, 1)

            # Margin ranking loss: positive should be higher than negative by margin
            margin = 1.0
            loss = ops.mean(
                ops.maximum(0.0, margin - (avg_positive - avg_negative)),
            )
        else:
            # Unsupervised learning: encourage diverse similarity distributions
            loss = -ops.mean(ops.var(similarities, axis=-1))

        # Add regularization losses from layers
        if self.losses:
            loss += ops.sum(self.losses)

        # Prepare metrics output
        metrics_output = {"loss": loss}

        # Compute metrics if targets are provided and custom metrics are configured
        if targets is not None and self._custom_metrics:
            # Get top-K recommendations for metrics
            # similarities shape: (batch_size, num_items)
            # Get top-K indices
            top_k_indices, _ = self.selector_layer(similarities, training=False)
            # top_k_indices shape: (batch_size, top_k)

            # Update custom recommendation metrics manually
            # These metrics need special inputs: (y_true as binary labels, y_pred as top-K indices)
            for metric in self._custom_metrics:
                # Custom recommendation metrics expect (y_true, y_pred)
                # where y_true is (batch_size, num_items) and y_pred is (batch_size, k)
                metric.update_state(targets, top_k_indices)

                # Get metric result to include in training output
                metric_result = metric.result()
                metric_name = metric.name if hasattr(metric, "name") else str(metric)
                metrics_output[metric_name] = metric_result

        # Return loss and metrics (Keras will handle gradient computation automatically)
        return metrics_output

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
