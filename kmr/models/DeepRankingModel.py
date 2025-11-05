"""Deep Ranking recommendation model.

This module implements a deep neural ranking model that combines user and item
features to predict relevance scores for ranking recommendations.
"""

from typing import Any, Optional
import keras
from keras import layers, ops, Model
from keras.saving import register_keras_serializable
from loguru import logger

from kmr.models._base import BaseModel
from kmr.layers import (
    DeepFeatureRanking,
    TopKRecommendationSelector,
)


@register_keras_serializable(package="kmr.models")
class DeepRankingModel(BaseModel):
    """Deep Neural Ranking recommendation model.

    Uses deep neural networks to score user-item pairs by combining their features.
    Features are concatenated and processed through multiple layers to predict
    relevance scores for ranking recommendations.

    Args:
        user_feature_dim: Dimension of user feature input.
        item_feature_dim: Dimension of item feature input.
        num_items: Number of items to rank.
        hidden_units: List of hidden layer units (default=[128, 64, 32]).
        activation: Activation function for hidden layers (default='relu').
        dropout_rate: Dropout rate for regularization (default=0.3).
        batch_norm: Whether to use batch normalization (default=True).
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
        - recommendation_scores: Top-K ranking scores (batch_size, top_k)

    Example:
        ```python
        import keras
        import numpy as np
        from kmr.models import DeepRankingModel

        model = DeepRankingModel(
            user_feature_dim=64,
            item_feature_dim=64,
            num_items=500,
            hidden_units=[128, 64, 32],
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
        activation: str = "relu",
        dropout_rate: float = 0.3,
        batch_norm: bool = True,
        l2_reg: float = 1e-4,
        top_k: int = 10,
        preprocessing_model: Optional[Model] = None,
        name: str = "deep_ranking_model",
        **kwargs: Any,
    ) -> None:
        """Initialize DeepRankingModel."""
        super().__init__(name=name, preprocessing_model=preprocessing_model, **kwargs)

        self.user_feature_dim = user_feature_dim
        self.item_feature_dim = item_feature_dim
        self.num_items = num_items
        self.hidden_units = hidden_units or [128, 64, 32]
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.l2_reg = l2_reg
        self.top_k = top_k
        self._custom_metrics = []  # Initialize for custom recommendation metrics

        self._validate_params()

        # Combined feature input dimension
        combined_dim = user_feature_dim + item_feature_dim

        # Deep ranking tower
        self.ranking_tower = DeepFeatureRanking(
            hidden_dim=self.hidden_units[0] if self.hidden_units else 128,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            l2_reg=l2_reg,
        )

        # Additional dense layers for ranking
        self.dense_layers = []
        for units in self.hidden_units[1:] if len(self.hidden_units) > 1 else [64, 32]:
            self.dense_layers.append(
                layers.Dense(
                    units,
                    activation=activation,
                    kernel_regularizer=keras.regularizers.l2(l2_reg),
                ),
            )
            if batch_norm:
                self.dense_layers.append(layers.BatchNormalization())
            self.dense_layers.append(layers.Dropout(dropout_rate))

        # Final output layer
        self.output_layer = layers.Dense(1, activation="sigmoid")

        # Top-K selector
        self.selector_layer = TopKRecommendationSelector(k=top_k)

        logger.debug(
            f"Initialized {name} with user_dim={user_feature_dim}, "
            f"item_dim={item_feature_dim}, top_k={top_k}",
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

        batch_size = ops.shape(item_features)[0]
        num_items_actual = ops.shape(item_features)[1]

        # Expand user features to match items
        # user_features: (batch_size, user_feature_dim)
        # -> (batch_size, 1, user_feature_dim)
        # -> (batch_size, num_items, user_feature_dim)
        user_features_exp = ops.expand_dims(user_features, axis=1)
        user_features_repeated = ops.tile(user_features_exp, (1, num_items_actual, 1))

        # Concatenate user and item features
        # (batch_size, num_items, user_feature_dim + item_feature_dim)
        combined_features = ops.concatenate(
            [user_features_repeated, item_features],
            axis=-1,
        )

        # Reshape for processing
        combined_flat = ops.reshape(
            combined_features,
            (-1, self.user_feature_dim + self.item_feature_dim),
        )

        # Process through ranking tower
        scores_flat = self.ranking_tower(combined_flat, training=training)

        # Process through additional dense layers
        x = scores_flat
        for layer_module in self.dense_layers:
            x = layer_module(x, training=training)

        # Final output
        scores_flat = self.output_layer(x)

        # Reshape back to (batch_size, num_items, 1)
        scores = ops.reshape(scores_flat, (batch_size, num_items_actual, 1))

        # Squeeze to (batch_size, num_items)
        scores = ops.squeeze(scores, axis=-1)

        # Select top-K
        rec_indices, rec_scores = self.selector_layer(scores)

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
            Similarity scores of shape (batch_size, num_items)
        """
        user_features, item_features = inputs

        batch_size = ops.shape(item_features)[0]
        num_items_actual = ops.shape(item_features)[1]

        # Expand user features to match items
        user_features_exp = ops.expand_dims(user_features, axis=1)
        user_features_repeated = ops.tile(user_features_exp, (1, num_items_actual, 1))

        # Concatenate user and item features
        combined_features = ops.concatenate(
            [user_features_repeated, item_features],
            axis=-1,
        )

        # Reshape for processing
        combined_flat = ops.reshape(
            combined_features,
            (-1, self.user_feature_dim + self.item_feature_dim),
        )

        # Process through ranking tower
        scores_flat = self.ranking_tower(combined_flat, training=training)

        # Process through additional dense layers
        x = scores_flat
        for layer_module in self.dense_layers:
            x = layer_module(x, training=training)

        # Final output
        scores_flat = self.output_layer(x)

        # Reshape back to (batch_size, num_items, 1)
        scores = ops.reshape(scores_flat, (batch_size, num_items_actual, 1))

        # Squeeze to (batch_size, num_items)
        scores = ops.squeeze(scores, axis=-1)

        return scores

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
            self._custom_metrics = custom_metrics_list
            metrics_to_compile = metrics
        else:
            self._custom_metrics = []
            metrics_to_compile = metrics

        # Call parent compile
        super().compile(metrics=metrics_to_compile, **kwargs)

        # Ensure compiled_metrics is built
        if hasattr(self, "compiled_metrics") and self.compiled_metrics:
            try:
                dummy_y_true = ops.zeros((1, 1), dtype="float32")
                dummy_y_pred = ops.zeros((1, 1), dtype="float32")
                self.compiled_metrics.update_state(dummy_y_true, dummy_y_pred)
            except Exception:
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

        # Compute similarity scores for all items
        similarities = self.compute_similarities(inputs, training=True)
        # similarities shape: (batch_size, num_items)

        # Compute loss
        if targets is not None:
            # Supervised learning: margin ranking loss
            positive_mask = targets > 0.5
            negative_mask = targets < 0.5

            n_positive = ops.sum(
                ops.cast(positive_mask, dtype="float32"),
                axis=-1,
                keepdims=True,
            )
            n_negative = ops.sum(
                ops.cast(negative_mask, dtype="float32"),
                axis=-1,
                keepdims=True,
            )

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
            )
            avg_negative = ops.sum(negative_scores, axis=-1, keepdims=True) / (
                n_negative + 1e-8
            )

            # Margin ranking loss
            margin = 1.0
            loss = ops.mean(
                ops.maximum(0.0, margin - (avg_positive - avg_negative)),
            )
        else:
            # Unsupervised learning: encourage diverse similarity distributions
            loss = -ops.mean(ops.var(similarities, axis=-1))

        # Add regularization losses
        if self.losses:
            loss += ops.sum(self.losses)

        # Prepare metrics output
        metrics_output = {"loss": loss}

        # Compute metrics if targets are provided and custom metrics are configured
        if targets is not None and self._custom_metrics:
            top_k_indices, _ = self.selector_layer(similarities, training=False)

            for metric in self._custom_metrics:
                metric.update_state(targets, top_k_indices)
                metric_result = metric.result()
                metric_name = metric.name if hasattr(metric, "name") else str(metric)
                metrics_output[metric_name] = metric_result

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
                "activation": self.activation,
                "dropout_rate": self.dropout_rate,
                "batch_norm": self.batch_norm,
                "l2_reg": self.l2_reg,
                "top_k": self.top_k,
            },
        )
        return config
