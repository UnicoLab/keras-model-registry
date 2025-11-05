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
    """Matrix Factorization recommendation model.

    Uses user and item embeddings with dot product similarity for collaborative
    filtering. Embeddings are learned to minimize ranking loss.

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
        - recommendation_indices: Top-K item indices (batch_size, top_k)
        - recommendation_scores: Top-K similarity scores (batch_size, top_k)

    Example:
        ```python
        import keras
        import numpy as np
        from kmr.models import MatrixFactorizationModel

        model = MatrixFactorizationModel(
            num_users=1000,
            num_items=500,
            embedding_dim=32,
            top_k=10
        )

        # Sample data
        user_ids = np.random.randint(0, 1000, (32,))
        item_ids = np.random.randint(0, 500, (32, 500))

        # Get recommendations
        indices, scores = model([user_ids, item_ids])
        print("Recommendation indices:", indices.shape)  # (32, 10)
        print("Recommendation scores:", scores.shape)    # (32, 10)
        ```
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

        # Store custom recommendation metrics for train_step
        self._custom_metrics = []

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
            Tuple of (recommendation_indices, recommendation_scores)
        """
        user_ids, item_ids = inputs

        # Get embeddings
        user_emb, item_emb = self.embedding_layer(
            [user_ids, item_ids],
            training=training,
        )

        # Compute similarities
        # user_emb: (batch_size, embedding_dim)
        # item_emb: (batch_size, num_items, embedding_dim)
        # Reshape for similarity computation: (batch_size, 1, embedding_dim)
        user_emb_exp = ops.expand_dims(user_emb, axis=1)

        # Compute dot product for each item
        # Result: (batch_size, num_items, 1)
        similarities = ops.sum(user_emb_exp * item_emb, axis=-1, keepdims=True)

        # Normalize
        user_norm = ops.sqrt(ops.sum(user_emb_exp**2, axis=-1, keepdims=True) + 1e-8)
        item_norm = ops.sqrt(ops.sum(item_emb**2, axis=-1, keepdims=True) + 1e-8)
        similarities = similarities / (user_norm * item_norm + 1e-8)

        # Squeeze to (batch_size, num_items)
        similarities = ops.squeeze(similarities, axis=-1)

        # Select top-K
        rec_indices, rec_scores = self.selector_layer(similarities)

        return rec_indices, rec_scores

    def compute_similarities(
        self,
        inputs: tuple,
        training: bool | None = None,
    ) -> Any:
        """Compute similarity scores for all items (before top-K selection).

        Useful for training with ranking losses.

        Args:
            inputs: Tuple of (user_ids, item_ids)
            training: Whether in training mode.

        Returns:
            Similarity scores of shape (batch_size, num_items)
        """
        user_ids, item_ids = inputs

        # Get embeddings
        user_emb, item_emb = self.embedding_layer(
            [user_ids, item_ids],
            training=training,
        )

        # Compute similarities
        user_emb_exp = ops.expand_dims(user_emb, axis=1)
        similarities = ops.sum(user_emb_exp * item_emb, axis=-1, keepdims=True)

        # Normalize
        user_norm = ops.sqrt(ops.sum(user_emb_exp**2, axis=-1, keepdims=True) + 1e-8)
        item_norm = ops.sqrt(ops.sum(item_emb**2, axis=-1, keepdims=True) + 1e-8)
        similarities = similarities / (user_norm * item_norm + 1e-8)

        # Squeeze to (batch_size, num_items)
        similarities = ops.squeeze(similarities, axis=-1)

        return similarities

    def train_step(self, data: tuple) -> dict:
        """Custom training step for recommendation learning with ranking loss.

        This allows the model to work with keras.fit() when provided with
        properly formatted data. Supports both supervised (with targets) and
        ranking-based training.

        Note: This uses standard Keras operations only, no TensorFlow imports.

        Args:
            data: Tuple of (inputs, targets) where:
                - inputs: (user_ids, item_ids) with item_ids of shape (batch_size, num_items)
                - targets: Binary labels of shape (batch_size, num_items) where 1=positive, 0=negative
                    OR None for unsupervised learning

        Returns:
            Dictionary of loss values
        """
        inputs, targets = data

        # Unpack inputs
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            user_ids, item_ids = inputs
        else:
            raise ValueError(
                "MatrixFactorizationModel.train_step expects inputs as (user_ids, item_ids)",
            )

        # Get similarity scores for all items
        similarities = self.compute_similarities([user_ids, item_ids], training=True)

        # Compute loss
        if targets is not None:
            # Supervised learning with margin ranking loss
            # Targets are binary labels: 1 for positive items, 0 for negative
            targets_float = ops.cast(targets, similarities.dtype)

            # Compute positive and negative masks
            positive_mask = targets_float > 0.5
            negative_mask = targets_float < 0.5

            # Count positive and negative items per user
            n_positive = ops.sum(
                ops.cast(positive_mask, "float32"),
                axis=-1,
                keepdims=True,
            )  # (batch_size, 1)
            n_negative = ops.sum(
                ops.cast(negative_mask, "float32"),
                axis=-1,
                keepdims=True,
            )  # (batch_size, 1)

            # Compute average positive and negative scores per user
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
                "num_users": self.num_users,
                "num_items": self.num_items,
                "embedding_dim": self.embedding_dim,
                "top_k": self.top_k,
                "l2_reg": self.l2_reg,
            },
        )
        return config
