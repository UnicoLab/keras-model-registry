"""Unsupervised geospatial clustering recommendation model.

This module implements a geospatial-based recommendation model using clustering
on location-based features for unsupervised recommendation ranking.
"""

from typing import Any, Optional
from keras import ops, Model
from keras.saving import register_keras_serializable
from loguru import logger

from kmr.models._base import BaseModel
from kmr.layers import (
    HaversineGeospatialDistance,
    SpatialFeatureClustering,
    GeospatialScoreRanking,
    TopKRecommendationSelector,
    ThresholdBasedMasking,
)


@register_keras_serializable(package="kmr.models")
class GeospatialClusteringModel(BaseModel):
    """Unsupervised geospatial clustering recommendation model.

    This model uses geospatial location data to cluster and recommend items
    based on spatial proximity. It's designed for unsupervised learning where
    no explicit user-item ratings are available.

    Args:
        num_items: Number of items to recommend from.
        embedding_dim: Dimension of spatial embeddings (default=32).
        num_clusters: Number of spatial clusters (default=8).
        top_k: Number of top recommendations to return (default=10).
        threshold: Threshold for score filtering (default=0.1).
        entropy_weight: Weight for entropy loss in unsupervised mode (default=0.1).
        variance_weight: Weight for variance loss (default=0.05).
        preprocessing_model: Optional preprocessing model for input features.
        name: Optional name for the model.

    Inputs:
        - user_latitude: User's latitude coordinate (batch_size,)
        - user_longitude: User's longitude coordinate (batch_size,)
        - item_latitudes: Item latitudes (batch_size, num_items)
        - item_longitudes: Item longitudes (batch_size, num_items)

    Outputs:
        Tuple of:
        - recommendation_indices: Top-K item indices (batch_size, top_k)
        - recommendation_scores: Top-K scores (batch_size, top_k)

    Example:
        ```python
        import keras
        import numpy as np
        from kmr.models import GeospatialClusteringModel

        model = GeospatialClusteringModel(
            num_items=100,
            embedding_dim=32,
            num_clusters=8,
            top_k=10
        )

        # Sample geospatial data
        user_lat = np.random.uniform(-90, 90, (32,))
        user_lon = np.random.uniform(-180, 180, (32,))
        item_lats = np.random.uniform(-90, 90, (32, 100))
        item_lons = np.random.uniform(-180, 180, (32, 100))

        # Get recommendations
        indices, scores = model([user_lat, user_lon, item_lats, item_lons])
        print("Recommendation indices:", indices.shape)  # (32, 10)
        print("Recommendation scores:", scores.shape)    # (32, 10)
        ```
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 32,
        num_clusters: int = 8,
        top_k: int = 10,
        threshold: float = 0.1,
        entropy_weight: float = 0.1,
        variance_weight: float = 0.05,
        preprocessing_model: Optional[Model] = None,
        name: str = "geospatial_clustering_model",
        **kwargs: Any,
    ) -> None:
        """Initialize GeospatialClusteringModel."""
        super().__init__(name=name, preprocessing_model=preprocessing_model, **kwargs)

        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        self.top_k = top_k
        self.threshold = threshold
        self.entropy_weight = entropy_weight
        self.variance_weight = variance_weight

        self._validate_params()

        # Initialize layers
        self.distance_layer = HaversineGeospatialDistance()
        self.clustering_layer = SpatialFeatureClustering(num_clusters=num_clusters)
        self.ranking_layer = GeospatialScoreRanking(embedding_dim=embedding_dim)
        self.masking_layer = ThresholdBasedMasking(threshold=threshold)
        self.selector_layer = TopKRecommendationSelector(k=top_k)

        logger.debug(
            f"Initialized {name} with num_items={num_items}, "
            f"embedding_dim={embedding_dim}, num_clusters={num_clusters}, top_k={top_k}",
        )

    def _validate_params(self) -> None:
        """Validate model parameters."""
        if self.num_items <= 0:
            raise ValueError(f"num_items must be positive, got {self.num_items}")
        if self.embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be positive, got {self.embedding_dim}",
            )
        if self.num_clusters <= 0:
            raise ValueError(f"num_clusters must be positive, got {self.num_clusters}")
        if self.top_k <= 0 or self.top_k > self.num_items:
            raise ValueError(
                f"top_k must be between 1 and {self.num_items}, got {self.top_k}",
            )
        if not (0 <= self.threshold <= 1):
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")

    def _compute_user_item_distances(
        self,
        user_lat: Any,
        user_lon: Any,
        item_lats: Any,
        item_lons: Any,
        training: bool | None = None,
    ) -> Any:
        """Compute distances between users and items.

        Args:
            user_lat: User latitudes (batch_size,)
            user_lon: User longitudes (batch_size,)
            item_lats: Item latitudes (batch_size, num_items) or (num_items,)
            item_lons: Item longitudes (batch_size, num_items) or (num_items,)
            training: Whether in training mode.

        Returns:
            Distance matrix (batch_size, num_items)
        """
        # Expand user coordinates to (batch_size, 1) for broadcasting
        user_lat_exp = ops.expand_dims(user_lat, axis=-1)
        user_lon_exp = ops.expand_dims(user_lon, axis=-1)

        # item_lats and item_lons should already be (batch_size, num_items)
        # Calculate pairwise distances using broadcasting
        # Using simple Euclidean distance for now (can be replaced with haversine)
        delta_lat = item_lats - user_lat_exp
        delta_lon = item_lons - user_lon_exp

        # Compute distances
        distances = ops.sqrt(delta_lat**2 + delta_lon**2)
        return distances

    def call(
        self,
        inputs: tuple,
        training: bool | None = None,
    ) -> tuple:
        """Forward pass for recommendation generation.

        Args:
            inputs: Tuple of (user_lat, user_lon, item_lats, item_lons)
            training: Whether in training mode.

        Returns:
            Tuple of (recommendation_indices, recommendation_scores)
        """
        user_lat, user_lon, item_lats, item_lons = inputs

        # Compute user-item distances
        distances = self._compute_user_item_distances(
            user_lat,
            user_lon,
            item_lats,
            item_lons,
            training=training,
        )

        # Perform spatial clustering on distances
        cluster_features = self.clustering_layer(distances, training=training)

        # Generate scores through ranking layer
        scores = self.ranking_layer(cluster_features, training=training)

        # Apply threshold masking
        masked_scores = self.masking_layer(scores)

        # Select top-K recommendations
        rec_indices, rec_scores = self.selector_layer(masked_scores)

        return rec_indices, rec_scores

    def train_step(self, data: tuple) -> dict:
        """Custom training step for unsupervised learning.

        Note: This uses standard Keras operations only, no TensorFlow imports.

        Args:
            data: Training data (inputs, None for unsupervised)

        Returns:
            Dictionary of loss values
        """
        # For unsupervised learning, we rely on the model's built-in losses
        # computed through add_loss() calls in the forward pass
        # This approach avoids needing explicit gradient tape handling

        inputs, _ = data

        # Forward pass - losses will be added via add_loss()
        y_pred = self(inputs, training=True)

        # Get any losses added via add_loss()
        loss = None
        if self.losses:
            loss = ops.sum(self.losses)

        # If no losses were added, compute a simple reconstruction loss
        if loss is None:
            # For unsupervised clustering, use cluster coherence as loss
            user_lat, user_lon, item_lats, item_lons = inputs
            distances = self._compute_user_item_distances(
                user_lat,
                user_lon,
                item_lats,
                item_lons,
                training=True,
            )
            cluster_features = self.clustering_layer(distances, training=True)

            # Entropy loss for diverse clusters
            cluster_probs = ops.softmax(cluster_features, axis=-1)
            entropy = -ops.sum(cluster_probs * ops.log(cluster_probs + 1e-10), axis=-1)
            entropy_loss = -ops.mean(entropy)

            # Variance loss for spread in scores
            scores = self.ranking_layer(cluster_features, training=True)
            score_variance = ops.var(scores)
            variance_loss = -score_variance

            # Total loss
            loss = (
                self.entropy_weight * entropy_loss
                + self.variance_weight * variance_loss
            )

        # Update metrics if defined
        for metric in self.metrics:
            if metric.name != "loss":
                metric.update_state(loss)

        return {"loss": loss}

    def get_config(self) -> dict:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "num_items": self.num_items,
                "embedding_dim": self.embedding_dim,
                "num_clusters": self.num_clusters,
                "top_k": self.top_k,
                "threshold": self.threshold,
                "entropy_weight": self.entropy_weight,
                "variance_weight": self.variance_weight,
            },
        )
        return config
