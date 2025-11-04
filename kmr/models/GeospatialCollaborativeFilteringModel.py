"""Geospatial collaborative filtering recommendation model with masking.

This module implements an extended geospatial recommendation model that combines
geospatial clustering with masking for more sophisticated filtering.
"""

from typing import Any, Optional
from keras import layers, ops, Model
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
class GeospatialCollaborativeFilteringModel(BaseModel):
    """Geospatial collaborative filtering with advanced masking.

    This model extends the geospatial clustering approach with additional masking
    and filtering capabilities for more sophisticated recommendation ranking. It
    combines spatial proximity with learnable masking patterns.

    Args:
        num_items: Number of items to recommend from.
        embedding_dim: Dimension of spatial embeddings (default=32).
        num_clusters: Number of spatial clusters (default=8).
        top_k: Number of top recommendations to return (default=10).
        threshold: Threshold for initial score filtering (default=0.1).
        mask_threshold: Threshold for mask filtering (default=0.2).
        entropy_weight: Weight for entropy loss (default=0.1).
        variance_weight: Weight for variance loss (default=0.05).
        mask_weight: Weight for mask regularization (default=0.05).
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
        - mask_features: Learned mask features for interpretability (batch_size, num_clusters)

    Example:
        ```python
        import keras
        import numpy as np
        from kmr.models import GeospatialCollaborativeFilteringModel

        model = GeospatialCollaborativeFilteringModel(
            num_items=100,
            embedding_dim=32,
            num_clusters=8,
            top_k=10
        )

        # Sample geospatial data with user preferences
        user_lat = np.random.uniform(-90, 90, (32,))
        user_lon = np.random.uniform(-180, 180, (32,))
        item_lats = np.random.uniform(-90, 90, (32, 100))
        item_lons = np.random.uniform(-180, 180, (32, 100))

        # Get recommendations with masking
        indices, scores, masks = model([user_lat, user_lon, item_lats, item_lons])
        print("Recommendation indices:", indices.shape)  # (32, 10)
        print("Recommendation scores:", scores.shape)    # (32, 10)
        print("Mask features:", masks.shape)             # (32, 8)
        ```
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 32,
        num_clusters: int = 8,
        top_k: int = 10,
        threshold: float = 0.1,
        mask_threshold: float = 0.2,
        entropy_weight: float = 0.1,
        variance_weight: float = 0.05,
        mask_weight: float = 0.05,
        preprocessing_model: Optional[Model] = None,
        name: str = "geospatial_collaborative_filtering_model",
        **kwargs: Any,
    ) -> None:
        """Initialize GeospatialCollaborativeFilteringModel."""
        super().__init__(name=name, preprocessing_model=preprocessing_model, **kwargs)

        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        self.top_k = top_k
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self.entropy_weight = entropy_weight
        self.variance_weight = variance_weight
        self.mask_weight = mask_weight

        self._validate_params()

        # Initialize layers
        self.distance_layer = HaversineGeospatialDistance()
        self.clustering_layer = SpatialFeatureClustering(num_clusters=num_clusters)
        self.ranking_layer = GeospatialScoreRanking(embedding_dim=embedding_dim)
        self.masking_layer = ThresholdBasedMasking(threshold=threshold)
        self.mask_filter_layer = ThresholdBasedMasking(threshold=mask_threshold)
        self.selector_layer = TopKRecommendationSelector(k=top_k)

        # Learnable mask generation
        self.mask_generator = layers.Dense(
            num_clusters,
            activation="sigmoid",
            name="mask_generator",
        )

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
        if not (0 <= self.mask_threshold <= 1):
            raise ValueError(
                f"mask_threshold must be in [0, 1], got {self.mask_threshold}",
            )

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
            item_lats: Item latitudes (batch_size, num_items)
            item_lons: Item longitudes (batch_size, num_items)
            training: Whether in training mode.

        Returns:
            Distance matrix (batch_size, num_items)
        """
        # Expand user coordinates to (batch_size, 1) for broadcasting
        user_lat_exp = ops.expand_dims(user_lat, axis=-1)
        user_lon_exp = ops.expand_dims(user_lon, axis=-1)

        # Calculate pairwise distances using broadcasting
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
        """Forward pass for recommendation generation with masking.

        Args:
            inputs: Tuple of (user_lat, user_lon, item_lats, item_lons)
            training: Whether in training mode.

        Returns:
            Tuple of (recommendation_indices, recommendation_scores, mask_features)
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

        # Perform spatial clustering
        cluster_features = self.clustering_layer(distances, training=training)

        # Generate learnable masks based on mean of cluster features
        # cluster_features shape: (batch_size, n_clusters_actual)
        mask_input = ops.mean(
            cluster_features,
            axis=1,
            keepdims=True,
        )  # (batch_size, 1)
        masks = self.mask_generator(
            mask_input,
            training=training,
        )  # (batch_size, num_clusters)

        # Apply mask to cluster features by reducing with mask weights
        # Reshape masks for broadcasting: (batch_size, num_clusters) -> (batch_size, 1)
        # Use mean of masks as a scalar multiplier
        mask_factor = ops.mean(masks, axis=1, keepdims=True)  # (batch_size, 1)
        masked_cluster_features = cluster_features * mask_factor

        # Generate scores through ranking layer
        scores = self.ranking_layer(masked_cluster_features, training=training)

        # Apply threshold masking
        masked_scores = self.masking_layer(scores)

        # Apply mask filter for additional filtering
        filtered_scores = self.mask_filter_layer(masked_scores)

        # Select top-K recommendations
        rec_indices, rec_scores = self.selector_layer(filtered_scores)

        return rec_indices, rec_scores, masks

    def train_step(self, data: tuple) -> dict:
        """Custom training step for unsupervised learning with masking.

        Note: This uses standard Keras operations only, no TensorFlow imports.

        Args:
            data: Training data (inputs, None for unsupervised)

        Returns:
            Dictionary of loss values
        """
        inputs, _ = data

        # Forward pass
        y_pred = self(inputs, training=True)

        # Get any losses added via add_loss()
        loss = None
        if self.losses:
            loss = ops.sum(self.losses)

        # If no losses were added, compute unsupervised losses
        if loss is None:
            user_lat, user_lon, item_lats, item_lons = inputs

            # Compute distances
            distances = self._compute_user_item_distances(
                user_lat,
                user_lon,
                item_lats,
                item_lons,
                training=True,
            )

            # Clustering step
            cluster_features = self.clustering_layer(distances, training=True)

            # Generate masks
            mask_input = ops.mean(cluster_features, axis=1, keepdims=True)
            masks = self.mask_generator(mask_input, training=True)

            # Apply mask factor to cluster features
            mask_factor = ops.mean(masks, axis=1, keepdims=True)
            masked_cluster_features = cluster_features * mask_factor

            # Entropy loss: encourage diverse clusters
            cluster_probs = ops.softmax(masked_cluster_features, axis=-1)
            entropy = -ops.sum(cluster_probs * ops.log(cluster_probs + 1e-10), axis=-1)
            entropy_loss = -ops.mean(entropy)

            # Variance loss: encourage spread in scores
            scores = self.ranking_layer(masked_cluster_features, training=True)
            score_variance = ops.var(scores)
            variance_loss = -score_variance

            # Mask regularization: encourage sparse masks
            mask_sparsity = ops.mean(ops.abs(masks))
            mask_loss = mask_sparsity

            # Total unsupervised loss
            loss = (
                self.entropy_weight * entropy_loss
                + self.variance_weight * variance_loss
                + self.mask_weight * mask_loss
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
                "mask_threshold": self.mask_threshold,
                "entropy_weight": self.entropy_weight,
                "variance_weight": self.variance_weight,
                "mask_weight": self.mask_weight,
            },
        )
        return config
