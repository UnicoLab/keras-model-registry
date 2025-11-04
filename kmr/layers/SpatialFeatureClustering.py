"""Spatial feature clustering layer for geospatial recommendation systems.

Performs learnable clustering based on spatial distance matrices using
batch normalization and softmax for probability distributions.
"""

from typing import Any
from keras import layers, ops
from keras import KerasTensor
from keras.saving import register_keras_serializable

from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class SpatialFeatureClustering(BaseLayer):
    """Performs learnable clustering based on spatial distance matrix.

    This layer uses a distance matrix (typically from haversine calculation)
    to create cluster probabilities via learnable weight transformations,
    batch normalization, and softmax activation. Useful for grouping
    geospatial items into clusters.

    Args:
        n_clusters: Number of clusters to create (default=5).
        name: Optional name for the layer.

    Input shape:
        Distance matrix of shape (batch_size, batch_size).

    Output shape:
        Cluster probabilities of shape (batch_size, n_clusters).

    Example:
        ```python
        import keras
        from kmr.layers import SpatialFeatureClustering

        # Create sample distance matrix
        distances = keras.random.uniform((32, 32))

        # Create clusters
        clustering = SpatialFeatureClustering(n_clusters=5)
        clusters = clustering(distances)
        print("Cluster probabilities shape:", clusters.shape)  # (32, 5)
        print("Probabilities sum to 1:", clusters.numpy().sum(axis=1))
        ```
    """

    def __init__(
        self,
        n_clusters: int = 5,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the SpatialFeatureClustering layer.

        Args:
            n_clusters: Number of clusters.
            name: Name of the layer.
            **kwargs: Additional keyword arguments.
        """
        # Set private attributes first
        self._n_clusters = n_clusters

        # Validate parameters
        self._validate_params()

        # Set public attributes BEFORE calling parent's __init__
        self.n_clusters = self._n_clusters
        self.batch_norm = None
        self.cluster_weights = None

        # Call parent's __init__
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if not isinstance(self._n_clusters, int) or self._n_clusters <= 0:
            raise ValueError(
                f"n_clusters must be a positive integer, got {self._n_clusters}",
            )

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Build layer with given input shape.

        Args:
            input_shape: Shape of input distance matrix (batch_size, batch_size).
        """
        # Create learnable cluster weight matrix
        self.cluster_weights = self.add_weight(
            name="cluster_weights",
            shape=(self.n_clusters, self.n_clusters),
            initializer="random_normal",
            trainable=True,
        )

        # Initialize batch normalization layer
        self.batch_norm = layers.BatchNormalization(axis=-1)
        self.batch_norm.build((input_shape[0], self.n_clusters))

        super().build(input_shape)

    def call(self, inputs: KerasTensor, training: bool | None = None) -> KerasTensor:
        """Calculate cluster probabilities from distance matrix.

        Args:
            inputs: Distance matrix of shape (batch_size, batch_size).
            training: Whether in training mode.

        Returns:
            Cluster probabilities of shape (batch_size, n_clusters).
        """
        # Extract features from distance matrix
        features = ops.mean(inputs, axis=1, keepdims=True)  # (batch_size, 1)
        features = ops.tile(features, [1, 3])  # (batch_size, 3)

        # Project to cluster space
        cluster_logits = ops.matmul(
            features,
            self.cluster_weights[:3, :],
        )  # (batch_size, n_clusters)

        # Apply batch normalization
        normalized = self.batch_norm(cluster_logits, training=training)

        # Convert to probabilities
        cluster_probs = ops.softmax(normalized, axis=-1)

        return cluster_probs

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "n_clusters": self.n_clusters,
            },
        )
        return config
