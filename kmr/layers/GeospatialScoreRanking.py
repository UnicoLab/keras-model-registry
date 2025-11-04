"""Geospatial score ranking layer for location-based recommendation systems.

Ranks items/products based on cluster features with deep neural network
processing and similarity calculation between all pairs.
"""

from typing import Any
from keras import layers, ops
from keras import KerasTensor
from keras.saving import register_keras_serializable

from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class GeospatialScoreRanking(BaseLayer):
    """Ranks items/products based on geospatial cluster features.

    This layer scores items using their cluster probability features through
    dense layers with batch normalization. It computes similarity scores
    between all pairs of items and applies sigmoid normalization.

    Args:
        embedding_dim: Embedding dimension for scoring network (default=32).
        input_dim: Input feature dimension (default=5).
        name: Optional name for the layer.

    Input shape:
        Cluster probabilities of shape (batch_size, input_dim).

    Output shape:
        Ranking scores matrix of shape (batch_size, batch_size) with values in [0, 1].

    Example:
        ```python
        import keras
        from kmr.layers import GeospatialScoreRanking

        # Create sample cluster features
        clusters = keras.random.uniform((32, 5))

        # Rank items
        ranking = GeospatialScoreRanking(embedding_dim=32, input_dim=5)
        scores = ranking(clusters)
        print("Ranking scores shape:", scores.shape)  # (32, 32)
        print("Score range:", scores.numpy().min(), "to", scores.numpy().max())
        ```
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        input_dim: int = 5,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the GeospatialScoreRanking layer.

        Args:
            embedding_dim: Embedding dimension for scoring network.
            input_dim: Input feature dimension.
            name: Name of the layer.
            **kwargs: Additional keyword arguments.
        """
        # Set private attributes first
        self._embedding_dim = embedding_dim
        self._input_dim = input_dim

        # Validate parameters
        self._validate_params()

        # Set public attributes BEFORE calling parent's __init__
        self.embedding_dim = self._embedding_dim
        self.input_dim = self._input_dim
        self.dense1 = None
        self.dense2 = None
        self.batch_norm1 = None
        self.batch_norm2 = None

        # Call parent's __init__
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if not isinstance(self._embedding_dim, int) or self._embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be a positive integer, got {self._embedding_dim}",
            )
        if not isinstance(self._input_dim, int) or self._input_dim <= 0:
            raise ValueError(
                f"input_dim must be a positive integer, got {self._input_dim}",
            )

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Build layer with given input shape.

        Args:
            input_shape: Shape of input cluster features.
        """
        # Create dense layers
        self.dense1 = layers.Dense(self.embedding_dim, activation="relu")
        self.dense2 = layers.Dense(1, activation="sigmoid")

        # Create batch norm layers
        self.batch_norm1 = layers.BatchNormalization(axis=-1)
        self.batch_norm2 = layers.BatchNormalization(axis=-1)

        # Build dense layers
        self.dense1.build(input_shape)
        dense1_output_shape = (input_shape[0], self.embedding_dim)
        self.dense2.build(dense1_output_shape)

        # Build batch norm layers
        self.batch_norm1.build(dense1_output_shape)
        self.batch_norm2.build((input_shape[0], 1))

        super().build(input_shape)

    def call(self, inputs: KerasTensor, training: bool | None = None) -> KerasTensor:
        """Calculate ranking scores from cluster features.

        Args:
            inputs: Cluster probability features of shape (batch_size, input_dim).
            training: Whether in training mode.

        Returns:
            Ranking scores matrix of shape (batch_size, batch_size).
        """
        # First dense layer with batch norm
        x = self.dense1(inputs)
        x = self.batch_norm1(x, training=training)

        # Second dense layer with batch norm
        x = self.dense2(x)
        x = self.batch_norm2(x, training=training)

        # Calculate similarity between all pairs
        similarity = ops.matmul(x, ops.transpose(x))  # (batch_size, batch_size)

        # Scale by embedding dimension for stability
        scaled_similarity = similarity / ops.sqrt(
            ops.cast(self.embedding_dim, similarity.dtype),
        )

        # Convert to probabilities using sigmoid
        scores = ops.sigmoid(scaled_similarity)

        return scores

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "input_dim": self.input_dim,
            },
        )
        return config
