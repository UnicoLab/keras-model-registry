"""Haversine geospatial distance layer for recommendation systems.

Calculates pairwise distances between latitude/longitude coordinates using
the haversine formula, useful for location-aware recommendations.
"""

from typing import Any
from keras import ops
from keras import KerasTensor
from keras.saving import register_keras_serializable

from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class HaversineGeospatialDistance(BaseLayer):
    """Calculates haversine distance between latitude/longitude coordinates.

    This layer computes pairwise distances between geographic coordinates using
    the haversine formula, which calculates the great-circle distance between
    two points on a sphere given their longitudes and latitudes.

    The distance is normalized to the range [0, 1] for better numerical stability
    during training. Input coordinates should be in radians.

    Args:
        earth_radius: Radius of Earth in kilometers (default=6371).
        name: Optional name for the layer.

    Input:
        Tuple of 4 tensors:
        - lat1: Source latitudes, shape (batch_size,) in radians
        - lon1: Source longitudes, shape (batch_size,) in radians
        - lat2: Target latitudes, shape (batch_size,) in radians
        - lon2: Target longitudes, shape (batch_size,) in radians

    Output shape:
        Distance matrix of shape (batch_size, batch_size), values in [0, 1].

    Example:
        ```python
        import keras
        import numpy as np
        from kmr.layers import HaversineGeospatialDistance

        # Create sample coordinates in radians
        batch_size = 32
        lat1 = keras.random.uniform((batch_size,), minval=-np.pi/2, maxval=np.pi/2)
        lon1 = keras.random.uniform((batch_size,), minval=-np.pi, maxval=np.pi)
        lat2 = keras.random.uniform((batch_size,), minval=-np.pi/2, maxval=np.pi/2)
        lon2 = keras.random.uniform((batch_size,), minval=-np.pi, maxval=np.pi)

        # Calculate distances
        layer = HaversineGeospatialDistance(earth_radius=6371)
        distances = layer([lat1, lon1, lat2, lon2])
        print("Distance matrix shape:", distances.shape)  # (32, 32)
        print("Distance range:", distances.numpy().min(), "to", distances.numpy().max())
        ```
    """

    def __init__(
        self,
        earth_radius: float = 6371.0,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the HaversineGeospatialDistance layer.

        Args:
            earth_radius: Radius of Earth in kilometers.
            name: Name of the layer.
            **kwargs: Additional keyword arguments.
        """
        # Set private attributes first
        self._earth_radius = float(earth_radius)

        # Validate parameters
        self._validate_params()

        # Set public attributes BEFORE calling parent's __init__
        self.earth_radius = self._earth_radius

        # Call parent's __init__
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if not isinstance(self._earth_radius, int | float) or self._earth_radius <= 0:
            raise ValueError(f"earth_radius must be positive, got {self._earth_radius}")

    def call(
        self,
        inputs: tuple[KerasTensor, KerasTensor, KerasTensor, KerasTensor],
    ) -> KerasTensor:
        """Calculate haversine distances between coordinates.

        Args:
            inputs: Tuple of (lat1, lon1, lat2, lon2).

        Returns:
            Normalized distance matrix of shape (batch_size, batch_size).
        """
        lat1, lon1, lat2, lon2 = inputs

        # Reshape for broadcasting: (batch_size, 1) and (batch_size,) -> (batch_size, batch_size)
        lat1 = ops.reshape(lat1, [-1, 1])
        lon1 = ops.reshape(lon1, [-1, 1])
        lat2 = ops.reshape(lat2, [-1])
        lon2 = ops.reshape(lon2, [-1])

        # Calculate differences
        delta_lat = ops.expand_dims(lat2, 1) - lat1
        delta_lon = ops.expand_dims(lon2, 1) - lon1

        # Haversine formula components
        a = (
            ops.sin(delta_lat / 2) ** 2
            + ops.cos(lat1)
            * ops.cos(ops.expand_dims(lat2, 1))
            * ops.sin(delta_lon / 2) ** 2
        )

        c = 2 * ops.arctan2(ops.sqrt(a), ops.sqrt(1 - a))
        distances = self.earth_radius * c

        # Normalize to [0, 1] range
        max_dist = ops.max(distances)
        min_dist = ops.min(distances)
        epsilon = 1e-6
        normalized_distances = (distances - min_dist) / (max_dist - min_dist + epsilon)

        return normalized_distances

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "earth_radius": self.earth_radius,
            },
        )
        return config
