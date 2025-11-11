"""Geospatial Margin Ranking Loss for location-aware recommendation systems.

This module implements a margin ranking loss that incorporates geospatial distance
penalties, making it suitable for location-aware recommendation tasks where nearby
items should be preferred.
"""

from typing import Any

import keras
from keras import ops
from keras.saving import register_keras_serializable
from loguru import logger

from kmr.losses.improved_margin_ranking_loss import ImprovedMarginRankingLoss


@register_keras_serializable(package="kmr.losses")
class GeospatialMarginLoss(ImprovedMarginRankingLoss):
    """Geospatial Margin Ranking Loss for location-aware recommendations.

    Extends ImprovedMarginRankingLoss to include distance-based penalties. This loss
    encourages the model to rank items that are closer to the user higher than those
    farther away, while still maintaining the margin-based ranking objective.

    The combined loss is:
        margin_loss + distance_weight * distance_penalty

    Where distance_penalty is the average distance weighted by item labels.

    Args:
        margin: The margin threshold for ranking (default=1.0).
        distance_weight: Weight for distance penalty term (default=0.1).
        max_min_weight: Weight for max-min margin loss (default=0.7).
        avg_weight: Weight for average margin loss (default=0.3).
        name: Name of the loss (default="geospatial_margin_loss").

    Input Format:
        y_true: Binary labels (batch_size, num_items), 1 = positive/relevant item
        y_pred: Concatenated [similarities, distances]
                Shape: (batch_size, num_items + 1) where last column is distance
                OR: Shape: (batch_size, num_items*2) with distances interleaved

    Example:
        ```python
        import keras
        from kmr.losses import GeospatialMarginLoss

        loss = GeospatialMarginLoss(
            margin=1.0,
            distance_weight=0.1,
            max_min_weight=0.7,
            avg_weight=0.3
        )

        # y_true: binary labels (batch_size, num_items)
        y_true = keras.ops.array([[1, 0, 1, 0, 0]])

        # y_pred: [similarities, distances] concatenated
        # Shape: (batch_size, num_items*2) or (batch_size, num_items+1)
        similarities = keras.ops.array([[0.8, 0.2, 0.7, 0.1, 0.0]])
        distances = keras.ops.array([[0.1, 0.5, 0.2, 0.8, 0.9]])
        y_pred = keras.ops.concatenate([similarities, distances], axis=-1)

        loss_value = loss(y_true, y_pred)
        ```

    Mathematical Formulation:
        L = L_margin(y_true, similarities) + w_dist * L_distance(distances, y_true)

        where:
        - L_margin is the improved margin ranking loss
        - L_distance = sum(distances * y_true) / (sum(y_true) + epsilon)
        - w_dist is the distance weight parameter

    When to Use:
        - Location-based recommendations (restaurants, stores, hotels)
        - Geospatial queries with distance constraints
        - Scenarios where item proximity affects relevance
        - Multi-objective ranking with distance penalties

    Advantages:
        - Incorporates geospatial constraints naturally
        - Flexible distance weighting for different scenarios
        - Maintains all benefits of ImprovedMarginRankingLoss
        - Scalable to large item catalogs with spatial information
    """

    def __init__(
        self,
        margin: float = 1.0,
        distance_weight: float = 0.1,
        max_min_weight: float = 0.7,
        avg_weight: float = 0.3,
        name: str = "geospatial_margin_loss",
        **kwargs: Any,
    ) -> None:
        """Initialize GeospatialMarginLoss.

        Args:
            margin: The margin threshold for ranking.
            distance_weight: Weight for distance penalty term.
            max_min_weight: Weight for max-min margin loss component.
            avg_weight: Weight for average margin loss component.
            name: Name of the loss.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(
            margin=margin,
            max_min_weight=max_min_weight,
            avg_weight=avg_weight,
            name=name,
            **kwargs,
        )
        self.distance_weight = distance_weight

        logger.debug(
            f"Initialized GeospatialMarginLoss with margin={margin}, "
            f"distance_weight={distance_weight}, max_min_weight={max_min_weight}, "
            f"avg_weight={avg_weight}, name={name}",
        )

    def call(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Compute geospatial margin ranking loss.

        Args:
            y_true: Binary labels of shape (batch_size, num_items) where 1 = positive item.
            y_pred: Can be:
                - Concatenated [similarities, distances] of shape:
                  - (batch_size, num_items*2): similarities and distances concatenated
                  - (batch_size, num_items+1): similarities followed by mean distance
                - Tuple of (masked_scores, indices, scores, masks) from unified model output
                - List format of the same

        Returns:
            Scalar loss value combining margin loss and distance penalty.

        Raises:
            ValueError: If y_pred has unexpected shape or invalid values.
        """
        # Extract concatenated similarities/distances from tuple if model returns unified output
        if isinstance(y_pred, (tuple, list)):
            y_pred = y_pred[0]  # Extract first element from tuple (masked_scores)

        # Determine shape and extract similarities and distances
        num_features = ops.shape(y_pred)[-1]
        num_items = ops.shape(y_true)[-1]

        # Handle different input formats
        # Format 1: [sim_1, sim_2, ..., sim_n, dist_1, dist_2, ..., dist_n]
        # Format 2: [sim_1, sim_2, ..., sim_n, mean_dist]
        if num_features == num_items * 2:
            # Split equally: first half is similarities, second half is distances
            similarities = y_pred[..., :num_items]
            distances = y_pred[..., num_items:]
            logger.debug("Using concatenated format: [similarities, distances]")
        elif num_features == num_items + 1:
            # Last column is distance (broadcasted or single value)
            similarities = y_pred[..., :num_items]
            distances = ops.expand_dims(y_pred[..., -1], axis=-1)
            # Broadcast to match num_items
            distances = ops.tile(distances, [1, num_items])
            logger.debug("Using single distance format, broadcasted to match items")
        else:
            raise ValueError(
                f"Invalid y_pred shape. Expected shape ending in {num_items} or {num_items * 2}, "
                f"but got {num_features}. "
                f"y_pred should be either [similarities, distances] concatenated "
                f"or [similarities, mean_distance].",
            )

        # Compute base margin ranking loss using parent class method
        margin_loss = super().call(y_true, similarities)

        # Compute distance penalty
        # Penalize distances for recommended (positive) items
        # Formula: mean(distances * y_true) / (sum(y_true) + epsilon)
        distance_penalty = self._compute_distance_penalty(y_true, distances)

        # Combine losses
        total_loss = margin_loss + self.distance_weight * distance_penalty

        logger.debug(
            f"Geospatial loss computed: margin_loss={margin_loss:.4f}, "
            f"distance_penalty={distance_penalty:.4f}, total_loss={total_loss:.4f}",
        )

        return total_loss

    def _compute_distance_penalty(
        self,
        y_true: keras.KerasTensor,
        distances: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Compute distance penalty for geospatial recommendations.

        The penalty is the weighted average distance of positive items:
            penalty = sum(distances * y_true) / (sum(y_true) + epsilon)

        Higher distances for positive items result in higher penalty.

        Args:
            y_true: Binary labels (batch_size, num_items)
            distances: Distance matrix (batch_size, num_items)

        Returns:
            Scalar distance penalty value
        """
        # Cast y_true to float for computation
        y_true_float = ops.cast(y_true, dtype="float32")

        # Compute weighted distance: distance * label
        weighted_distances = distances * y_true_float

        # Sum weighted distances and count positive items
        sum_weighted_distances = ops.sum(weighted_distances, axis=-1, keepdims=True)
        num_positive = ops.sum(y_true_float, axis=-1, keepdims=True)

        # Avoid division by zero with epsilon
        epsilon = 1e-8
        penalty = sum_weighted_distances / (num_positive + epsilon)

        # Return mean penalty across batch
        return ops.mean(penalty)

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration of the loss.

        Returns:
            dict: A dictionary containing the configuration of the loss.
        """
        base_config = super().get_config()
        base_config.update({"distance_weight": self.distance_weight})
        return base_config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GeospatialMarginLoss":
        """Creates a new instance of the loss from its config.

        Args:
            config: A dictionary containing the configuration of the loss.

        Returns:
            GeospatialMarginLoss: A new instance of the loss.
        """
        return cls(**config)
