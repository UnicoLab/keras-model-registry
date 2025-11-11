"""Top-K recommendation selector layer for recommendation systems.

This layer selects the top K items with highest scores from a ranking score matrix,
returning both the indices and the scores of selected items.
"""

from typing import Any
from keras import ops
from keras import KerasTensor
from keras.saving import register_keras_serializable

from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class TopKRecommendationSelector(BaseLayer):
    """Selects top-K items with highest scores for recommendations.

    This layer selects the top K items/products with the highest recommendation
    scores for each sample in a batch. It dynamically adjusts K if fewer than K
    items are available, and returns both indices and scores.

    Args:
        k: Number of top items to select (default=10).
        name: Optional name for the layer.

    Input shape:
        Tensor of shape `(batch_size, num_items)` containing scores.

    Output shape:
        Tuple of:
        - indices: `(batch_size, min(k, num_items))` - Indices of top K items
        - scores: `(batch_size, min(k, num_items))` - Scores of top K items

    Example:
        ```python
        import keras
        from kmr.layers import TopKRecommendationSelector

        # Create sample scores for batch_size=32, num_items=100
        scores = keras.random.normal((32, 100))

        # Select top 10 items
        selector = TopKRecommendationSelector(k=10)
        indices, top_scores = selector(scores)

        print("Indices shape:", indices.shape)  # (32, 10)
        print("Scores shape:", top_scores.shape)  # (32, 10)
        ```
    """

    def __init__(self, k: int = 10, name: str | None = None, **kwargs: Any) -> None:
        """Initialize the TopKRecommendationSelector layer.

        Args:
            k: Number of top items to select.
            name: Name of the layer.
            **kwargs: Additional keyword arguments.
        """
        # Set private attributes first
        self._k = k

        # Validate parameters
        self._validate_params()

        # Set public attributes BEFORE calling parent's __init__
        self.k = self._k

        # Call parent's __init__
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if not isinstance(self._k, int) or self._k <= 0:
            raise ValueError(f"k must be a positive integer, got {self._k}")

    def call(self, scores: KerasTensor) -> tuple[KerasTensor, KerasTensor]:
        """Select top K items by score.

        Args:
            scores: Score tensor of shape (batch_size, num_items).

        Returns:
            Tuple of (indices, scores) both with shape (batch_size, min(k, num_items)).
        """
        # Get number of items
        num_items = ops.shape(scores)[-1]

        # Adjust k to not exceed number of items
        actual_k = ops.minimum(self.k, num_items)

        # Use top_k to get indices and scores
        top_scores, top_indices = ops.top_k(scores, k=actual_k)

        return top_indices, top_scores

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "k": self.k,
            },
        )
        return config
