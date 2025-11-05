"""Mean Reciprocal Rank (MRR) metric for recommendation systems.

This module provides a custom Keras metric that calculates Mean Reciprocal Rank,
which measures the average reciprocal rank of the first positive item found
in the recommendations.

Example:
    ```python
    import keras
    from kmr.metrics import MeanReciprocalRank

    # Create and use the metric
    metric = MeanReciprocalRank()
    metric.update_state(y_true, y_pred)
    mrr = metric.result()
    ```
"""

from typing import Any

import keras
from keras import ops
from keras.metrics import Metric
from keras.saving import register_keras_serializable
from loguru import logger


@register_keras_serializable(package="kmr.metrics")
class MeanReciprocalRank(Metric):
    """A custom Keras metric that calculates Mean Reciprocal Rank (MRR) for recommendation systems.

    MRR measures the average reciprocal rank of the first positive item found
    in the recommendations. The reciprocal rank is 1/rank if a positive item is found,
    and 0 otherwise.

    Args:
        name: Name of the metric (default="mean_reciprocal_rank").

    Example:
        ```python
        import keras
        from kmr.metrics import MeanReciprocalRank

        # Create metric
        mrr = MeanReciprocalRank(name="mrr")

        # y_true: binary labels (batch_size, num_items), 1 = positive item
        # y_pred: top-K recommendation indices (batch_size, k)
        y_true = keras.ops.array([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]])  # Items 0 and 2 are positive
        y_pred = keras.ops.array([[1, 0, 3, 4, 5]])  # Top-5 recommendations, item 0 is at position 2 (1-indexed)

        mrr.update_state(y_true, y_pred)
        result = mrr.result()  # 1/2 = 0.5 (first positive at rank 2)
        ```
    """

    def __init__(self, name: str = "mean_reciprocal_rank", **kwargs: Any) -> None:
        """Initializes the MeanReciprocalRank metric.

        Args:
            name: Name of the metric.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(name=name, **kwargs)
        self.total_rr = self.add_weight(name="total_rr", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

        logger.debug(f"Initialized MeanReciprocalRank metric with name={name}")

    def update_state(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
    ) -> None:
        """Updates the metric state with new predictions.

        Args:
            y_true: Binary labels of shape (batch_size, num_items) where 1 = positive item.
            y_pred: Top-K recommendation indices of shape (batch_size, k).
        """
        batch_size_tensor = ops.shape(y_true)[0]

        # Get batch size as int if possible
        try:
            batch_size = int(batch_size_tensor)
        except (TypeError, ValueError):
            batch_size = 32

        # Compute reciprocal rank for each user in the batch
        rr_sum = ops.cast(0.0, dtype="float32")

        for batch_idx in range(batch_size):
            # Get user's positive items and top-K recommendations
            user_positives = ops.take(y_true, batch_idx, axis=0)  # (num_items,)
            user_top_k_indices = ops.take(y_pred, batch_idx, axis=0)  # (k,)

            # Find the rank of the first positive item (1-indexed)
            # Gather positive flags for top-K items
            positive_flags = ops.take(
                user_positives,
                user_top_k_indices,
                axis=0,
            )  # (k,)

            # Find first positive item (index in top-K list)
            # Use argmax to find first True (1) value
            first_positive_idx = ops.argmax(positive_flags)

            # Check if any positive was found
            has_positive = ops.maximum(ops.max(positive_flags), 0.0)

            # Compute reciprocal rank: 1/rank if positive found, else 0
            # First positive found at rank (first_positive_idx + 1) in 1-indexed
            rank = ops.cast(first_positive_idx + 1, dtype="float32")
            reciprocal_rank_when_found = 1.0 / (rank + 1e-8)

            # Use ops.where to handle case when no positive found
            reciprocal_rank = ops.where(
                has_positive > 0.5,
                reciprocal_rank_when_found,
                ops.cast(0.0, dtype="float32"),
            )

            rr_sum = rr_sum + reciprocal_rank

        # Update running totals
        self.total_rr.assign_add(rr_sum)
        self.count.assign_add(ops.cast(batch_size_tensor, dtype="float32"))

    def result(self) -> keras.KerasTensor:
        """Returns the current Mean Reciprocal Rank value.

        Returns:
            KerasTensor: The current MRR metric value.
        """
        return self.total_rr / (self.count + 1e-8)

    def reset_state(self) -> None:
        """Resets the metric state."""
        self.total_rr.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration of the metric.

        Returns:
            dict: A dictionary containing the configuration of the metric.
        """
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MeanReciprocalRank":
        """Creates a new instance of the metric from its config.

        Args:
            config: A dictionary containing the configuration of the metric.

        Returns:
            MeanReciprocalRank: A new instance of the metric.
        """
        return cls(**config)
