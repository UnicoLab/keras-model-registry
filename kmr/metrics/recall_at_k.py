"""Recall@K metric for recommendation systems.

This module provides a custom Keras metric that calculates Recall@K,
which measures the fraction of positive items that are in the top-K recommendations.

Example:
    ```python
    import keras
    from kmr.metrics import RecallAtK

    # Create and use the metric
    metric = RecallAtK(k=10)
    metric.update_state(y_true, y_pred)
    recall_at_k = metric.result()
    ```
"""

from typing import Any

import keras
from keras import ops
from keras.metrics import Metric
from keras.saving import register_keras_serializable
from loguru import logger


@register_keras_serializable(package="kmr.metrics")
class RecallAtK(Metric):
    """A custom Keras metric that calculates Recall@K for recommendation systems.

    Recall@K measures the fraction of positive items that are in the top-K recommendations.
    This is a common metric for recommendation systems and collaborative filtering.

    Args:
        k: Number of top recommendations to consider (default=10).
        name: Name of the metric (default="recall_at_k").

    Example:
        ```python
        import keras
        from kmr.metrics import RecallAtK

        # Create metric
        recall_at_5 = RecallAtK(k=5, name="recall@5")

        # y_true: binary labels (batch_size, num_items), 1 = positive item
        # y_pred: top-K recommendation indices (batch_size, k)
        y_true = keras.ops.array([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]])  # Items 0 and 2 are positive
        y_pred = keras.ops.array([[0, 1, 3, 2, 4]])  # Top-5 recommendations

        recall_at_5.update_state(y_true, y_pred)
        result = recall_at_5.result()  # 1.0 (both positive items 0 and 2 are in top-5)
        ```
    """

    def __init__(self, k: int = 10, name: str = "recall_at_k", **kwargs: Any) -> None:
        """Initializes the RecallAtK metric.

        Args:
            k: Number of top recommendations to consider.
            name: Name of the metric.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(name=name, **kwargs)
        self.k = k
        self.total_recall = self.add_weight(name="total_recall", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

        logger.debug(f"Initialized RecallAtK metric with k={k}, name={name}")

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

        # Compute recall@K for each user in the batch
        recall_sum = ops.cast(0.0, dtype="float32")

        for batch_idx in range(batch_size):
            # Get user's positive items and top-K recommendations
            user_positives = ops.take(y_true, batch_idx, axis=0)  # (num_items,)
            user_top_k_indices = ops.take(y_pred, batch_idx, axis=0)  # (k,)

            # Count total positive items for this user
            n_total_positive = ops.sum(user_positives)

            # Count how many positive items are in top-K
            positive_flags = ops.take(
                user_positives,
                user_top_k_indices,
                axis=0,
            )  # (k,)
            n_relevant_in_top_k = ops.sum(positive_flags)

            # Compute recall: n_relevant_in_top_k / n_total_positive
            # Use ops.where to handle case when n_total_positive = 0
            recall = ops.where(
                n_total_positive > 0,
                n_relevant_in_top_k / (n_total_positive + 1e-8),
                ops.cast(0.0, dtype="float32"),
            )

            recall_sum = recall_sum + recall

        # Update running totals
        self.total_recall.assign_add(recall_sum)
        self.count.assign_add(ops.cast(batch_size_tensor, dtype="float32"))

    def result(self) -> keras.KerasTensor:
        """Returns the current Recall@K value.

        Returns:
            KerasTensor: The current Recall@K metric value.
        """
        return self.total_recall / (self.count + 1e-8)

    def reset_state(self) -> None:
        """Resets the metric state."""
        self.total_recall.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration of the metric.

        Returns:
            dict: A dictionary containing the configuration of the metric.
        """
        base_config = super().get_config()
        base_config.update({"k": self.k})
        return base_config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RecallAtK":
        """Creates a new instance of the metric from its config.

        Args:
            config: A dictionary containing the configuration of the metric.

        Returns:
            RecallAtK: A new instance of the metric.
        """
        return cls(**config)
