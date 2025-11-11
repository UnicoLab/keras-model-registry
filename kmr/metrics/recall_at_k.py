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
        y_pred: keras.KerasTensor | dict,
        sample_weight=None,  # noqa: ARG002
    ) -> None:
        """Updates the metric state with new predictions using vectorized operations.

        Args:
            y_true: Binary labels of shape (batch_size, num_items) where 1 = positive item.
            y_pred: Can be:
                - Dictionary with 'rec_indices' key (from model.call() dict output)
                - Top-K recommendation indices of shape (batch_size, k)
                - Tuple of (similarities, indices, scores) from unified model output
                - Full similarity matrix (batch_size, num_items) - will extract top-K internally
            sample_weight: Not used, for compatibility with Keras metric interface.
        """
        # Smart input detection and conversion
        if isinstance(y_pred, dict):
            # Extract indices from dictionary
            y_pred = y_pred["rec_indices"]
        elif isinstance(y_pred, tuple | list):
            # Extract indices from tuple (similarities, indices, scores)
            y_pred = y_pred[1]
        else:
            # Check if it's a full similarity matrix instead of indices
            pred_shape = ops.shape(y_pred)
            if len(y_pred.shape) == 2 and pred_shape[1] > self.k:
                # Full similarity matrix - extract top-K indices
                y_pred = ops.argsort(y_pred, axis=1)[:, -self.k :]

        batch_size = ops.shape(y_true)[0]
        num_items = ops.cast(ops.shape(y_true)[1], dtype="int32")
        k = ops.shape(y_pred)[1]

        # Clamp indices to valid range [0, num_items-1]
        y_pred_int = ops.cast(y_pred, dtype="int32")
        y_pred_clamped = ops.clip(y_pred_int, 0, num_items - 1)

        # Create batch indices for gathering: (batch_size, k)
        batch_indices = ops.arange(0, batch_size, dtype="int32")  # (batch_size,)
        batch_indices = ops.expand_dims(batch_indices, axis=1)  # (batch_size, 1)
        batch_indices = ops.tile(batch_indices, [1, k])  # (batch_size, k)

        # Flatten y_true and create flat indices
        y_true_flat = ops.reshape(y_true, [-1])  # (batch_size * num_items,)

        # Create flat indices: batch_idx * num_items + item_idx for each (batch_idx, item_idx)
        flat_indices = batch_indices * num_items + y_pred_clamped  # (batch_size, k)
        flat_indices = ops.reshape(flat_indices, [-1])  # (batch_size * k,)

        # Gather positive flags
        positive_flags = ops.take(
            y_true_flat,
            flat_indices,
            axis=0,
        )  # (batch_size * k,)
        positive_flags = ops.reshape(positive_flags, [batch_size, k])  # (batch_size, k)

        # Count total positive items per user
        n_total_positive_per_user = ops.sum(y_true, axis=1)  # (batch_size,)

        # Count how many positive items are in top-K for each user
        n_relevant_in_top_k_per_user = ops.sum(positive_flags, axis=1)  # (batch_size,)

        # Compute recall per user: n_relevant_in_top_k / n_total_positive
        # Handle case when n_total_positive = 0 (use 0.0 for recall)
        recall_per_user = ops.where(
            n_total_positive_per_user > 0,
            n_relevant_in_top_k_per_user / (n_total_positive_per_user + 1e-8),
            ops.cast(0.0, dtype="float32"),
        )  # (batch_size,)

        # Sum recall across batch
        recall_sum = ops.sum(recall_per_user)  # scalar

        # Update running totals
        self.total_recall.assign_add(ops.cast(recall_sum, dtype="float32"))
        self.count.assign_add(ops.cast(batch_size, dtype="float32"))

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
