"""Accuracy@K metric for recommendation systems.

This module provides a custom Keras metric that calculates Accuracy@K,
which measures the percentage of users where at least one positive item
is in the top-K recommendations.

Example:
    ```python
    import keras
    from kmr.metrics import AccuracyAtK

    # Create and use the metric
    metric = AccuracyAtK(k=5)
    metric.update_state(y_true, y_pred)
    acc_at_k = metric.result()
    ```
"""

from typing import Any

import keras
from keras import ops
from keras.metrics import Metric
from keras.saving import register_keras_serializable
from loguru import logger


@register_keras_serializable(package="kmr.metrics")
class AccuracyAtK(Metric):
    """A custom Keras metric that calculates Accuracy@K for recommendation systems.

    Accuracy@K measures the percentage of users where at least one positive item
    is in the top-K recommendations. This is a common metric for recommendation
    systems and collaborative filtering.

    Args:
        k: Number of top recommendations to consider (default=10).
        name: Name of the metric (default="accuracy_at_k").

    Example:
        ```python
        import keras
        from kmr.metrics import AccuracyAtK

        # Create metric
        acc_at_5 = AccuracyAtK(k=5, name="acc@5")

        # y_true: binary labels (batch_size, num_items), 1 = positive item
        # y_pred: top-K recommendation indices (batch_size, k)
        y_true = keras.ops.array([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]])  # Items 0 and 2 are positive
        y_pred = keras.ops.array([[0, 1, 3, 4, 5]])  # Top-5 recommendations

        acc_at_5.update_state(y_true, y_pred)
        result = acc_at_5.result()  # 1.0 (item 0 is in top-5)
        ```
    """

    def __init__(self, k: int = 10, name: str = "accuracy_at_k", **kwargs: Any) -> None:
        """Initializes the AccuracyAtK metric.

        Args:
            k: Number of top recommendations to consider.
            name: Name of the metric.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(name=name, **kwargs)
        self.k = k
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

        logger.debug(f"Initialized AccuracyAtK metric with k={k}, name={name}")

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
        # We need to gather from y_true using indices from y_pred
        batch_indices = ops.arange(0, batch_size, dtype="int32")  # (batch_size,)
        batch_indices = ops.expand_dims(batch_indices, axis=1)  # (batch_size, 1)
        batch_indices = ops.tile(batch_indices, [1, k])  # (batch_size, k)

        # Gather positive flags for all users' top-K items
        # We need to use advanced indexing: for each (batch_idx, item_idx) pair,
        # get y_true[batch_idx, item_idx]
        # Since ops.gather doesn't support 2D indexing directly, we'll flatten and reshape

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

        # For each user, check if any item in top-K is positive
        # has_hit = 1 if max(positive_flags for that user) > 0, else 0
        max_per_user = ops.max(positive_flags, axis=1)  # (batch_size,)
        has_hit = ops.maximum(max_per_user, 0.0)  # (batch_size,)

        # Sum hits across batch
        hits_sum = ops.sum(has_hit)  # scalar

        # Update running totals
        self.total.assign_add(ops.cast(hits_sum, dtype="float32"))
        self.count.assign_add(ops.cast(batch_size, dtype="float32"))

    def result(self) -> keras.KerasTensor:
        """Returns the current Accuracy@K value.

        Returns:
            KerasTensor: The current Accuracy@K metric value.
        """
        return self.total / (self.count + 1e-8)

    def reset_state(self) -> None:
        """Resets the metric state."""
        self.total.assign(0.0)
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
    def from_config(cls, config: dict[str, Any]) -> "AccuracyAtK":
        """Creates a new instance of the metric from its config.

        Args:
            config: A dictionary containing the configuration of the metric.

        Returns:
            AccuracyAtK: A new instance of the metric.
        """
        return cls(**config)
