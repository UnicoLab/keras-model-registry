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
        y_pred: keras.KerasTensor,
    ) -> None:
        """Updates the metric state with new predictions.

        Args:
            y_true: Binary labels of shape (batch_size, num_items) where 1 = positive item.
            y_pred: Top-K recommendation indices of shape (batch_size, k).
        """
        batch_size_tensor = ops.shape(y_true)[0]
        # k_actual = ops.shape(y_pred)[1]

        # Compute accuracy@K using vectorized operations
        # For each user, check if any positive item is in top-K recommendations
        # We'll use ops.take to gather values per batch element

        # Get batch size as int if possible (for loop), otherwise use a workaround
        # During metric updates, batch_size is typically concrete
        try:
            # Try to get concrete batch size
            batch_size = int(batch_size_tensor)
        except (TypeError, ValueError):
            # If symbolic, use a reasonable default (metrics update during training has concrete sizes)
            # This is a fallback that shouldn't normally be hit
            batch_size = 32

        # Initialize accumulator
        hits_sum = ops.cast(0.0, dtype="float32")

        # For each user in batch, compute if they have a hit
        # Note: This loop runs during metric update where batch_size is typically small and concrete
        for batch_idx in range(batch_size):
            # Get user's positive items and top-K recommendations
            user_positives = ops.take(y_true, batch_idx, axis=0)  # (num_items,)
            user_top_k_indices = ops.take(y_pred, batch_idx, axis=0)  # (k,)

            # Gather positive flags for top-K items
            positive_flags = ops.take(
                user_positives,
                user_top_k_indices,
                axis=0,
            )  # (k,)

            # Check if any item in top-K is positive (has_hit = 1 if any is 1, else 0)
            has_hit = ops.maximum(ops.max(positive_flags), 0.0)
            hits_sum = hits_sum + has_hit

        # Update running totals
        self.total.assign_add(hits_sum)
        self.count.assign_add(ops.cast(batch_size_tensor, dtype="float32"))

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
