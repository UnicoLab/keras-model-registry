"""Precision@K metric for recommendation systems.

This module provides a custom Keras metric that calculates Precision@K,
which measures the fraction of top-K recommendations that are positive items.

Example:
    ```python
    import keras
    from kmr.metrics import PrecisionAtK

    # Create and use the metric
    metric = PrecisionAtK(k=10)
    metric.update_state(y_true, y_pred)
    prec_at_k = metric.result()
    ```
"""

from typing import Any

import keras
from keras import ops
from keras.metrics import Metric
from keras.saving import register_keras_serializable
from loguru import logger


@register_keras_serializable(package="kmr.metrics")
class PrecisionAtK(Metric):
    """A custom Keras metric that calculates Precision@K for recommendation systems.

    Precision@K measures the fraction of top-K recommendations that are positive items.
    This is a common metric for recommendation systems and collaborative filtering.

    Args:
        k: Number of top recommendations to consider (default=10).
        name: Name of the metric (default="precision_at_k").

    Example:
        ```python
        import keras
        from kmr.metrics import PrecisionAtK

        # Create metric
        prec_at_5 = PrecisionAtK(k=5, name="prec@5")

        # y_true: binary labels (batch_size, num_items), 1 = positive item
        # y_pred: top-K recommendation indices (batch_size, k)
        y_true = keras.ops.array([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]])  # Items 0 and 2 are positive
        y_pred = keras.ops.array([[0, 1, 3, 2, 4]])  # Top-5 recommendations

        prec_at_5.update_state(y_true, y_pred)
        result = prec_at_5.result()  # 0.4 (2 out of 5 are positive: items 0 and 2)
        ```
    """

    def __init__(
        self,
        k: int = 10,
        name: str = "precision_at_k",
        **kwargs: Any,
    ) -> None:
        """Initializes the PrecisionAtK metric.

        Args:
            k: Number of top recommendations to consider.
            name: Name of the metric.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(name=name, **kwargs)
        self.k = k
        self.total_precision = self.add_weight(
            name="total_precision",
            initializer="zeros",
        )
        self.count = self.add_weight(name="count", initializer="zeros")

        logger.debug(f"Initialized PrecisionAtK metric with k={k}, name={name}")

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
        y_true_shape = ops.shape(y_true)
        y_pred_shape = ops.shape(y_pred)
        batch_size_tensor = y_true_shape[0]
        batch_size_pred = y_pred_shape[0]
        k_actual = ops.shape(y_pred)[1]

        # Get batch size as int for Python loop
        try:
            batch_size_true = int(batch_size_tensor)
        except (TypeError, ValueError):
            if hasattr(batch_size_tensor, "numpy"):
                batch_size_true = int(batch_size_tensor.numpy())
            else:
                batch_size_true = 32

        try:
            batch_size_pred_int = int(batch_size_pred)
        except (TypeError, ValueError):
            if hasattr(batch_size_pred, "numpy"):
                batch_size_pred_int = int(batch_size_pred.numpy())
            else:
                batch_size_pred_int = batch_size_true

        # Get actual batch size at runtime - this is the source of truth
        actual_batch_size = ops.shape(y_true)[0]
        # Use computed batch_size as fallback
        fallback_batch_size = min(batch_size_true, batch_size_pred_int)
        try:
            actual_batch_size_int = int(actual_batch_size)
            batch_size = actual_batch_size_int
        except (TypeError, ValueError):
            # If we can't get concrete size, use fallback but cap it
            batch_size = min(fallback_batch_size, 32)
            return

        # Compute precision@K for each user in the batch
        precision_sum = ops.cast(0.0, dtype="float32")

        for batch_idx in range(batch_size):
            batch_idx = min(batch_idx, batch_size - 1)
            batch_idx_tensor = ops.cast(batch_idx, dtype="int32")

            # Get user's positive items and top-K recommendations
            user_positives = ops.take(y_true, batch_idx_tensor, axis=0)  # (num_items,)
            user_top_k_indices = ops.take(y_pred, batch_idx_tensor, axis=0)  # (k,)

            # Clamp indices to valid range to prevent out-of-bounds errors
            # This handles edge cases where y_true might have unexpected shape
            num_items_actual = ops.shape(user_positives)[0]
            user_top_k_indices_clamped = ops.clip(
                user_top_k_indices,
                0,
                num_items_actual - 1,
            )

            # Gather positive flags for top-K items
            positive_flags = ops.take(
                user_positives,
                user_top_k_indices_clamped,
                axis=0,
            )  # (k,)

            # Count how many of top-K are positive
            n_relevant = ops.sum(positive_flags)
            precision = n_relevant / (ops.cast(k_actual, dtype="float32") + 1e-8)
            precision_sum = precision_sum + precision

        # Update running totals
        self.total_precision.assign_add(precision_sum)
        self.count.assign_add(ops.cast(batch_size_tensor, dtype="float32"))

    def result(self) -> keras.KerasTensor:
        """Returns the current Precision@K value.

        Returns:
            KerasTensor: The current Precision@K metric value.
        """
        return self.total_precision / (self.count + 1e-8)

    def reset_state(self) -> None:
        """Resets the metric state."""
        self.total_precision.assign(0.0)
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
    def from_config(cls, config: dict[str, Any]) -> "PrecisionAtK":
        """Creates a new instance of the metric from its config.

        Args:
            config: A dictionary containing the configuration of the metric.

        Returns:
            PrecisionAtK: A new instance of the metric.
        """
        return cls(**config)
