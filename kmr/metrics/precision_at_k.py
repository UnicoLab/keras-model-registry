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

        # For each user, count how many of top-K are positive
        # Precision = sum(positive_flags) / k for each user
        n_relevant_per_user = ops.sum(positive_flags, axis=1)  # (batch_size,)
        precision_per_user = n_relevant_per_user / (
            ops.cast(k, dtype="float32") + 1e-8
        )  # (batch_size,)

        # Sum precision across batch
        precision_sum = ops.sum(precision_per_user)  # scalar

        # Update running totals
        self.total_precision.assign_add(ops.cast(precision_sum, dtype="float32"))
        self.count.assign_add(ops.cast(batch_size, dtype="float32"))

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
