"""NDCG@K (Normalized Discounted Cumulative Gain) metric for recommendation systems.

This module provides a custom Keras metric that calculates NDCG@K,
which measures ranking quality with position-based discounting.

Example:
    ```python
    import keras
    from kmr.metrics import NDCGAtK

    # Create and use the metric
    metric = NDCGAtK(k=10)
    metric.update_state(y_true, y_pred)
    ndcg = metric.result()
    ```
"""

from typing import Any

import keras
from keras import ops
from keras.metrics import Metric
from keras.saving import register_keras_serializable
from loguru import logger


@register_keras_serializable(package="kmr.metrics")
class NDCGAtK(Metric):
    """A custom Keras metric that calculates NDCG@K for recommendation systems.

    NDCG@K (Normalized Discounted Cumulative Gain) measures ranking quality
    with position-based discounting. Higher positions contribute more to the score,
    and the score is normalized by the ideal DCG (IDCG).

    Args:
        k: Number of top recommendations to consider (default=10).
        name: Name of the metric (default="ndcg_at_k").

    Example:
        ```python
        import keras
        from kmr.metrics import NDCGAtK

        # Create metric
        ndcg_at_5 = NDCGAtK(k=5, name="ndcg@5")

        # y_true: binary labels (batch_size, num_items), 1 = positive item
        # y_pred: top-K recommendation indices (batch_size, k)
        y_true = keras.ops.array([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]])  # Items 0 and 2 are positive
        y_pred = keras.ops.array([[0, 1, 3, 2, 4]])  # Top-5 recommendations

        ndcg_at_5.update_state(y_true, y_pred)
        result = ndcg_at_5.result()  # NDCG@5 score
        ```
    """

    def __init__(self, k: int = 10, name: str = "ndcg_at_k", **kwargs: Any) -> None:
        """Initializes the NDCGAtK metric.

        Args:
            k: Number of top recommendations to consider.
            name: Name of the metric.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(name=name, **kwargs)
        self.k = k
        self.total_ndcg = self.add_weight(name="total_ndcg", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

        logger.debug(f"Initialized NDCGAtK metric with k={k}, name={name}")

    def _compute_dcg(self, relevance_scores: keras.KerasTensor) -> keras.KerasTensor:
        """Compute Discounted Cumulative Gain.

        Args:
            relevance_scores: Relevance scores for top-K items, shape (k,).

        Returns:
            DCG value.
        """
        k = ops.shape(relevance_scores)[0]
        positions = ops.arange(1, k + 1, dtype="float32")  # 1-indexed positions
        log_positions = ops.log(positions + 1.0) / ops.log(2.0)  # log2(i+1)
        dcg = ops.sum(relevance_scores / log_positions)
        return dcg

    def _compute_idcg(self, n_relevant: keras.KerasTensor, k: int) -> keras.KerasTensor:
        """Compute Ideal Discounted Cumulative Gain.

        Args:
            n_relevant: Number of relevant items.
            k: Number of top items to consider.

        Returns:
            IDCG value.
        """
        # IDCG is DCG of ideal ranking (all relevant items at top)
        n_to_consider = ops.minimum(ops.cast(n_relevant, dtype="int32"), k)
        n_to_consider = ops.maximum(n_to_consider, 1)  # At least 1

        # Create ideal relevance vector: [1, 1, ..., 0, 0, ...]
        ideal_scores = ops.ones((n_to_consider,), dtype="float32")

        # Compute DCG for ideal ranking
        positions = ops.arange(1, n_to_consider + 1, dtype="float32")
        log_positions = ops.log(positions + 1.0) / ops.log(2.0)
        idcg = ops.sum(ideal_scores / log_positions)

        return idcg

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

        # Compute NDCG for each user in the batch
        ndcg_sum = ops.cast(0.0, dtype="float32")

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

            # Gather relevance scores for top-K items
            relevance_scores = ops.take(
                user_positives,
                user_top_k_indices_clamped,
                axis=0,
            )  # (k,)

            # Compute DCG
            dcg = self._compute_dcg(relevance_scores)

            # Compute IDCG (ideal DCG)
            n_relevant = ops.sum(user_positives)
            idcg = self._compute_idcg(n_relevant, k_actual)

            # Compute NDCG: dcg / idcg if idcg > 0, else 0
            ndcg = ops.where(
                idcg > 0,
                dcg / (idcg + 1e-8),
                ops.cast(0.0, dtype="float32"),
            )

            ndcg_sum = ndcg_sum + ndcg

        # Update running totals
        self.total_ndcg.assign_add(ndcg_sum)
        self.count.assign_add(ops.cast(batch_size_tensor, dtype="float32"))

    def result(self) -> keras.KerasTensor:
        """Returns the current NDCG@K value.

        Returns:
            KerasTensor: The current NDCG@K metric value.
        """
        return self.total_ndcg / (self.count + 1e-8)

    def reset_state(self) -> None:
        """Resets the metric state."""
        self.total_ndcg.assign(0.0)
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
    def from_config(cls, config: dict[str, Any]) -> "NDCGAtK":
        """Creates a new instance of the metric from its config.

        Args:
            config: A dictionary containing the configuration of the metric.

        Returns:
            NDCGAtK: A new instance of the metric.
        """
        return cls(**config)
