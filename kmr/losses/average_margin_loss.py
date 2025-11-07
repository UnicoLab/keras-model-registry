"""Average Margin Loss for recommendation systems.

This module implements a margin ranking loss that maximizes the margin between
the average positive item score and the average negative item score for each user.
"""

from typing import Any

import keras
from keras import ops
from keras.losses import Loss
from keras.saving import register_keras_serializable
from loguru import logger


@register_keras_serializable(package="kmr.losses")
class AverageMarginLoss(Loss):
    """Average Margin Loss for recommendation systems.

    This loss encourages the model to rank positive items higher than negative items
    by maximizing the margin between the average positive score and the average negative score.
    This provides stability compared to max-min margin which only looks at extremes.

    Args:
        margin: The margin threshold (default=0.5).
        name: Name of the loss (default="average_margin_loss").

    Example:
        ```python
        import keras
        from kmr.losses import AverageMarginLoss

        loss = AverageMarginLoss(margin=0.5)

        # y_true: binary labels (batch_size, num_items), 1 = positive item
        # y_pred: similarity scores (batch_size, num_items)
        y_true = keras.ops.array([[1, 0, 1, 0, 0]])  # Items 0 and 2 are positive
        y_pred = keras.ops.array([[0.8, 0.2, 0.7, 0.1, 0.0]])  # Scores for each item

        loss_value = loss(y_true, y_pred)
        ```
    """

    def __init__(
        self,
        margin: float = 0.5,
        name: str = "average_margin_loss",
        **kwargs: Any,
    ) -> None:
        """Initialize AverageMarginLoss.

        Args:
            margin: The margin threshold for ranking.
            name: Name of the loss.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(name=name, **kwargs)
        self.margin = margin
        logger.debug(f"Initialized AverageMarginLoss with margin={margin}, name={name}")

    def call(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Compute average margin loss.

        Args:
            y_true: Binary labels of shape (batch_size, num_items) where 1 = positive item.
            y_pred: Similarity scores of shape (batch_size, num_items).

        Returns:
            Scalar loss value.
        """
        # Convert to float for computation
        y_true_float = ops.cast(y_true, dtype="float32")
        y_pred_float = ops.cast(y_pred, dtype="float32")

        # Create masks for positive and negative items
        positive_mask = y_true_float > 0.5  # (batch_size, num_items)
        negative_mask = y_true_float < 0.5  # (batch_size, num_items)

        # Count positive and negative items per user
        n_positive = ops.sum(
            ops.cast(positive_mask, dtype="float32"),
            axis=-1,
            keepdims=True,
        )  # (batch_size, 1)
        n_negative = ops.sum(
            ops.cast(negative_mask, dtype="float32"),
            axis=-1,
            keepdims=True,
        )  # (batch_size, 1)

        # Compute average positive score
        positive_scores = ops.where(
            positive_mask,
            y_pred_float,
            ops.zeros_like(y_pred_float),
        )
        avg_positive = ops.sum(positive_scores, axis=-1, keepdims=True) / (
            n_positive + 1e-8
        )  # (batch_size, 1)

        # Compute average negative score
        negative_scores = ops.where(
            negative_mask,
            y_pred_float,
            ops.zeros_like(y_pred_float),
        )
        avg_negative = ops.sum(negative_scores, axis=-1, keepdims=True) / (
            n_negative + 1e-8
        )  # (batch_size, 1)

        # Compute margin loss: max(0, margin - (avg_pos - avg_neg))
        margin_loss = ops.maximum(0.0, self.margin - (avg_positive - avg_negative))

        # Average across batch
        return ops.mean(margin_loss)

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration of the loss.

        Returns:
            dict: A dictionary containing the configuration of the loss.
        """
        base_config = super().get_config()
        base_config.update({"margin": self.margin})
        return base_config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AverageMarginLoss":
        """Creates a new instance of the loss from its config.

        Args:
            config: A dictionary containing the configuration of the loss.

        Returns:
            AverageMarginLoss: A new instance of the loss.
        """
        return cls(**config)
