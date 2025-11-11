"""Max-Min Margin Loss for recommendation systems.

This module implements a margin ranking loss that maximizes the margin between
the best positive item and the worst negative item for each user.
"""

from typing import Any

import keras
from keras import ops
from keras.losses import Loss
from keras.saving import register_keras_serializable
from loguru import logger


@register_keras_serializable(package="kmr.losses")
class MaxMinMarginLoss(Loss):
    """Max-Min Margin Loss for recommendation systems.

    This loss encourages the model to rank positive items higher than negative items
    by maximizing the margin between the best positive score and the worst negative score.

    Args:
        margin: The margin threshold (default=1.0).
        name: Name of the loss (default="max_min_margin_loss").

    Example:
        ```python
        import keras
        from kmr.losses import MaxMinMarginLoss

        loss = MaxMinMarginLoss(margin=1.0)

        # y_true: binary labels (batch_size, num_items), 1 = positive item
        # y_pred: similarity scores (batch_size, num_items)
        y_true = keras.ops.array([[1, 0, 1, 0, 0]])  # Items 0 and 2 are positive
        y_pred = keras.ops.array([[0.8, 0.2, 0.7, 0.1, 0.0]])  # Scores for each item

        loss_value = loss(y_true, y_pred)
        ```
    """

    def __init__(
        self,
        margin: float = 1.0,
        name: str = "max_min_margin_loss",
        **kwargs: Any,
    ) -> None:
        """Initialize MaxMinMarginLoss.

        Args:
            margin: The margin threshold for ranking.
            name: Name of the loss.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(name=name, **kwargs)
        self.margin = margin
        logger.debug(f"Initialized MaxMinMarginLoss with margin={margin}, name={name}")

    def call(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Compute max-min margin loss.

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

        # Get max positive score for each user
        max_positive = ops.max(
            ops.where(
                positive_mask,
                y_pred_float,
                ops.full_like(
                    y_pred_float,
                    -1e9,
                ),  # Very negative for non-positive items
            ),
            axis=-1,
            keepdims=True,
        )  # (batch_size, 1)

        # Get min negative score for each user
        min_negative = ops.min(
            ops.where(
                negative_mask,
                y_pred_float,
                ops.full_like(
                    y_pred_float,
                    1e9,
                ),  # Very positive for non-negative items
            ),
            axis=-1,
            keepdims=True,
        )  # (batch_size, 1)

        # Compute margin loss: max(0, margin - (max_pos - min_neg))
        # When max_pos > min_neg + margin, loss is 0 (desired state)
        # Otherwise, loss is positive
        margin_loss = ops.maximum(0.0, self.margin - (max_positive - min_negative))

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
    def from_config(cls, config: dict[str, Any]) -> "MaxMinMarginLoss":
        """Creates a new instance of the loss from its config.

        Args:
            config: A dictionary containing the configuration of the loss.

        Returns:
            MaxMinMarginLoss: A new instance of the loss.
        """
        return cls(**config)
