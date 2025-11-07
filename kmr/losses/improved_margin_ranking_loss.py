"""Improved Margin Ranking Loss for recommendation systems.

This module implements a combined margin ranking loss that uses both max-min and average
margin losses with configurable weights for balanced learning.
"""

from typing import Any

import keras
from keras.losses import Loss
from keras.saving import register_keras_serializable
from loguru import logger

from kmr.losses.max_min_margin_loss import MaxMinMarginLoss
from kmr.losses.average_margin_loss import AverageMarginLoss


@register_keras_serializable(package="kmr.losses")
class ImprovedMarginRankingLoss(Loss):
    """Improved Margin Ranking Loss for recommendation systems.

    This loss combines MaxMinMarginLoss and AverageMarginLoss with configurable weights
    to provide both a strong signal for top-K ranking (max-min) and stability (average).

    The combined loss is: max_min_weight * max_min_loss + avg_weight * avg_loss

    Args:
        margin: The margin threshold (default=1.0).
        max_min_weight: Weight for max-min margin loss (default=0.7).
        avg_weight: Weight for average margin loss (default=0.3).
        name: Name of the loss (default="improved_margin_ranking_loss").

    Example:
        ```python
        import keras
        from kmr.losses import ImprovedMarginRankingLoss

        loss = ImprovedMarginRankingLoss(margin=1.0, max_min_weight=0.7, avg_weight=0.3)

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
        max_min_weight: float = 0.7,
        avg_weight: float = 0.3,
        name: str = "improved_margin_ranking_loss",
        **kwargs: Any,
    ) -> None:
        """Initialize ImprovedMarginRankingLoss.

        Args:
            margin: The margin threshold for ranking.
            max_min_weight: Weight for max-min margin loss component.
            avg_weight: Weight for average margin loss component.
            name: Name of the loss.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(name=name, **kwargs)
        self.margin = margin
        self.max_min_weight = max_min_weight
        self.avg_weight = avg_weight

        # Initialize component losses
        self.max_min_loss = MaxMinMarginLoss(margin=margin, name="max_min_margin")
        self.avg_loss = AverageMarginLoss(margin=margin, name="avg_margin")

        logger.debug(
            f"Initialized ImprovedMarginRankingLoss with margin={margin}, "
            f"max_min_weight={max_min_weight}, avg_weight={avg_weight}, name={name}",
        )

    def call(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor | dict,
    ) -> keras.KerasTensor:
        """Compute improved margin ranking loss.

        Args:
            y_true: Binary labels of shape (batch_size, num_items) where 1 = positive item.
            y_pred: Can be:
                - Dictionary with 'similarities' key (from model.call() dict output)
                - Dictionary with 'scores' key (from models like DeepRankingModel)
                - Dictionary with any single scores key
                - Similarity scores of shape (batch_size, num_items)
                - Tuple/list of (similarities, indices, scores) from unified model output

        Returns:
            Scalar loss value combining both margin losses.
        """
        # Extract similarities from dictionary if model returns dict output
        if isinstance(y_pred, dict):
            # Try to find the scores/similarities key (handles different model types)
            if "similarities" in y_pred:
                similarities = y_pred["similarities"]
            elif "scores" in y_pred:
                similarities = y_pred["scores"]
            elif "combined_scores" in y_pred:
                similarities = y_pred["combined_scores"]
            elif "masked_scores" in y_pred:
                similarities = y_pred["masked_scores"]
            else:
                # Fall back to first value if no known key found
                similarities = next(iter(y_pred.values()))
        # Extract similarities from tuple if model returns unified output
        elif isinstance(y_pred, (tuple, list)):
            similarities = y_pred[0]  # Extract similarities (batch_size, num_items)
        else:
            similarities = y_pred  # Backward compatibility with raw similarities

        # Compute component losses
        max_min_loss_value = self.max_min_loss(y_true, similarities)
        avg_loss_value = self.avg_loss(y_true, similarities)

        # Combine with weights
        combined_loss = (
            self.max_min_weight * max_min_loss_value + self.avg_weight * avg_loss_value
        )

        return combined_loss

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration of the loss.

        Returns:
            dict: A dictionary containing the configuration of the loss.
        """
        base_config = super().get_config()
        base_config.update(
            {
                "margin": self.margin,
                "max_min_weight": self.max_min_weight,
                "avg_weight": self.avg_weight,
            },
        )
        return base_config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ImprovedMarginRankingLoss":
        """Creates a new instance of the loss from its config.

        Args:
            config: A dictionary containing the configuration of the loss.

        Returns:
            ImprovedMarginRankingLoss: A new instance of the loss.
        """
        return cls(**config)
