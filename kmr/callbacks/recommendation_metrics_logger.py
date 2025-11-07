"""Recommendation metrics logger callback for tracking model performance.

This callback logs custom recommendation metrics (Accuracy@K, Precision@K, Recall@K)
during training and provides formatted output for monitoring model progress.
"""

from typing import Any

import keras
from loguru import logger


class RecommendationMetricsLogger(keras.callbacks.Callback):
    """Logs custom recommendation metrics during training.

    This callback tracks Accuracy@K, Precision@K, and Recall@K metrics
    and provides formatted logging at each epoch for better monitoring.

    Args:
        verbose: Verbosity level (0=silent, 1=progress, 2=one line per epoch).
        log_frequency: Log metrics every N epochs (default=1).
        name: Optional name for the logger.

    Example:
        ```python
        from kmr.callbacks import RecommendationMetricsLogger
        from kmr.models import TwoTowerModel
        from kmr.losses import ImprovedMarginRankingLoss
        from kmr.metrics import AccuracyAtK, PrecisionAtK

        model = TwoTowerModel(num_items=100)
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=ImprovedMarginRankingLoss(),
            metrics=[AccuracyAtK(k=5), PrecisionAtK(k=5)]
        )

        callback = RecommendationMetricsLogger(verbose=1)
        model.fit(
            x=train_data,
            y=train_labels,
            epochs=10,
            callbacks=[callback]
        )
        ```
    """

    def __init__(
        self,
        verbose: int = 1,
        log_frequency: int = 1,
        name: str = "RecommendationMetricsLogger",
        **kwargs: Any,
    ) -> None:
        """Initialize the logger callback."""
        super().__init__(**kwargs)
        self.verbose = verbose
        self.log_frequency = log_frequency
        self.name = name
        self.epoch_metrics: dict[str, list] = {}

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        """Log metrics at the end of each epoch.

        Args:
            epoch: Current epoch number (0-indexed).
            logs: Dictionary containing metric values.
        """
        if logs is None:
            logs = {}

        # Store metrics for this epoch
        for metric_name, metric_value in logs.items():
            if metric_name not in self.epoch_metrics:
                self.epoch_metrics[metric_name] = []
            self.epoch_metrics[metric_name].append(metric_value)

        # Log only at specified frequency
        if (epoch + 1) % self.log_frequency == 0:
            if self.verbose >= 1:
                self._log_epoch_metrics(epoch, logs)

    def _log_epoch_metrics(self, epoch: int, logs: dict[str, float]) -> None:
        """Format and log epoch metrics.

        Args:
            epoch: Current epoch number.
            logs: Dictionary containing metric values.
        """
        # Separate loss and recommendation metrics
        loss = logs.get("loss", 0.0)
        recommendation_metrics = {
            k: v
            for k, v in logs.items()
            if k not in ["loss"] and not k.startswith("val_")
        }

        # Build log message
        log_msg = f"Epoch {epoch + 1}: loss={loss:.4f}"

        # Add recommendation metrics
        if recommendation_metrics:
            metrics_str = ", ".join(
                f"{k}={v:.4f}" for k, v in sorted(recommendation_metrics.items())
            )
            log_msg += f" | {metrics_str}"

        logger.info(log_msg)

        # Validation metrics if present
        val_metrics = {k: v for k, v in logs.items() if k.startswith("val_")}
        if val_metrics:
            val_str = ", ".join(f"{k}={v:.4f}" for k, v in sorted(val_metrics.items()))
            logger.info(f"  Validation: {val_str}")

    def on_train_end(self, logs: dict[str, float] | None = None) -> None:
        """Log training summary at the end.

        Args:
            logs: Final metric values.
        """
        if self.verbose >= 1 and self.epoch_metrics:
            logger.info("✅ Training completed!")
            logger.info("Training metrics summary:")

            for metric_name, values in sorted(self.epoch_metrics.items()):
                if values:
                    logger.info(
                        f"  {metric_name}: "
                        f"initial={values[0]:.4f}, "
                        f"final={values[-1]:.4f}, "
                        f"best={max(values):.4f}",
                    )

    def get_config(self) -> dict[str, Any]:
        """Get callback configuration for serialization.

        Returns:
            Dictionary with callback configuration.
        """
        return {
            "verbose": self.verbose,
            "log_frequency": self.log_frequency,
            "name": self.name,
        }
