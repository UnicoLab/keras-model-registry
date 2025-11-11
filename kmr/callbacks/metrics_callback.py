"""Custom callback for computing recommendation metrics during training."""

from typing import Optional

import keras
from loguru import logger


class RecommendationMetricsCallback(keras.callbacks.Callback):
    """Callback that computes custom recommendation metrics after each epoch.

    This callback solves the issue of Keras 3 not properly supporting
    dictionary-mapped metrics with multi-output models by computing metrics
    manually and logging them to the training history.

    Args:
        metrics: List of metric instances (e.g., [AccuracyAtK(k=5), PrecisionAtK(k=5)])
        metric_names: Optional list of metric names (defaults to metric.name)
        validation_data: Optional tuple (x, y) for validation metrics
    """

    def __init__(
        self,
        metrics: list[keras.metrics.Metric],
        metric_names: Optional[list[str]] = None,
        validation_data: Optional[tuple] = None,
    ):
        """Initialize the callback.

        Args:
            metrics: List of metric instances to compute
            metric_names: Optional custom names for metrics
            validation_data: Optional (x, y) tuple for validation metrics
        """
        super().__init__()
        self.metrics_to_compute = metrics
        self.metric_names = metric_names or [m.name for m in metrics]
        self.validation_data = validation_data

        logger.debug(
            f"Initialized RecommendationMetricsCallback with metrics: {self.metric_names}",  # noqa: E501
        )

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Compute metrics at the end of each epoch.

        Args:
            epoch: The epoch number
            logs: Dictionary of logs from the epoch
        """
        if logs is None:
            logs = {}

        # Compute metrics on training data if validation_data not provided
        if self.validation_data is not None:
            x, y = self.validation_data
            y_pred = self.model.predict(x, verbose=0)

            # Reset metrics before computing
            for metric in self.metrics_to_compute:
                metric.reset_state()

            # Compute metrics
            for metric, name in zip(
                self.metrics_to_compute, self.metric_names, strict=True
            ):
                metric.update_state(y, y_pred)
                metric_value = metric.result().numpy()
                logs[f"val_{name}"] = metric_value
                logger.debug(f"Epoch {epoch+1}: val_{name} = {metric_value:.4f}")

        # Log to console if requested
        if self.metrics_to_compute:
            metric_str = " - ".join(
                [
                    f"{name}: {logs.get(f'val_{name}', 0.0):.4f}"
                    for name in self.metric_names
                ],
            )
            if metric_str and epoch % 5 == 0:
                logger.info(f"Epoch {epoch+1} Metrics: {metric_str}")


class MetricsLogger(keras.callbacks.Callback):
    """Simpler callback that just logs metrics to console."""

    def __init__(self, log_interval: int = 1):
        """Initialize the logger.

        Args:
            log_interval: Log metrics every N epochs
        """
        super().__init__()
        self.log_interval = log_interval

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Log metrics at the end of each epoch.

        Args:
            epoch: The epoch number
            logs: Dictionary of logs from the epoch
        """
        if logs is None or epoch % self.log_interval != 0:
            return

        # Format metrics nicely
        metrics_parts = []
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                metrics_parts.append(f"{key}: {value:.4f}")

        if metrics_parts:
            logger.info(f"Epoch {epoch+1}: {' - '.join(metrics_parts)}")
