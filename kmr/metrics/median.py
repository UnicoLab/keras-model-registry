"""Median metric for Keras Model Registry.

This module provides a custom Keras metric that calculates the median
of predicted values, useful for anomaly detection and robust statistical analysis.

Example:
    ```python
    import keras
    from kmr.metrics import Median

    # Create and use the metric
    metric = Median()
    metric.update_state(predictions)
    median_value = metric.result()
    ```
"""

from typing import Any

import keras
from keras import ops
from keras.metrics import Metric
from keras.saving import register_keras_serializable
from loguru import logger


@register_keras_serializable(package="kmr.metrics")
class Median(Metric):
    """A custom Keras metric that calculates the median of the predicted values.

    This class is a custom implementation of a Keras metric,
    which calculates the median of the predicted values during model training.
    The median is a robust measure of central tendency that is less sensitive
    to outliers compared to the mean, making it particularly useful for
    anomaly detection tasks.

    Attributes:
        values (keras.Variable): A trainable weight that stores the calculated median.

    Example:
        ```python
        import keras
        from kmr.metrics import Median

        # Create metric
        median_metric = Median(name="prediction_median")

        # Update with predictions
        predictions = keras.ops.random.normal((100, 10))
        median_metric.update_state(predictions)

        # Get result
        median_value = median_metric.result()
        print(f"Median: {median_value}")
        ```
    """

    def __init__(self, name: str = "median", **kwargs: Any) -> None:
        """Initializes the Median metric with a given name.

        Args:
            name (str, optional): The name of the metric. Defaults to 'median'.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(name=name, **kwargs)
        self.values = self.add_weight(name="values", initializer="zeros")

        logger.debug(f"Initialized Median metric with name: {name}")

    def update_state(self, y_pred: keras.KerasTensor) -> None:
        """Updates the state of the metric with the median of the predicted values.

        Args:
            y_pred (KerasTensor): The predicted values.
        """
        # Calculate median using Keras operations
        sorted_values = ops.sort(y_pred, axis=0)
        n = ops.shape(sorted_values)[0]
        mid = n // 2

        if n % 2 == 0:
            median = (sorted_values[mid - 1] + sorted_values[mid]) / 2
        else:
            median = sorted_values[mid]

        # Ensure median is a scalar
        median = ops.cast(median, dtype="float32")
        if median.shape != ():
            median = ops.mean(median)  # Take mean if it's not a scalar

        self.values.assign(median)

    def result(self) -> keras.KerasTensor:
        """Returns the current state of the metric, i.e., the current median.

        Returns:
            KerasTensor: The current median.
        """
        return self.values

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration of the metric.

        Returns:
            dict: A dictionary containing the configuration of the metric.
        """
        base_config = super().get_config()
        return {**base_config}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Median":
        """Creates a new instance of the metric from its config.

        Args:
            config (dict): A dictionary containing the configuration of the metric.

        Returns:
            Median: A new instance of the metric.
        """
        return cls(**config)
