"""Standard deviation metric for Keras Model Registry.

This module provides a custom Keras metric that calculates the standard deviation
of predicted values, useful for anomaly detection and statistical analysis.

Example:
    ```python
    import keras
    from kmr.metrics import StandardDeviation

    # Create and use the metric
    metric = StandardDeviation()
    metric.update_state(predictions)
    std_value = metric.result()
    ```
"""

from typing import Any

import keras
from keras import ops
from keras.metrics import Metric
from keras.saving import register_keras_serializable
from loguru import logger


@register_keras_serializable(package="kmr.metrics")
class StandardDeviation(Metric):
    """A custom Keras metric that calculates the standard deviation of the predicted values.

    This class is a custom implementation of a Keras metric,
    which calculates the standard deviation of the predicted values during model training.
    It's particularly useful for anomaly detection tasks where you need to track
    the variability of model predictions.

    Attributes:
        values (keras.Variable): A trainable weight that stores the calculated standard deviation.

    Example:
        ```python
        import keras
        from kmr.metrics import StandardDeviation

        # Create metric
        std_metric = StandardDeviation(name="prediction_std")

        # Update with predictions
        predictions = keras.ops.random.normal((100, 10))
        std_metric.update_state(predictions)

        # Get result
        std_value = std_metric.result()
        print(f"Standard deviation: {std_value}")
        ```
    """

    def __init__(self, name: str = "standard_deviation", **kwargs: Any) -> None:
        """Initializes the StandardDeviation metric with a given name.

        Args:
            name (str, optional): The name of the metric. Defaults to 'standard_deviation'.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(name=name, **kwargs)
        self.values = self.add_weight(name="values", initializer="zeros")

        logger.debug(f"Initialized StandardDeviation metric with name: {name}")

    def update_state(self, y_pred: keras.KerasTensor) -> None:
        """Updates the state of the metric with the standard deviation of the predicted values.

        Args:
            y_pred (KerasTensor): The predicted values.
        """
        self.values.assign(ops.cast(ops.std(y_pred), dtype="float32"))

    def result(self) -> keras.KerasTensor:
        """Returns the current state of the metric, i.e., the current standard deviation.

        Returns:
            KerasTensor: The current standard deviation.
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
    def from_config(cls, config: dict[str, Any]) -> "StandardDeviation":
        """Creates a new instance of the metric from its config.

        Args:
            config (dict): A dictionary containing the configuration of the metric.

        Returns:
            StandardDeviation: A new instance of the metric.
        """
        return cls(**config)
