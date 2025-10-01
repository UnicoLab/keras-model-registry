# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

# project plugin
from config import Config
from loguru import logger

conf = Config()


class StandardDeviation(tf.keras.metrics.Metric):
    """A custom Keras metric that calculates the standard deviation of the predicted values.

    This class is a custom implementation of a Keras metric,
    which calculates the standard deviation of the predicted values during model training.

    Attributes:
        values (tf.Variable): A trainable weight that stores the calculated standard deviation.
    """

    def __init__(self, name="standard_deviation", **kwargs):
        """Initializes the StandardDeviation metric with a given name.

        Args:
            name (str, optional): The name of the metric. Defaults to 'standard_deviation'.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(name=name, **kwargs)
        self.values = self.add_weight(name="values", initializer="zeros")

    def update_state(self, y_pred) -> None:
        """Updates the state of the metric with the standard deviation of the predicted values.

        Args:
            y_pred (Tensor): The predicted values.
        """
        self.values.assign(tf.cast(tf.math.reduce_std(y_pred), dtype=tf.float32))

    def result(self) -> float:
        """Returns the current state of the metric, i.e., the current standard deviation.

        Returns:
            Tensor: The current standard deviation.
        """
        return self.values

    def get_config(self) -> dict:
        """Returns the configuration of the metric.

        Returns:
            dict: A dictionary containing the configuration of the metric.
        """
        base_config = super().get_config()
        return {**base_config}

    @classmethod
    def from_config(cls, config) -> object:
        """Creates a new instance of the metric from its config.

        Args:
            config (dict): A dictionary containing the configuration of the metric.

        Returns:
            StandardDeviation: A new instance of the metric.
        """
        return cls(**config)


class Median(tf.keras.metrics.Metric):
    """A custom Keras metric that calculates the median of the predicted values.

    This class is a custom implementation of a Keras metric,
    which calculates the median of the predicted values during model training.

    Attributes:
        values (tf.Variable): A trainable weight that stores the calculated median.
    """

    def __init__(self, name="median", **kwargs):
        """Initializes the Median metric with a given name.

        Args:
            name (str, optional): The name of the metric. Defaults to 'median'.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(name=name, **kwargs)
        self.values = self.add_weight(name="values", initializer="zeros")

    def update_state(self, y_pred) -> None:
        """Updates the state of the metric with the median of the predicted values.

        Args:
            y_pred (Tensor): The predicted values.
        """
        m = y_pred.get_shape()[0] // 2
        median = tf.reduce_min(tf.nn.top_k(y_pred, m, sorted=False).values)
        self.values.assign(tf.cast(median, dtype=tf.float32))

    def result(self) -> float:
        """Returns the current state of the metric, i.e., the current median.

        Returns:
            Tensor: The current median.
        """
        return self.values

    def get_config(self) -> dict:
        """Returns the configuration of the metric.

        Returns:
            dict: A dictionary containing the configuration of the metric.
        """
        base_config = super().get_config()
        return {**base_config}

    @classmethod
    def from_config(cls, config) -> object:
        """Creates a new instance of the metric from its config.

        Args:
            config (dict): A dictionary containing the configuration of the metric.

        Returns:
            Median: A new instance of the metric.
        """
        return cls(**config)


class Autoencoder(tf.keras.Model):
    """An autoencoder model for anomaly detection.

    This class implements an autoencoder neural network model used for anomaly detection.
    It includes methods for model compilation, fitting, threshold setup, and anomaly detection.

    Attributes:
        input_dim (int): The dimension of the input data.
        encoding_dim (int): The dimension of the encoded representation.
        intermediate_dim (int): The dimension of the intermediate layer.
        _threshold (tf.Variable): The threshold for anomaly detection.
        _median (tf.Variable): The median of the anomaly scores.
        _std (tf.Variable): The standard deviation of the anomaly scores.
    """

    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = 64,
        intermediate_dim: int = 32,
        threshold: float = 2,
        **kwargs,
    ):
        """Initializes the Autoencoder model.

        Args:
            input_dim (int): The dimension of the input data.
            encoding_dim (int, optional): The dimension of the encoded representation. Defaults to 64.
            intermediate_dim (int, optional): The dimension of the intermediate layer. Defaults to 32.
            threshold (float, optional): The initial threshold for anomaly detection. Defaults to 2.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.intermediate_dim = intermediate_dim
        self._threshold = tf.Variable(threshold, dtype=tf.float32)
        self._median = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self._std = tf.Variable(0.0, dtype=tf.float32, trainable=False)

        # Encoder layers
        self.encoder_dense1 = tf.keras.layers.Dense(intermediate_dim, activation="relu")
        self.encoder_dropout1 = tf.keras.layers.Dropout(0.1)
        self.encoder_dense2 = tf.keras.layers.Dense(encoding_dim, activation="relu")
        self.encoder_dropout2 = tf.keras.layers.Dropout(0.1)

        # Decoder layers
        self.decoder_dense1 = tf.keras.layers.Dense(intermediate_dim, activation="relu")
        self.decoder_dropout1 = tf.keras.layers.Dropout(0.1)
        self.decoder_dense2 = tf.keras.layers.Dense(input_dim, activation="sigmoid")
        self.decoder_dropout2 = tf.keras.layers.Dropout(0.1)

    def call(self, inputs) -> tf.Tensor:
        """Performs the forward pass of the autoencoder.

        Args:
            inputs (Tensor): The input data.

        Returns:
            Tensor: The reconstructed input data.
        """
        # Encoder
        x = self.encoder_dense1(inputs)
        x = self.encoder_dropout1(x)
        x = self.encoder_dense2(x)
        encoded = self.encoder_dropout2(x)

        # Decoder
        x = self.decoder_dense1(encoded)
        x = self.decoder_dropout1(x)
        x = self.decoder_dense2(x)
        decoded = self.decoder_dropout2(x)

        return decoded

    @property
    def threshold(self) -> np.ndarray:
        """Gets the current threshold value.

        Returns:
            float: The current threshold value.
        """
        return self._threshold.numpy()

    @property
    def median(self) -> np.ndarray:
        """Gets the current median value.

        Returns:
            float: The current median value.
        """
        return self._median.numpy()

    @property
    def std(self) -> np.ndarray:
        """Gets the current standard deviation value.

        Returns:
            float: The current standard deviation value.
        """
        return self._std.numpy()

    def compile_model(self, optimizer, loss, **kwargs) -> None:
        """Compiles the model with the given optimizer and loss function.

        Args:
            optimizer: The optimizer to use for training.
            loss: The loss function to use for training.
            **kwargs: Additional keyword arguments passed to the compile method.
        """
        super().compile(optimizer=optimizer, loss=loss, **kwargs)

    def fit(self, data, epochs: int, callbacks=None, **kwargs) -> tf.keras.callbacks.History:
        """Fits the model to the given data.

        Args:
            data: The training data.
            epochs (int): The number of epochs to train for.
            callbacks (optional): A list of callbacks to use during training.
            **kwargs: Additional keyword arguments passed to the fit method.

        Returns:
            History: A History object containing training history.
        """
        history = super().fit(data, epochs=epochs, callbacks=callbacks, **kwargs)
        self.setup_threshold(data)
        return history

    def setup_threshold(self, ds: tf.data.Dataset) -> None:
        """Sets up the threshold for anomaly detection based on the given dataset.

        Args:
            ds (tf.data.Dataset): The dataset to use for threshold calculation.
        """
        logger.info("Setting up the threshold ...")
        # builtin metrics:
        mean_metric = tf.keras.metrics.Mean()
        # custom metrics:
        median_metric = Median()
        std_metric = StandardDeviation()

        for x, _ in ds:
            ds_hat = self(x)
            scores = tf.reduce_mean(tf.abs(x - ds_hat), axis=1)
            mean_metric.update_state(scores)
            std_metric.update_state(scores)
            median_metric.update_state(scores)

        self._median.assign(median_metric.result())
        self._std.assign(std_metric.result())

        logger.debug(f"mean: {mean_metric.result().numpy()}")
        logger.debug(f"median: {median_metric.result().numpy()}")
        logger.debug(f"std: {std_metric.result().numpy()}")
        logger.debug(f"assigned _mean: {self._median}")
        logger.debug(f"assigned _std: {self._std}")

    def predict(self, data: tf.Tensor) -> np.ndarray:
        """Predicts anomaly scores for the given data.

        Args:
            data (tf.Tensor): The input data to predict on.

        Returns:
            np.ndarray: An array of anomaly scores.
        """
        x_pred = self(data)
        scores = tf.reduce_mean(tf.abs(data - x_pred), axis=1)
        return scores.numpy()

    def is_anomaly(self, data: tf.data.Dataset, percentile_to_use: str = "median") -> dict[str, list[np.ndarray]]:
        """Determines if the given data contains anomalies.

        Args:
            data (tf.data.Dataset): The dataset to check for anomalies.
            percentile_to_use (str, optional): The percentile to use for anomaly detection. Defaults to "median".

        Returns:
            dict[str, list[np.ndarray]]: A dictionary containing anomaly scores, flags, and threshold information.
        """
        scores = []
        anomalies = []
        self.percentile = getattr(self, percentile_to_use)

        for batch in data:
            batch_scores = self.predict(batch)
            batch_anomalies = batch_scores > self.percentile + (self.threshold * self.std)
            scores.append(batch_scores)
            anomalies.append(batch_anomalies)

        scores = np.concatenate(scores)
        anomalies = np.concatenate(anomalies)

        return {
            "score": scores.flatten(),
            "anomaly": anomalies,
            "std": self.std,
            "threshold": self.threshold,
            percentile_to_use: self.percentile,
        }

    def get_config(self) -> dict:
        """Returns the configuration of the model.

        Returns:
            dict: A dictionary containing the configuration of the model.
        """
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "encoding_dim": self.encoding_dim,
                "intermediate_dim": self.intermediate_dim,
                "threshold": self.threshold,
                "median": self.median,
                "std": self.std,
            },
        )
        return config

    @classmethod
    def from_config(cls, config) -> object:
        """Creates a new instance of the model from its config.

        Args:
            config (dict): A dictionary containing the configuration of the model.

        Returns:
            Autoencoder: A new instance of the model.
        """
        instance = cls(
            input_dim=config["input_dim"],
            encoding_dim=config["encoding_dim"],
            intermediate_dim=config["intermediate_dim"],
            threshold=config["threshold"],
        )
        instance._median.assign(config["median"])
        instance._std.assign(config["std"])
        return instance


class AnomalyDetectionModelProd(tf.keras.Model):
    """A production-ready anomaly detection model combining preprocessing and anomaly detection.

    This class combines a preprocessing model and an anomaly detection model into a single
    TensorFlow Keras model for production use. It calculates anomaly scores and determines
    if input data points are anomalies based on a threshold.

    Attributes:
        preprocessing_model (tf.keras.Model): The model used for preprocessing input data.
        anomaly_model (tf.keras.Model): The model used for anomaly detection.
        median (tf.Variable): The median of the anomaly scores.
        std (tf.Variable): The standard deviation of the anomaly scores.
        threshold (tf.Variable): The threshold for determining anomalies.
        percentile (tf.Variable): The percentile used for anomaly detection (set to median).
        anomaly_threshold (tf.Tensor): The calculated threshold for anomaly detection.
        inputs (dict[str, tuple]): A dictionary of input shapes for the model.
    """

    def __init__(
        self,
        preprocessing_model: tf.keras.Model | dict,
        anomaly_model: tf.keras.Model | dict,
        median: float,
        std: float,
        threshold: float,
        inputs: dict[str, tuple] | None = None,
    ) -> None:
        """Initializes the AnomalyDetectionModelProd.

        Args:
            preprocessing_model (tf.keras.Model | dict): The preprocessing model or its config dict.
            anomaly_model (tf.keras.Model | dict): The anomaly detection model or its config dict.
            median (float): The median of the anomaly scores.
            std (float): The standard deviation of the anomaly scores.
            threshold (float): The threshold for determining anomalies.
            inputs (Optional[dict[str, tuple]]): A dictionary of input shapes for the model.
        """
        super().__init__()
        self.preprocessing_model = self._ensure_model(preprocessing_model)
        self.anomaly_model = self._ensure_model(anomaly_model)
        self.median = tf.Variable(median, dtype=tf.float32)
        self.std = tf.Variable(std, dtype=tf.float32)
        self.threshold = tf.Variable(threshold, dtype=tf.float32)
        self.percentile = self.median
        self.anomaly_threshold = self.percentile + (self.threshold * self.std)

        # Store input shapes
        self.inputs = inputs

    def _ensure_model(self, model_or_dict: tf.keras.Model | dict) -> tf.keras.Model:
        """Ensures that the input is a tf.keras.Model.

        If the input is a dictionary, it converts it to a model using model_from_json.

        Args:
            model_or_dict (tf.keras.Model | dict): The model or its configuration dictionary.

        Returns:
            tf.keras.Model: The ensured Keras model.
        """
        if isinstance(model_or_dict, dict):
            return tf.keras.models.model_from_json(model_or_dict)
        return model_or_dict

    def call(self, inputs) -> dict[str, tf.Tensor]:
        """Performs the forward pass of the model.

        This method preprocesses the inputs, applies the anomaly detection model,
        and calculates the anomaly scores and flags.

        Args:
            inputs (dict): A dictionary of input tensors.

        Returns:
            dict[str, tf.Tensor]: A dictionary containing the following keys:
                - 'score': The anomaly scores.
                - 'anomaly': Boolean tensor indicating whether each input is an anomaly.
                - 'median': The median used for anomaly detection.
                - 'std': The standard deviation used for anomaly detection.
                - 'threshold': The threshold used for anomaly detection.
        """
        filtered_inputs = {k: v for k, v in inputs.items() if k in self.inputs}

        x = self.preprocessing_model(filtered_inputs)
        reconstructed_x = self.anomaly_model(x)
        anomaly_score = tf.reduce_mean(tf.abs(x - reconstructed_x), axis=1)
        is_anomaly = tf.math.greater(anomaly_score, self.anomaly_threshold)

        return {
            "score": anomaly_score,
            "anomaly": is_anomaly,
            "median": self.median,
            "std": self.std,
            "threshold": self.threshold,
        }

    def get_config(self) -> dict:
        """Returns the configuration of the model.

        Returns:
            dict: A dictionary containing the configuration of the model.
        """
        config = super().get_config()
        config.update(
            {
                "preprocessing_model": self.preprocessing_model.to_json(),
                "anomaly_model": self.anomaly_model.to_json(),
                "median": self.median.numpy(),
                "std": self.std.numpy(),
                "threshold": self.threshold.numpy(),
                "inputs": self.inputs,
            },
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> object:
        """Creates a new instance of the model from its config.

        Args:
            config (dict): A dictionary containing the configuration of the model.

        Returns:
            AnomalyDetectionModelProd: A new instance of the model.
        """
        instance = cls(
            preprocessing_model=tf.keras.models.model_from_json(config["preprocessing_model"]),
            anomaly_model=tf.keras.models.model_from_json(config["anomaly_model"]),
            median=config["median"],
            std=config["std"],
            threshold=config["threshold"],
            inputs=config["inputs"],
        )
        instance.median.assign(config["median"])
        instance.std.assign(config["std"])
        instance.threshold.assign(config["threshold"])
        return instance
