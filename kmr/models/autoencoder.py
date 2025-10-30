"""Autoencoder model for anomaly detection in the Keras Model Registry.

This module provides an autoencoder-based model for anomaly detection that can
optionally integrate with preprocessing models for production use.

Example:
    ```python
    import keras
    from kmr.models import Autoencoder

    # Create and train an autoencoder
    autoencoder = Autoencoder(
        input_dim=100,
        encoding_dim=32,
        intermediate_dim=64
    )
    
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(data, epochs=10)
    
    # Use for anomaly detection
    scores = autoencoder.predict_anomaly_scores(test_data)
    anomalies = autoencoder.is_anomaly(test_data)
    
    # With preprocessing model
    preprocessing_model = keras.Sequential([...])
    autoencoder_with_preprocessing = Autoencoder(
        input_dim=100,
        encoding_dim=32,
        preprocessing_model=preprocessing_model
    )
    ```
"""

from typing import Any

import keras
from keras import layers, ops
from keras.saving import register_keras_serializable
from loguru import logger

from kmr.models._base import BaseModel
from kmr.metrics import StandardDeviation, Median


@register_keras_serializable(package="kmr.models")
class Autoencoder(BaseModel):
    """An autoencoder model for anomaly detection with optional preprocessing integration.

    This class implements an autoencoder neural network model used for anomaly detection.
    It can optionally integrate with preprocessing models for production use, making it
    a single, unified model for both training and inference.

    Attributes:
        input_dim (int): The dimension of the input data.
        encoding_dim (int): The dimension of the encoded representation.
        intermediate_dim (int): The dimension of the intermediate layer.
        preprocessing_model (keras.Model | None): Optional preprocessing model.
        _threshold (keras.Variable): The threshold for anomaly detection.
        _median (keras.Variable): The median of the anomaly scores.
        _std (keras.Variable): The standard deviation of the anomaly scores.
    """

    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = 64,
        intermediate_dim: int = 32,
        threshold: float = 2.0,
        preprocessing_model: keras.Model | None = None,
        inputs: dict[str, tuple[int, ...]] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the Autoencoder model.

        Args:
            input_dim (int): The dimension of the input data.
            encoding_dim (int, optional): The dimension of the encoded representation. Defaults to 64.
            intermediate_dim (int, optional): The dimension of the intermediate layer. Defaults to 32.
            threshold (float, optional): The initial threshold for anomaly detection. Defaults to 2.0.
            preprocessing_model (keras.Model, optional): Optional preprocessing model for production use. Defaults to None.
            inputs (dict[str, tuple], optional): Input shapes for preprocessing model. Defaults to None.
            name (str, optional): The name of the model. Defaults to None.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        # Set private attributes first
        self._input_dim = input_dim
        self._encoding_dim = encoding_dim
        self._intermediate_dim = intermediate_dim
        self._threshold = threshold

        # Validate parameters
        self._validate_params()

        # Set public attributes BEFORE calling parent's __init__
        self.input_dim = self._input_dim
        self.encoding_dim = self._encoding_dim
        self.intermediate_dim = self._intermediate_dim

        # Initialize variables
        self._threshold_var = keras.Variable(
            threshold,
            dtype="float32",
            name="threshold",
        )
        self._median = keras.Variable(
            0.0,
            dtype="float32",
            trainable=False,
            name="median",
        )
        self._std = keras.Variable(0.0, dtype="float32", trainable=False, name="std")

        # Call parent's __init__ with preprocessing model support
        super().__init__(
            preprocessing_model=preprocessing_model,
            inputs=inputs,
            name=name,
            **kwargs,
        )

        # Build the model architecture
        self._build_architecture()

    def _validate_params(self) -> None:
        """Validate model parameters."""
        if self._input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self._input_dim}")
        if self._encoding_dim <= 0:
            raise ValueError(f"encoding_dim must be positive, got {self._encoding_dim}")
        if self._intermediate_dim <= 0:
            raise ValueError(
                f"intermediate_dim must be positive, got {self._intermediate_dim}",
            )
        if self._threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {self._threshold}")

    def _build_architecture(self) -> None:
        """Build the autoencoder architecture."""
        # Encoder layers
        self.encoder_dense1 = layers.Dense(
            self.intermediate_dim,
            activation="relu",
            name="encoder_dense1",
        )
        self.encoder_dropout1 = layers.Dropout(0.1, name="encoder_dropout1")
        self.encoder_dense2 = layers.Dense(
            self.encoding_dim,
            activation="relu",
            name="encoder_dense2",
        )
        self.encoder_dropout2 = layers.Dropout(0.1, name="encoder_dropout2")

        # Decoder layers
        self.decoder_dense1 = layers.Dense(
            self.intermediate_dim,
            activation="relu",
            name="decoder_dense1",
        )
        self.decoder_dropout1 = layers.Dropout(0.1, name="decoder_dropout1")
        self.decoder_dense2 = layers.Dense(
            self.input_dim,
            activation="sigmoid",
            name="decoder_dense2",
        )
        self.decoder_dropout2 = layers.Dropout(0.1, name="decoder_dropout2")

        logger.debug(
            f"Autoencoder built with input_dim={self.input_dim}, "
            f"encoding_dim={self.encoding_dim}, intermediate_dim={self.intermediate_dim}, "
            f"preprocessing_model={'Yes' if self.preprocessing_model else 'No'}",
        )

    def call(
        self,
        inputs: Any,
        training: bool | None = None,
    ) -> keras.KerasTensor | dict[str, keras.KerasTensor]:
        """Performs the forward pass of the autoencoder with universal input handling.

        This method supports various input formats:
        - Single tensors/vectors (numpy arrays, tensors)
        - Lists/tuples of tensors
        - Dictionaries (regular dict, OrderedDict)
        - Mixed input formats

        Args:
            inputs: Input data in various formats (dict, list, tensor, etc.)
            training (bool, optional): Whether the model is in training mode. Defaults to None.

        Returns:
            KerasTensor | dict: The reconstructed input data or anomaly detection results.
        """
        # Use BaseModel's intelligent input processing
        # For autoencoder, we don't need feature splitting, just concatenation
        processed_inputs = self._process_inputs_for_model(
            inputs,
            expected_keys=None,  # No specific feature names for autoencoder
            auto_split=False,  # Don't split single inputs
            auto_reshape=False,  # Don't reshape, let the model handle it
        )

        # Handle the processed inputs
        if isinstance(processed_inputs, list):
            # Multiple inputs - concatenate them
            x = keras.ops.concatenate(processed_inputs, axis=-1)
        else:
            # Single input
            x = processed_inputs

        # Encoder
        encoded = self.encoder_dense1(x)
        encoded = self.encoder_dropout1(encoded, training=training)
        encoded = self.encoder_dense2(encoded)
        encoded = self.encoder_dropout2(encoded, training=training)

        # Decoder
        decoded = self.decoder_dense1(encoded)
        decoded = self.decoder_dropout1(decoded, training=training)
        decoded = self.decoder_dense2(decoded)
        decoded = self.decoder_dropout2(decoded, training=training)

        # If preprocessing model is used, return anomaly detection results
        if self.preprocessing_model is not None:
            # Calculate anomaly score
            anomaly_score = ops.mean(ops.abs(x - decoded), axis=1)

            # Determine if anomaly
            is_anomaly = ops.greater(
                anomaly_score,
                self._median + (self._threshold_var * self._std),
            )

            return {
                "reconstruction": decoded,
                "score": anomaly_score,
                "anomaly": is_anomaly,
                "median": self._median,
                "std": self._std,
                "threshold": self._threshold_var,
            }

        return decoded

    @property
    def threshold(self) -> float:
        """Gets the current threshold value.

        Returns:
            float: The current threshold value.
        """
        return float(self._threshold_var.numpy())

    @property
    def median(self) -> float:
        """Gets the current median value.

        Returns:
            float: The current median value.
        """
        return float(self._median.numpy())

    @property
    def std(self) -> float:
        """Gets the current standard deviation value.

        Returns:
            float: The current standard deviation value.
        """
        return float(self._std.numpy())

    def setup_threshold(self, data: keras.KerasTensor | Any) -> None:
        """Sets up the threshold for anomaly detection based on the given data.

        This method automatically calculates the median and standard deviation of
        reconstruction errors from the provided data and sets up the threshold
        for anomaly detection.

        Args:
            data (KerasTensor | Any): The data to use for threshold calculation.
                Can be a tensor or a dataset.
        """
        logger.info("Setting up the threshold ...")

        # Built-in metrics
        mean_metric = keras.metrics.Mean()
        # Custom metrics
        median_metric = Median()
        std_metric = StandardDeviation()

        # Handle both tensor and dataset inputs
        if (
            hasattr(data, "__iter__")
            and not isinstance(data, keras.KerasTensor)
            and hasattr(data, "__class__")
            and "Dataset" in str(type(data))
        ):
            # Process dataset batch by batch
            for batch in data:
                if isinstance(batch, tuple):
                    # If dataset contains (features, labels), use features only
                    x = batch[0]
                else:
                    x = batch

                # Calculate reconstruction errors
                reconstructed = self(x, training=False)
                scores = ops.mean(ops.abs(x - reconstructed), axis=1)

                # Update metrics
                mean_metric.update_state(scores)
                std_metric.update_state(scores)
                median_metric.update_state(scores)
        else:
            # Handle tensor input
            reconstructed = self(data, training=False)
            scores = ops.mean(ops.abs(data - reconstructed), axis=1)

            # Update metrics
            mean_metric.update_state(scores)
            std_metric.update_state(scores)
            median_metric.update_state(scores)

        # Update model variables
        self._median.assign(median_metric.result())
        self._std.assign(std_metric.result())

        logger.debug(f"mean: {mean_metric.result().numpy()}")
        logger.debug(f"median: {median_metric.result().numpy()}")
        logger.debug(f"std: {std_metric.result().numpy()}")
        logger.debug(f"assigned _median: {self._median}")
        logger.debug(f"assigned _std: {self._std}")

    def auto_configure_threshold(
        self,
        data: keras.KerasTensor | Any,
        percentile: float = 0.95,
        method: str = "iqr",
    ) -> None:
        """Automatically configure threshold using statistical methods.

        This method provides different approaches to automatically set the
        anomaly detection threshold based on statistical properties of the data.

        Args:
            data (KerasTensor | Any): The data to use for threshold calculation.
            percentile (float, optional): Percentile to use for threshold calculation. Defaults to 0.95.
            method (str, optional): Method to use for threshold calculation.
                Options: 'iqr' (Interquartile Range), 'percentile', 'zscore'. Defaults to 'iqr'.
        """
        logger.info(f"Auto-configuring threshold using method: {method}")

        # Calculate reconstruction errors
        scores = []

        if (
            hasattr(data, "__iter__")
            and not isinstance(data, keras.KerasTensor)
            and hasattr(data, "__class__")
            and "Dataset" in str(type(data))
        ):
            for batch in data:
                if isinstance(batch, tuple):
                    x = batch[0]
                else:
                    x = batch
                batch_scores = self.predict_anomaly_scores(x)
                scores.append(batch_scores.numpy())
        else:
            batch_scores = self.predict_anomaly_scores(data)
            scores.append(batch_scores.numpy())

        # Concatenate all scores
        all_scores = ops.concatenate([ops.convert_to_tensor(s) for s in scores])

        if method == "iqr":
            # Interquartile Range method
            q1 = ops.quantile(all_scores, 0.25)
            q3 = ops.quantile(all_scores, 0.75)
            iqr = q3 - q1
            threshold_value = q3 + 1.5 * iqr
        elif method == "percentile":
            # Percentile method
            threshold_value = ops.quantile(all_scores, percentile)
        elif method == "zscore":
            # Z-score method (assuming 3 standard deviations)
            mean_score = ops.mean(all_scores)
            std_score = ops.std(all_scores)
            threshold_value = mean_score + 3 * std_score
        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'iqr', 'percentile', or 'zscore'",
            )

        # Update threshold variable
        self._threshold_var.assign(ops.cast(threshold_value, dtype="float32"))

        # Also update median and std for consistency
        self._median.assign(ops.cast(ops.median(all_scores), dtype="float32"))
        self._std.assign(ops.cast(ops.std(all_scores), dtype="float32"))

        logger.info(f"Auto-configured threshold: {threshold_value.numpy()}")
        logger.debug(f"Updated median: {self._median.numpy()}")
        logger.debug(f"Updated std: {self._std.numpy()}")

    def fit(
        self,
        x: keras.KerasTensor | Any = None,
        y: Any = None,
        epochs: int = 1,
        auto_setup_threshold: bool = True,
        threshold_method: str = "iqr",
        callbacks: list | None = None,
        **kwargs: Any,
    ) -> keras.callbacks.History:
        """Fits the model to the given data with optional automatic threshold setup.

        Args:
            x (KerasTensor | Any): The training data (features).
            y (Any): The training targets (labels).
            epochs (int): The number of epochs to train for.
            auto_setup_threshold (bool, optional): Whether to automatically setup threshold after training. Defaults to True.
            threshold_method (str, optional): Method for threshold setup. Defaults to "iqr".
            callbacks (list, optional): A list of callbacks to use during training. Defaults to None.
            **kwargs: Additional keyword arguments passed to the fit method.

        Returns:
            keras.callbacks.History: A History object containing training history.
        """
        # Use the base class fit method which handles preprocessing model integration
        history = super().fit(x=x, y=y, epochs=epochs, callbacks=callbacks, **kwargs)

        # Automatically setup threshold if requested (autoencoder-specific functionality)
        if auto_setup_threshold and x is not None:
            logger.info("Auto-setting up threshold after training...")
            if threshold_method in ["iqr", "percentile", "zscore"]:
                self.auto_configure_threshold(x, method=threshold_method)
            else:
                self.setup_threshold(x)

        return history

    def create_functional_model(self) -> keras.Model | None:
        """Create a functional model that combines preprocessing and autoencoder.

        This method creates a functional Keras model that integrates the preprocessing
        model (if provided) with the autoencoder for end-to-end inference.

        Returns:
            keras.Model: Functional model combining preprocessing and autoencoder, or None if no preprocessing.
        """
        return self._create_functional_model()

    def predict_anomaly_scores(self, data: keras.KerasTensor) -> keras.KerasTensor:
        """Predicts anomaly scores for the given data.

        Args:
            data (KerasTensor): The input data to predict on.

        Returns:
            KerasTensor: An array of anomaly scores.
        """
        x_pred = self(data, training=False)
        # Ensure both tensors have the same dtype to avoid type mismatch errors
        data = ops.cast(data, x_pred.dtype)
        scores = ops.mean(ops.abs(data - x_pred), axis=1)
        return scores

    def predict(
        self,
        data: keras.KerasTensor | dict[str, keras.KerasTensor] | Any,
        **kwargs,
    ) -> keras.KerasTensor | dict[str, keras.KerasTensor]:
        """Predicts reconstruction or anomaly detection results.

        This method provides a unified interface for both reconstruction prediction
        and anomaly detection, depending on whether a preprocessing model is used.

        Args:
            data (KerasTensor | dict | Any): The input data to predict on.
            **kwargs: Additional keyword arguments (ignored for compatibility).

        Returns:
            KerasTensor | dict: Reconstruction results or anomaly detection results.
        """
        # Handle dataset inputs
        if (
            hasattr(data, "__iter__")
            and not isinstance(data, keras.KerasTensor)
            and not isinstance(data, dict)
            and hasattr(data, "__class__")
            and "Dataset" in str(type(data))
        ):
            # Process dataset batch by batch
            predictions = []
            for batch in data:
                if isinstance(batch, tuple):
                    # If dataset contains (features, labels), use features only
                    x = batch[0]
                else:
                    x = batch
                batch_pred = self(x, training=False)
                predictions.append(batch_pred)
            # Concatenate all predictions
            return ops.concatenate(predictions)
        else:
            return self(data, training=False)

    def is_anomaly(
        self,
        data: keras.KerasTensor | dict[str, keras.KerasTensor] | Any,
        percentile_to_use: str = "median",
    ) -> dict[str, Any]:
        """Determines if the given data contains anomalies.

        This method can handle both individual samples and datasets, providing
        comprehensive anomaly detection results.

        Args:
            data (KerasTensor | dict | Any): The data to check for anomalies.
            percentile_to_use (str, optional): The percentile to use for anomaly detection. Defaults to "median".

        Returns:
            dict[str, Any]: A dictionary containing anomaly scores, flags, and threshold information.
        """
        if (
            hasattr(data, "__iter__")
            and not isinstance(data, keras.KerasTensor)
            and not isinstance(data, dict)
            and hasattr(data, "__class__")
            and "Dataset" in str(type(data))
        ):
            # Handle dataset input
            scores = []
            anomalies = []

            for batch in data:
                if isinstance(batch, tuple):
                    x = batch[0]
                else:
                    x = batch

                # Calculate scores directly to avoid recursion
                if self.preprocessing_model is not None:
                    # Use the call method which handles preprocessing and returns anomaly results
                    results = self(x, training=False)
                    batch_scores = results["score"]
                    batch_anomalies = results["anomaly"]
                else:
                    # Standard autoencoder mode
                    batch_scores = self.predict_anomaly_scores(x)
                    percentile = getattr(self, percentile_to_use)
                    batch_anomalies = ops.cast(
                        batch_scores > (percentile + (self.threshold * self.std)),
                        dtype="bool",
                    )

                scores.append(batch_scores)
                anomalies.append(batch_anomalies)

            # Concatenate results
            all_scores = ops.concatenate(scores)
            all_anomalies = ops.concatenate(anomalies)

            return {
                "score": all_scores,
                "anomaly": all_anomalies,
                "std": self.std,
                "threshold": self.threshold,
                percentile_to_use: getattr(self, percentile_to_use),
            }

        if self.preprocessing_model is not None:
            # Use the call method which handles preprocessing and returns anomaly results
            results = self(data, training=False)
            return {
                "score": results["score"],
                "anomaly": results["anomaly"],
                "std": results["std"],
                "threshold": results["threshold"],
                percentile_to_use: results["median"],
            }
        else:
            # Standard autoencoder mode
            scores = self.predict_anomaly_scores(data)
            percentile = getattr(self, percentile_to_use)

            anomalies = ops.cast(
                scores > (percentile + (self.threshold * self.std)),
                dtype="bool",
            )

            return {
                "score": scores,
                "anomaly": anomalies,
                "std": self.std,
                "threshold": self.threshold,
                percentile_to_use: percentile,
            }

    def get_config(self) -> dict[str, Any]:
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
                "preprocessing_model": self.preprocessing_model.to_json()
                if self.preprocessing_model
                else None,
                "inputs": self.inputs,
            },
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Autoencoder":
        """Creates a new instance of the model from its config.

        Args:
            config (dict): A dictionary containing the configuration of the model.

        Returns:
            Autoencoder: A new instance of the model.
        """
        preprocessing_model = None
        if config.get("preprocessing_model"):
            preprocessing_model = keras.models.model_from_json(
                config["preprocessing_model"],
            )

        instance = cls(
            input_dim=config["input_dim"],
            encoding_dim=config["encoding_dim"],
            intermediate_dim=config["intermediate_dim"],
            threshold=config["threshold"],
            preprocessing_model=preprocessing_model,
            inputs=config.get("inputs"),
        )
        instance._median.assign(config["median"])
        instance._std.assign(config["std"])
        return instance
