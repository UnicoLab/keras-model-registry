# -*- coding: utf-8 -*-
# project plugin

from pathlib import Path

import numpy as np
import tensorflow as tf
from config import Config
from loguru import logger
from theparrot.tf import CallbackManager

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

    def result(self) -> tf.Tensor:
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
        # tf.contrib.distributions.percentile(v, 50.0)
        self.values.assign(tf.cast(median, dtype=tf.float32))

    def result(self) -> tf.Tensor:
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


class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z a latent vector from the distribution."""

    def call(self, inputs: tf.data.Dataset) -> tf.Tensor:
        """Samples a latent vector from the distribution."""
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VariationalAutoencoder(tf.keras.models.Model):
    """VariationalAutoencoder backend functional model.

    This class builds and manages an autoencoder, a type of artificial neural network
    used for learning efficient codings of input data. The autoencoder consists of an encoder,
    a decoder, and a combined autoencoder model.

    Attributes:
        input_dim (int): The dimensionality of the input data.
        encoding_dim (int): The dimensionality of the encoded representations.
        intermediate_dim (int): The dimensionality of the intermediate dense layer.
        _encoder (tf.keras.Model): The encoder part of the autoencoder.
        _decoder (tf.keras.Model): The decoder part of the autoencoder.
        _autoencoder (tf.keras.Model): The complete autoencoder model.

    """

    def __init__(self, input_dim: int, encoding_dim: int, intermediate_dim: int) -> None:
        """Initializes the Autoencoder with the given dimensions.

        Args:
            input_dim (int): The dimensionality of the input data.
            encoding_dim (int): The dimensionality of the encoded representations.
            intermediate_dim (int): The dimensionality of the intermediate dense layer.

        """
        super().__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.intermediate_dim = intermediate_dim

        self._encoder = None
        self._decoder = None
        self._vae = None

        # metrics storage
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

        self._initialize()

    def _initialize(self) -> None:
        """Initializes the encoder, decoder, and autoencoder models."""
        # Building the encoder
        encoder_input = tf.keras.Input(shape=(self.input_dim,))
        encoded = tf.keras.layers.Dense(units=self.intermediate_dim, activation="relu")(encoder_input)
        encoded = tf.keras.layers.Dropout(0.2)(encoded)
        z_mean = tf.keras.layers.Dense(units=self.encoding_dim, name="z_mean")(encoded)
        z_log_var = tf.keras.layers.Dense(units=self.encoding_dim, name="z_log_var")(encoded)
        z = Sampling()([z_mean, z_log_var])
        self._encoder = tf.keras.Model(inputs=encoder_input, outputs=[z_mean, z_log_var, z], name="encoder")

        # Building the decoder
        decoder_input = tf.keras.Input(shape=(self.encoding_dim,))
        decoded = tf.keras.layers.Dense(units=self.intermediate_dim, activation="relu")(decoder_input)
        decoded = tf.keras.layers.Dropout(0.2)(decoded)
        decoded = tf.keras.layers.Dense(units=self.input_dim, activation="sigmoid")(decoded)
        decoded = tf.keras.layers.Dropout(0.2)(decoded)
        self._decoder = tf.keras.Model(inputs=decoder_input, outputs=decoded, name="decoder")

        # Building the autoencoder
        vae_input = tf.keras.Input(shape=(self.input_dim,))
        _, _, z = self._encoder(vae_input)
        decoded = self._decoder(z)
        self._vae = tf.keras.Model(inputs=vae_input, outputs=decoded, name="autoencoder")

    @property
    def encoder(self) -> tf.keras.Model:
        """tf.keras.Model: Returns the encoder part of the autoencoder."""
        return self._encoder

    @property
    def decoder(self) -> tf.keras.Model:
        """tf.keras.Model: Returns the decoder part of the autoencoder."""
        return self._decoder

    @property
    def vae(self) -> tf.keras.Model:
        """tf.keras.Model: Returns the complete autoencoder model."""
        return self._vae

    @property
    def metrics(self) -> list:
        """Returns the list of metrics tracked by the model."""
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs, training=False) -> tf.Tensor:
        """Perform the forward pass of the Variational Autoencoder.

        Args:
            inputs: Input tensor, or list/tuple of input tensors.
            training: Boolean indicating whether the call is for training or inference.

        Returns:
            The output tensor of the decoder.
        """
        if training:
            logger.warning(f"training arg passed: {training}")
        _, _, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed

    def compile_model(self, optimizer, loss, **kwargs) -> None:
        """Compiles the variational-autoencoder model with the given optimizer and loss function.

        Args:
            optimizer (str, tf.keras.optimizers.Optimizer): The optimizer to use.
            loss (str, tf.keras.losses.Loss): The loss function to use.
            **kwargs: Additional keyword arguments passed to the parent class.

        """
        self._vae.compile(optimizer=optimizer, loss=loss, **kwargs)

    def train_step(self, data) -> dict:
        """Trains the model on the given data."""
        epsilon = 1e-7  # Small constant for numerical stability
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(data - reconstruction)))

            # Improved KL divergence calculation with epsilon
            kl_loss = -0.5 * tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(tf.clip_by_value(z_log_var, -10, 10)) + epsilon,
                axis=-1,
            )
            kl_loss = tf.reduce_mean(kl_loss)

            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]  # Gradient clipping
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights, strict=False))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def predict(self, data: tf.data.Dataset) -> tf.data.Dataset:
        """Makes predictions with the v-autoencoder model on the given data.

        Args:
            data (tf.data.Dataset): The data to make predictions on.

        Returns:
            tf.data.Dataset: The predicted output.

        """
        return self._vae.predict(data)

    def summary(self) -> None:
        """Printing combined model summary."""
        logger.info("Encoder Summary")
        self._encoder.summary()
        logger.info("\nDecoder Summary")
        self._decoder.summary()
        logger.info("\nVariational-Autoencoder Summary")
        self._vae.summary()


class AnomalyDetector:
    """Anomaly detection model based on a variational autoencoder."""

    def __init__(
        self,
        input_dim: int,
        intermediate_dim: int = 64,
        encoding_dim: int = 32,
        threshold: float = 2,
    ):
        """Initialize the anomaly detector.

        Args:
            input_dim (int): Dimension of the input data.
            intermediate_dim (int): Dimension of the intermediate dense layer in the VAE.
            encoding_dim (int): Dimension of the latent space in the AE.
            threshold (float): Z-score threshold for anomaly detection.

        Note:
            This model is a wrapper on top of the VariationalAutoencoder to simplyfy the training process
            and won't be usefull for inference itself as the model will be packaged with the preprocessor.

        """
        logger.info("Initializing the anomaly detector VariationalAutoEncoder model")
        self.vae = VariationalAutoencoder(
            input_dim=input_dim,
            intermediate_dim=intermediate_dim,
            encoding_dim=encoding_dim,
        )
        logger.info("Initializing placeholders")
        # initializing anomalies detection thresholds storage
        self.threshold = threshold
        self.mean = None
        self.median = None
        self.std = None
        logger.info("Anomaly detector initialized.")

    def setup_callbacks(self) -> list:
        """Sets up the callbacks for keras model."""
        # assuring existance of the CSV output folder
        Path(conf.paths.TRAIN__UPLOAD__SYNC__ARTIFACTS, conf.model.MODEL_NAME).mkdir(parents=True, exist_ok=True)

        # settin gup callbacks
        manager = CallbackManager(
            model_name=conf.model.MODEL_NAME,
            root_folder=conf.paths.TRAIN__UPLOAD__SYNC__ARTIFACTS,
            USE_CHECKPOINT=conf.callback.CALLBACK_USE_CHECKPOINT,
            USE_TENSORBOARD=conf.callback.CALLBACK_USE_TENSORBOARD,
            USE_CSV_LOGGER=conf.callback.CALLBACK_USE_CSV_LOGGER,
            USE_NAN_CHECK=conf.callback.CALLBACK_USE_NAN_CHECK,
            USE_REDUCE_ON_PLATEAU=conf.callback.CALLBACK_USE_REDUCE_ON_PLATEAU,
            REDUCE_ON_PLATEAU_CONFIG=conf.callback.CALLBACK_REDUCE_ON_PLATEAU_CONFIG,
            USE_EARLY_STOPPING=conf.callback.CALLBACK_USE_EARLY_STOPPING,
            EARLY_STOPPING_CONFIG=conf.callback.CALLBACK_USE_EARLY_STOPPING_CONFIG,
        )
        callbacks_list = manager.setup_callbacks()
        logger.info("Model Callbacks OK")
        return callbacks_list

    def train(self, ds, epochs: int = 2) -> object:
        """Train the anomaly detector on a batch of data.

        Args:
            ds (tf.data.DataSet): DataSet to train the anomaly detector on.
            epochs (int): Number of epochs to train for.
            batch_size (int): Batch size to use during training.

        Returns:
            fit_history (object): Model Fit hostory object

        Note:
            Data that needs to passed for training should be:
            data = (
                pm.ds
                .map(lambda x: (preprocessing_model(x), preprocessing_model(x)))
            )

        """
        logger.info("Compiling the anomaly detector model")
        self.vae.compile(optimizer="adam")
        logger.info("Model compiled with optimizer: adam and loss: mean_squared_error ✅")

        logger.info("Starting model training... ⚙️")
        self.fit_history = self.vae.fit(
            ds,
            epochs=epochs,
            callbacks=self.setup_callbacks(),
        )
        logger.info("Training completed ✅, predicting scores for the training data (incrementally) 📊")

        # setting up threshold
        logger.info("Setting up threshold")
        self.setup_threshold(ds=ds)
        return self.fit_history

    def setup_threshold(self, ds: tf.data.Dataset) -> None:
        """Extracting meand and standard deviation from a dataset to setup anomalies detection threshold.

        Args:
            ds (tf.data.Dataset): training dataset to extract statistics from.
        """
        # The mean and std in a streaming (or online) manner, where you update the statistics for each batch.
        # TensorFlow offers tf.keras.metrics.Mean and tf.keras.metrics.StandardDeviation for this purpose.
        # These metrics maintain a state (the total and count so far) that can be updated incrementally batch by batch.
        mean_metric = tf.keras.metrics.Mean()
        median_metric = Median()
        std_metric = StandardDeviation()

        for x in ds:
            # Predict on the preprocessed input
            ds_hat = self.vae.predict(x)
            # calculating scores
            scores = tf.reduce_mean(tf.abs(x - ds_hat), axis=1)
            # updating metrics
            mean_metric.update_state(scores)
            std_metric.update_state(scores)
            median_metric.update_state(scores)

        logger.info("Retrieving total values of stats")
        self.mean = mean_metric.result().numpy()
        self.median = median_metric.result().numpy()
        self.std = std_metric.result().numpy()

        logger.info(f"Mean: {self.mean}, median: {self.median}, standard deviation: {self.std} estimated ✅")
        logger.info("Setting up the threshold compleate completed -> ✅!")

    def predict(self, ds: tf.data.Dataset) -> np.ndarray:
        """Compute anomaly scores for a batch of data.

        Args:
            ds (tf.data.DataSet): Dataset to predict anomaly scores for.

        Returns:
            np.array: Anomaly scores for each sample in the batch.

        """
        x_pred: np.ndarray = self.vae.predict(ds)
        logger.info("Predicting anomaly scores samples 📊")
        scores = tf.reduce_mean(tf.abs(ds - x_pred), axis=1)
        return scores.numpy()

    def is_anomaly(self, ds: tf.data.Dataset, percentile_to_use: str = "median") -> dict[str, list[np.ndarray]]:
        """Determine whether each sample in a batch of data is an anomaly.

        Args:
            ds (tf.data.Dataset): Dataset to check for anomalies.
            percentile_to_use (str): Percentile to use for anomaly detection.

        Returns:
            (Dict[str, List[np.ndarray]]): A dictionary containing the anomaly scores for each sample in the batch

        """
        scores = []
        anomalies = []
        # choosing percentile to use
        self.percentile = getattr(self, percentile_to_use)

        logger.info(f"Applying anomaly detection threshold in batches: {self.threshold} ⚙️")
        for batch in ds:
            logger.debug("Predicting batch of data")
            batch_scores = self.predict(batch)
            batch_anomalies = batch_scores > self.percentile + (self.threshold * self.std)
            # appending storage
            scores.append(batch_scores)
            anomalies.append(batch_anomalies)

        logger.debug("Concatenating scores and anomalies 🔍")
        scores = np.concatenate(scores)
        anomalies = np.concatenate(anomalies)

        # extracting stats
        logger.debug("Extracting statistics 📊")
        u, c = np.unique(anomalies, return_counts=True)
        perc_anomalies = (c[1] / c[0]) * 100
        logger.info(f"Anomaly scores: {u}, counts: {c}, anomalies: {perc_anomalies} % 🚨")
        # defining outputs
        output = {
            "score": scores.flatten(),
            "anomaly": anomalies,
            "std": self.std,
            "threshold": self.threshold,
        }
        # adding percentile to the output
        output[percentile_to_use] = self.percentile

        return output


class AnomalyDetectionModelProd(tf.keras.Model):
    """Production ready anomaly detector model with preprocessing and all required training stats included."""

    def __init__(
        self,
        preprocessing_model: tf.keras.models.Model,
        anomaly_model: tf.keras.models.Model,
        percentile: float,
        std: float,
        threshold: int,
    ) -> None:
        """Anomaly detection model using AutoEncoder with building preprocessing model.

        Arguments:
            preprocessing_model (tf.keras.models.Model): Preprocessing model.
            anomaly_model (tf.keras.models.Model): Anomaly detection model.
            percentile (float): Mean anomaly score.
            std (float): Standard deviation anomaly score.
            threshold (int): Threshold for anomaly detection.

        Note:
            Before saving model to the tf format we need to either:
                - get a batch of data through it so that the TF figures out the inputs shape
                - we can further define the inputs themselfs somehow (TODO), to get rid of the first required step

        """
        super().__init__()
        # models:
        self.preprocessing_model = preprocessing_model
        self.anomaly_model = anomaly_model
        # variables:
        self.percentile = tf.Variable(percentile, dtype=tf.float32)
        self.std = tf.Variable(std, dtype=tf.float32)
        self.threshold = tf.Variable(threshold, dtype=tf.float32)
        self.anomaly_threshold = self.percentile + (self.threshold * self.std)

        # Get the input shape from the preprocessing model
        self.input_shapes = preprocessing_model.input_shape

        # Define the inputs
        self.inputs = {name: tf.keras.Input(shape=shape[1:], name=name) for name, shape in self.input_shapes.items()}

    def call(self, inputs) -> dict[str, tf.Tensor]:
        """Main model entrypoint.

        Args:
            inputs (_type_): Ordered dictionary containing input data for anomalies estimation.

        Returns:
            (Dict[str, Any]): Response containing data about anomaly estimation.

        Note:
            We need to output tf.tensors and not .numpy() if we want the model to be serializable in the tf format.

        """
        # Filter input data to match the expected keys
        filtered_inputs = {k: v for k, v in inputs.items() if k in self.inputs}

        x = self.preprocessing_model(filtered_inputs)
        reconstructed_x = self.anomaly_model(x)
        anomaly_score = tf.reduce_mean(tf.abs(x - reconstructed_x), axis=1)
        is_anomaly = tf.math.greater(anomaly_score, self.anomaly_threshold)

        return {
            "score": anomaly_score,
            "anomaly": is_anomaly,
            "percentile": self.percentile,
            "std": self.std,
            "threshold": self.threshold,
        }

    def get_config(self) -> dict:
        """Return the configuration of the anomaly detection model."""
        config = super().get_config()
        config.update(
            {
                "preprocessing_model": self.preprocessing_model,
                "anomaly_model": self.anomaly_model,
                "percentile": self.percentile.numpy(),
                "std": self.std.numpy(),
                "threshold": self.threshold.numpy(),
            },
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> object:
        """Create an anomaly detection model from its configuration."""
        return cls(**config)


class AnomalyDetectionModelProdFunctional(tf.keras.Model):
    """Anomaly detection model that integrates preprocessing and anomaly detection
    within a functional API architecture, with added functionality to filter out
    unexpected input data fields.

    This model assumes that `preprocessing_model` and `anomaly_model` are provided
    as pre-trained or predefined models. It applies these models sequentially to
    input data that matches their expected input shapes, ignoring any additional
    input fields.

    Attributes:
        preprocessing_model (tf.keras.Model): Model for preprocessing input data.
        anomaly_model (tf.keras.Model): Model for detecting anomalies in preprocessed data.
        percentile (float): The percentile value used to calculate the anomaly threshold.
        std (float): The standard deviation value used to calculate the anomaly threshold.
        threshold (int): The threshold value for determining an anomaly.
        model (tf.keras.Model): The integrated model combining preprocessing and anomaly detection.

    Example:
        ```python
        preprocessing_model = ...  # Define or load your preprocessing model
        anomaly_model = ...  # Define or load your anomaly detection model
        model = AnomalyDetectionModelProd(preprocessing_model, anomaly_model, 95.0, 5.0, 1)

        # Prepare your input data as a dictionary where keys match the expected input names
        input_data = {"feature1": data1, "feature2": data2, "unused_feature": data3}

        # Predict anomalies (note that 'unused_feature' will be ignored)
        results = model.predict(input_data)
        ```
    """

    def __init__(
        self,
        preprocessing_model: tf.keras.Model,
        anomaly_model: tf.keras.Model,
        percentile: float,
        std: float,
        threshold: int,
    ) -> None:
        """Initialize the AnomalyDetectionModelProd2 class."""
        super().__init__()
        self.preprocessing_model = preprocessing_model
        self.anomaly_model = anomaly_model

        # Initialize percentile, std, and threshold as TensorFlow variables
        self.percentile = tf.Variable(percentile, dtype=tf.float32)
        self.std = tf.Variable(std, dtype=tf.float32)
        self.threshold = tf.Variable(threshold, dtype=tf.float32)
        self.anomaly_threshold = self.percentile + (self.threshold * self.std)

        # Dynamically create inputs based on the preprocessing model's input shape
        self.input_shapes = dict(zip(preprocessing_model.input_names, preprocessing_model.input_shape, strict=True))
        self.inputs = {name: tf.keras.Input(shape=shape[1:], name=name) for name, shape in self.input_shapes.items()}

        # Data Flow definition
        x = preprocessing_model(self.inputs)
        reconstructed_x = anomaly_model(x)
        anomaly_score = tf.reduce_mean(tf.abs(x - reconstructed_x), axis=1)
        is_anomaly = tf.math.greater(anomaly_score, self.anomaly_threshold)

        # outputs definition
        self.outputs = {
            "score": anomaly_score,
            "anomaly": is_anomaly,
            "percentile": self.percentile,
            "std": self.std,
            "threshold": self.threshold,
        }

        # Create the functional model
        self.model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)

    def call(self, inputs: dict[str, tf.Tensor]) -> tf.Tensor:
        """Main model entrypoint."""
        # Filter input data to match the expected keys from preprocessing_model
        filtered_inputs = {k: v for k, v in inputs.items() if k in self.inputs}
        return self.model(filtered_inputs)
