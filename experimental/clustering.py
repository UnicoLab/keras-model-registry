# standard library
from typing import List, Tuple

# pypi/conda library
import numpy as np
import tensorflow as tf
from loguru import logger
from tqdm import tqdm

# project plugin
from config import Config

CONF = Config()


class Autoencoder(tf.keras.models.Model):
    """
    Autoencoder backend functional model.

    This class builds and manages an autoencoder, a type of artificial neural network used for learning efficient codings of input data.
    The autoencoder consists of an encoder, a decoder, and a combined autoencoder model.

    Attributes:
        input_dim (int): The dimensionality of the input data.
        encoding_dim (int): The dimensionality of the encoded representations.
        intermediate_dim (int): The dimensionality of the intermediate dense layer.
        _encoder (tf.keras.Model): The encoder part of the autoencoder.
        _decoder (tf.keras.Model): The decoder part of the autoencoder.
        _autoencoder (tf.keras.Model): The complete autoencoder model.
    """

    def __init__(self, input_dim: int, encoding_dim: int, intermediate_dim: int):
        """
        Initializes the Autoencoder with the given dimensions.

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
        self._autoencoder = None

        self._initialize()

    def _initialize(self):
        """
        Initializes the encoder, decoder, and autoencoder models.
        """
        # Building the encoder
        encoder_input = tf.keras.Input(shape=(self.input_dim,))
        encoded = tf.keras.layers.Dense(units=self.intermediate_dim, activation="relu")(encoder_input)
        encoded = tf.keras.layers.Dropout(0.1)(encoded)
        encoded = tf.keras.layers.Dense(units=self.encoding_dim, activation="relu")(encoded)
        encoded = tf.keras.layers.Dropout(0.1)(encoded)
        self._encoder = tf.keras.Model(inputs=encoder_input, outputs=encoded, name="encoder")

        # Building the decoder
        decoder_input = tf.keras.Input(shape=(self.encoding_dim,))
        decoded = tf.keras.layers.Dense(units=self.intermediate_dim, activation="relu")(decoder_input)
        decoded = tf.keras.layers.Dropout(0.1)(decoded)
        decoded = tf.keras.layers.Dense(units=self.input_dim, activation="sigmoid")(decoded)
        decoded = tf.keras.layers.Dropout(0.1)(decoded)
        self._decoder = tf.keras.Model(inputs=decoder_input, outputs=decoded, name="decoder")

        # Building the autoencoder
        autoencoder_input = tf.keras.Input(shape=(self.input_dim,))
        encoded = self._encoder(autoencoder_input)
        decoded = self._decoder(encoded)
        self._autoencoder = tf.keras.Model(inputs=autoencoder_input, outputs=decoded, name="autoencoder")

    @property
    def encoder(self):
        """
        tf.keras.Model: Returns the encoder part of the autoencoder.
        """
        return self._encoder

    @property
    def decoder(self):
        """
        tf.keras.Model: Returns the decoder part of the autoencoder.
        """
        return self._decoder

    @property
    def autoencoder(self):
        """
        tf.keras.Model: Returns the complete autoencoder model.
        """
        return self._autoencoder

    def compile(self, optimizer, loss):
        """
        Compiles the autoencoder model with the given optimizer and loss function.

        Args:
            optimizer (str, tf.keras.optimizers.Optimizer): The optimizer to use.
            loss (str, tf.keras.losses.Loss): The loss function to use.
        """
        self._autoencoder.compile(optimizer=optimizer, loss=loss)

    def fit(self, data, epochs: int, callbacks=None):
        """
        Trains the autoencoder model on the given data.

        Args:
            data (tf.data.Dataset): The data to train on.
            epochs (int): The number of epochs to train for.
            callbacks (list[tf.keras.callbacks.Callback], optional): List of keras callbacks to apply during training.
        """
        self._autoencoder.fit(data, epochs=epochs, callbacks=callbacks)

    def predict(self, data):
        """
        Makes predictions with the autoencoder model on the given data.

        Args:
            data (tf.data.Dataset): The data to make predictions on.

        Returns:
            tf.data.Dataset: The predicted output.
        """
        return self._autoencoder.predict(data)

    def summary(self) -> None:
        """Printing combined model summary"""
        print("Encoder Summary")
        self._encoder.summary()
        print("\nDecoder Summary")
        self._decoder.summary()
        print("\nAutoencoder Summary")
        self._autoencoder.summary()


class AEKMeans(tf.keras.models.Model):
    def __init__(
        self,
        latent_dim,
        preprocessing_model,
        encoder,
        optimizer=None,
        nr_epochs: int = 10,
        cluster_range: int = 10,
        **kwargs,
    ):
        super().__init__()

        # params
        self.latent_dim = latent_dim
        self.cluster_range = cluster_range
        self.nr_epochs = nr_epochs
        # models
        self.encoder = encoder
        self.preprocessing_model = preprocessing_model
        self.optimizer = optimizer if optimizer is not None else tf.keras.optimizers.Adam(learning_rate=0.001)

    def _compute_distances(self, inputs):
        """
        Compute the distances between the inputs and clusters.

        Args:
            inputs (tf.Tensor): The input data, shape should be [batch_size, latent_dim].

        Returns:
            tf.Tensor: The distances, shape [batch_size, num_clusters].

        Example:
            ```python
            aekmeans = AEKMeans(latent_dim=16, nr_epochs=10)
            # Assume inputs is a tf.Tensor of shape [32, 16]
            distances = aekmeans._compute_distances(inputs)
            ```
        """
        try:
            if len(inputs.shape) != 2 or inputs.shape[1] != self.latent_dim:
                logger.error(f"Invalid input shape: {inputs.shape}, expected [batch_size, {self.latent_dim}]")
                return

            x_sq = tf.reduce_sum(tf.square(inputs), axis=1, keepdims=True)
            clusters_sq = tf.reduce_sum(tf.square(self.clusters), axis=1)

            # Note: The batch size is now the first dimension of 'inputs'
            cross_term = 2 * tf.matmul(inputs, tf.transpose(self.clusters))
            distances = tf.sqrt(x_sq + clusters_sq - cross_term)

            return distances

        except Exception as e:
            logger.error(f"Matrix multiplication failed: {e}")
            raise

    def compute_inertia(self, x) -> np.ndarray:
        """Compute and return the current inertia."""
        x = self.preprocessing_model.predict(x)
        X_transformed = self.encoder.predict(x)
        distances = self._compute_distances(X_transformed)
        min_distances = tf.reduce_min(distances, axis=1)
        inertia = tf.reduce_sum(min_distances)
        return inertia.numpy()

    def optimize_clusters(self, x, cluster_range: int) -> Tuple[int, List]:
        """
        Optimizes number of cluster using elbow method.

        Args:
            x (_type_): input dat to use for the optimization.
            cluster_range (int): Range of cluster in which to optimize.

        Returns:
            optimal_k (int): The optimal number of clusters.
            inertia_values (list): List of inertias for each cluster.
        """
        _cluster_range = range(1, cluster_range)
        inertia_values = []
        for k in tqdm(_cluster_range, total=cluster_range - 1):  # Start from 1 to avoid zero clusters
            self.clusters = self.add_weight(
                shape=(k, self.latent_dim),
                initializer="random_normal",
                trainable=True,
            )
            self.fit(x, epochs=self.nr_epochs, verbose=0)
            inertia = self.compute_inertia(x=x)
            inertia_values.append(inertia)
        # adding +1 to call actual index since python starts from 0
        optimal_k = _cluster_range[np.argmin(np.diff(inertia_values, 2)) + 1]
        logger.info(f"Optimal number: {optimal_k}, initializing cluster centroids")
        self.clusters = self.add_weight(
            name="centroids",
            shape=(optimal_k, self.latent_dim),
            initializer="random_normal",
            trainable=True,
        )
        return optimal_k, inertia_values

    def fit(self, x, *args, **kwargs):
        """
        Model fit method.

        Args:
            x (_type_): Input data to use for the model fit.
        """
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # encoding raw data
        x = self.preprocessing_model.predict(x)

        # Transform the data using the trained Encoder
        x_encoded = self.encoder.predict(x)

        # Create a new tf.data.Dataset with the transformed data
        dataset_transformed = tf.data.Dataset.from_tensor_slices(x_encoded).batch(CONF.train.TRAIN_BATCH_SIZE)

        # Train K-means on the transformed data
        for _ in tqdm(range(self.nr_epochs), total=self.nr_epochs):
            for X_batch in dataset_transformed:
                with tf.GradientTape() as tape:
                    distances = self._compute_distances(X_batch)
                    loss = tf.reduce_sum(tf.reduce_min(distances, axis=1))
                grads = tape.gradient(loss, [self.clusters])
                self.optimizer.apply_gradients(zip(grads, [self.clusters]))

    def predict(self, x):
        """
        Predict method for the model.

        Args:
            x (_type_): Input data to use for predictions

        Returns:
            clusters (np.ndarray): The predicted clusters.
        """
        # preprocessing inputs
        x = self.preprocessing_model.predict(x)

        # Transform the data using the trained Encoder
        X_transformed = self.encoder.predict(x)

        # Compute distances to cluster centroids
        distances = self._compute_distances(X_transformed)

        # Assign each point to the nearest cluster and return the indices
        return tf.argmin(distances, axis=1).numpy()

    def call(self, x):
        """
        Main modle entry method, same as predict.

        Args:
            x (_type_): Input data to use for predictions.

        Returns:
            clusters (np.ndarray): The predicted clusters.
        """
        return self.predict(x)

    def get_cluster_centers(self):
        """Return the current cluster centers."""
        return self.clusters.numpy()


class AEKMeansProd(tf.keras.models.Model):
    """
    Production ready anomaly detector model with preprocessing and all required training stats included.

    Args:
        preprocessing_model (tf.keras.models.Model): preprocessing model for tabular data.
        encoder (tf.keras.models.Model): encoder model to reduce dimensionality of the input data.
        clusters (tf.keras.layers.Layer): Configured clusters centers layer.
    """

    def __init__(
        self,
        preprocessing_model: tf.keras.models.Model,
        encoder: tf.keras.models.Model,
        clusters: tf.keras.layers.Layer,
    ) -> None:
        super().__init__()

        # models:
        self.preprocessing_model = preprocessing_model
        self.encoder = encoder
        self.clusters = self.add_weight(
            name="clusters",
            shape=clusters.shape,
            initializer=tf.keras.initializers.Constant(clusters.numpy()),
            trainable=True,
        )

        # Get the input shape from the preprocessing model
        self.input_shapes = self.preprocessing_model.input_shape

        # Define the inputs
        self.inputs = {name: tf.keras.Input(shape=shape[1:], name=name) for name, shape in self.input_shapes.items()}

    def get_config(self):
        """
        Serialization Method for the model. We are storing all the objects
        in the config so that they can be restored later.

        Returns:
            config (Dict): Config dictionary.
        """
        config = {
            "preprocessing_model": self.preprocessing_model,
            "encoder": self.encoder,
            "clusters": self.clusters.numpy(),  # Assuming clusters can be converted to numpy
        }
        return config

    @classmethod
    def from_config(cls, config):
        """
        Method to restore the model from a config dictionary.

        Args:
            config (Dict): Config dictionary.

        Returns:
            Model instance: Model with restored objects.
        """
        preprocessing_model = config.pop("preprocessing_model")
        encoder = config.pop("encoder")
        clusters = tf.constant(config.pop("clusters"))  # Convert numpy array back to tensor
        return cls(preprocessing_model, encoder, clusters)

    def _compute_distances(self, inputs):
        """
        Method to compute the distances between the inputs and clusters.

        Args:
            inputs: Input data

        Returns:
            distances (tf.Tensor): Distance matrix
        """
        x_sq = tf.reduce_sum(tf.square(inputs), axis=1, keepdims=True)
        clusters_sq = tf.reduce_sum(tf.square(self.clusters), axis=1)
        cross_term = 2 * tf.matmul(inputs, tf.transpose(self.clusters))
        distances = tf.sqrt(x_sq + clusters_sq - cross_term)
        return distances

    def call(self, x):
        """
        Model entrypoint

        Args:
            x (_type_): Input data to use for the model predictions.

        Returns:
            cluster indices (tf.Tensor): The predicted cluster indices.
        """
        # preprocessing
        x = self.preprocessing_model(x)

        # Transform the data using the trained Encoder
        encoded_data = self.encoder(x)

        # Compute distances to cluster centroids
        distances = self._compute_distances(encoded_data)

        # Assign each point to the nearest cluster and return the indices
        return tf.argmin(distances, axis=1)
