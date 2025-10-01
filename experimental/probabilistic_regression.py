# -*- coding: utf-8 -*-
# pypi/conda library

from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp
from loguru import logger
from theparrot.tf import CallbackManager
from theparrot.tf.preprocessing import PreprocessModel

tfd = tfp.distributions

# project plugin
from config import Config


class NegLogLikelihood(tf.keras.losses.Loss):
    def __init__(self, name="negloglik", **kwargs):
        super(NegLogLikelihood, self).__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        """Compute the negative log-likelihood.

        Args:
            y_true (tf.Tensor): The true values.
            y_pred (tfp.distributions.Distribution): The predicted probability distribution.

        Returns:
            tf.Tensor: The computed negative log-likelihood loss.

        """
        return -y_pred.log_prob(y_true)

    def get_config(self):
        """Returns the configuration of the custom loss.

        Returns:
            dict: A Python dictionary containing the configuration of the loss function.

        """
        config = super(NegLogLikelihood, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        """Instantiates a `NegLogLikelihood` object from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            NegLogLikelihood: A new instance of `NegLogLikelihood`.

        """
        return cls(**config)


class Model:
    """Main class containing model architecture (only the architecture, no related fit nor predict !)."""

    def __init__(self):
        self.CONF = Config()

    def train_preprocessing_model(self) -> None:
        """Training the feature space to preprocess the raw data."""
        logger.info("🏗️ Preparing training dataset using Preprocessing model")
        # we are skipping the label for the training
        self.pm = PreprocessModel(
            path_data=self.CONF.paths.TRAIN__DOWNLOAD__SYNC__TRAIN_DATA,
            numeric_features=self.CONF.model.NUM_COLS,
            categorical_features=self.CONF.model.CAT_COLS,
            output_mode="concat",
        )
        logger.info("Training preprocessing model 🔍")
        self.preprocessor_output = self.pm.build_preprocessor()
        self.preprocessing_model = self.preprocessor_output["model"]
        self.inputs_dim = self.preprocessor_output["output_dims"]
        logger.info(f"Embeddings vector dims: {self.inputs_dim}")
        logger.info("preprocessing model trained ✅")

    @tf.function
    def negloglik(self, y, p_y):
        """Helper loss function which works well with NormalLog distribution."""
        y = tf.cast(y, tf.float32)
        return -tf.cast(p_y.log_prob(y), dtype=tf.float32)

    def init_model(self) -> tf.keras.models.Model:
        # get inputs form the preprocessor
        inputs: dict = self.pm.inputs
        logger.info(f"Defining inputs: {inputs}")

        # get the preprocessing layer
        logger.info("Setting up preprocessing layer...")
        input_data = self.preprocessing_model(inputs)

        # preparing first layer
        logger.info("Preparing layers...")
        nn_dims = self.CONF.model.NN_DIMS.split("-")

        logger.info("Preparing first layer...")
        first_layer_dim = nn_dims.pop(0)
        x = tf.keras.layers.Dense(
            units=first_layer_dim,
            activation=self.CONF.model.ACTIVATION,
            kernel_regularizer=self.CONF.model.REGULARIZATION,
            name=f"layer_{first_layer_dim}",
        )(input_data)
        x = tf.keras.layers.Dropout(
            self.CONF.model.DROPOUT,
            name=f"dropout_{first_layer_dim}",
        )(x)

        for nr_units in nn_dims:
            logger.info(f"Preparing layer... {nr_units}")
            x = tf.keras.layers.Dense(
                units=nr_units,
                activation=self.CONF.model.ACTIVATION,
                kernel_regularizer=self.CONF.model.REGULARIZATION,
                name=f"layer_{nr_units}",
            )(x)
            x = tf.keras.layers.Dropout(
                self.CONF.model.DROPOUT,
                name=f"dropout_{nr_units}",
            )(x)

        # preparing output layer
        logger.info("Preparing output layer...")
        output = tf.keras.layers.Dense(units=1 + 1, name="output")(x)

        # building LogNormal distribution as an output
        dist = tfp.layers.DistributionLambda(
            lambda t: tfd.LogNormal(
                loc=t[..., :1],
                scale=tf.math.softplus(0.05 * t[..., 1:]),
            ),
            dtype=tf.float32,
            name="dist",
        )(output)

        # defining inputs and output
        logger.info("Defining inputs and output...")
        self.model = tf.keras.models.Model(inputs=inputs, outputs=dist)
        logger.info("Model ready to be trained ✅")

    def compile_model(self) -> None:
        """Compaling model with appropiate loss and metrics."""
        logger.info("Compiling model")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.CONF.model.LEARNING_RATE,
            ),
            loss=NegLogLikelihood(),
            # loss="mean_squared_error",
            metrics=[
                tf.keras.metrics.MeanAbsolutePercentageError(),
                tf.keras.metrics.MeanAbsoluteError(),
                tf.keras.metrics.LogCoshError(name="logcosh", dtype=None),
            ],
        )
        logger.info("Model successfully compiled ✅")

    def save_model(self, path: str) -> None:
        """Saves the model to the given path.

        Args:
            path(str): Path to save the model.

        """
        self.model.save(path)
        logger.info(f"Model successfully saved to: {path} ✅")

    def load_model(self, path: str) -> tf.keras.models.Model:
        """Loads the model from the given path.

        Args:
            path(str): Path to load the model.

        Note:
            We need to load custom loss function when loading this model.

        """
        self.model = tf.keras.models.load_model(
            path,
            custom_objects={"NegLogLikelihood": NegLogLikelihood()},
        )
        logger.info(f"Model successfully loaded from: {path} ✅")
        return self.model

    def setup_callbacks(self):
        """Sets up the callbacks for keras model."""
        # settin gup callbacks
        manager = CallbackManager(
            model_name=self.CONF.model.MODEL_NAME,
            root_folder=self.CONF.paths.TRAIN__UPLOAD__SYNC__ARTIFACTS,
            USE_CHECKPOINT=self.CONF.callback.CALLBACK_USE_CHECKPOINT,
            USE_TENSORBOARD=self.CONF.callback.CALLBACK_USE_TENSORBOARD,
            USE_TENSORBOARD_CONFIG=self.CONF.callback.CALLBACK_USE_TENSORBOARD_CONFIG,
            USE_CSV_LOGGER=self.CONF.callback.CALLBACK_USE_CSV_LOGGER,
            USE_NAN_CHECK=self.CONF.callback.CALLBACK_USE_NAN_CHECK,
            USE_REDUCE_ON_PLATEAU=self.CONF.callback.CALLBACK_USE_REDUCE_ON_PLATEAU,
            REDUCE_ON_PLATEAU_CONFIG=self.CONF.callback.CALLBACK_REDUCE_ON_PLATEAU_CONFIG,
            USE_EARLY_STOPPING=self.CONF.callback.CALLBACK_USE_EARLY_STOPPING,
            EARLY_STOPPING_CONFIG=self.CONF.callback.CALLBACK_USE_EARLY_STOPPING_CONFIG,
        )
        callbacks_list = manager.setup_callbacks()
        logger.info("Model Callbacks OK")
        return callbacks_list

    def main(self):
        """Main pipeline for the model initialization."""
        self.train_preprocessing_model()
        self.init_model()
        self.compile_model()
        return self.model


class ModelTransform(tf.keras.Model):
    def __init__(self, model: tf.keras.models.Model):
        """Initializes the instance of the model with statistics extraction for the distributin layer.

        Args:
            model: tf.keras.models.Model: The model to be used for predictions.

        """
        super().__init__()
        self.model = model

    def call(self, inputs, training=False) -> dict[str, Any]:
        yhat = self.model(inputs)
        median = yhat.quantile(0.5)
        q25 = yhat.quantile(0.25)
        q75 = yhat.quantile(0.75)
        stddev = yhat.stddev()
        mean = yhat.mean()
        p_range = q75 - q25
        return {
            "mean": mean,
            "median": median,
            "stddev": stddev,
            "q25": q25,
            "q75": q75,
            "p_range": p_range,
        }
