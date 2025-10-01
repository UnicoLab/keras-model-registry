# standard library

# pypi/conda library
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from loguru import logger
from theparrot.tf import CallbackManager

# project plugin
from config import Config

tfd = tfp.distributions


class GatedLinearUnit(tf.keras.layers.Layer):
    """
    GatedLinearUnit is a custom Keras layer that implements a gated linear unit,
    which is a type of neural network activation function that selectively filters input data using a sigmoid gate.
    This layer applies a dense linear transformation to the input tensor and multiplies the result with the output of a dense sigmoid transformation.
    The result is a tensor where the input data is filtered based on the learned weights and biases of the layer.

    Args:
        units (int): Positive integer, dimensionality of the output space.

    Returns:
        Tensor: Output tensor of the GatedLinearUnit layer.

    Example:
        ```python
        gl = GatedLinearUnit(64)
        x = tf.random.normal((32, 100))
        y = gl(x)
        ```
    """

    def __init__(self, units: int):
        super().__init__()
        self.linear = tf.keras.layers.Dense(units)
        self.sigmoid = tf.keras.layers.Dense(units, activation="sigmoid")

    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)


class GatedResidualNetwork(tf.keras.layers.Layer):
    """
    GatedResidualNetwork is a custom Keras layer that implements a gated residual network, which is a type of neural network architecture that uses residual connections to enable training of very deep networks.
    This layer applies a series of transformations to the input tensor and combines it with the original input using a residual connection.
    The transformations include a dense layer with ELU activation, a dense linear layer, a dropout layer, a gated linear unit layer, layer normalization, and a final dense layer.

    Args:
        units (int): Positive integer, dimensionality of the output space.
        dropout_rate (float): Float between 0 and 1. Fraction of the input units to drop.

    Returns:
        Tensor: Output tensor of the GatedResidualNetwork layer.

    Example:
        ```python
        grn = GatedResidualNetwork(units=64, dropout_rate=0.2)
        x = tf.random.normal((32, 100))
        y = grn(x)
        ```
    """

    def __init__(self, units: int, dropout_rate: float = 0.2):
        super().__init__()
        self.units = units
        self.elu_dense = tf.keras.layers.Dense(units, activation="elu")
        self.linear_dense = tf.keras.layers.Dense(units)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.gated_linear_unit = GatedLinearUnit(units=units)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.project = tf.keras.layers.Dense(units)

    def call(self, inputs):
        x = self.elu_dense(inputs)
        x = self.linear_dense(x)
        x = self.dropout(x)
        if inputs.shape[-1] != self.units:
            inputs = self.project(inputs)
        x = inputs + self.gated_linear_unit(x)
        x = self.layer_norm(x)
        return x


class VariableSelection(tf.keras.layers.Layer):
    """
    VariableSelection is a custom Keras layer that implements a variable selection mechanism for multi-modal inputs,
    which is a type of neural network architecture that selectively weighs the contribution of each input feature based on learned weights.
    This layer applies a gated residual network to each feature independently and concatenates the results.
    It then applies another gated residual network to the concatenated tensor and uses a softmax layer to calculate the weights for each feature.
    Finally, it combines the weighted features using matrix multiplication.

    Args:
        nr_features (int): Positive integer, number of input features.
        units (int): Positive integer, dimensionality of the output space.
        dropout_rate (float): Float between 0 and 1. Fraction of the input units to drop.

    Returns:
        Tensor: Output tensor of the VariableSelection layer.

    Example:
        ```python
        vs = VariableSelection(nr_features=3, units=64, dropout_rate=0.2)
        x1 = tf.random.normal((32, 100))
        x2 = tf.random.normal((32, 200))
        x3 = tf.random.normal((32, 300))
        y = vs([x1, x2, x3])
        ```
    """

    def __init__(self, nr_features: int, units: int, dropout_rate: float = 0.2):
        super().__init__()
        self.grns = list()
        # Create a GRN for each feature independently
        for _ in range(nr_features):
            grn = GatedResidualNetwork(units=units, dropout_rate=dropout_rate)
            self.grns.append(grn)
        # Create a GRN for the concatenation of all the features
        self.grn_concat = GatedResidualNetwork(units=units, dropout_rate=dropout_rate)
        self.softmax = tf.keras.layers.Dense(units=nr_features, activation="softmax")

    def call(self, inputs):
        v = tf.keras.layers.concatenate(inputs)
        v = self.grn_concat(v)
        v = tf.expand_dims(self.softmax(v), axis=-1)

        x = []
        for idx, input in enumerate(inputs):
            x.append(self.grns[idx](input))
        x = tf.stack(x, axis=1)

        outputs = tf.squeeze(tf.matmul(v, x, transpose_a=True), axis=1)
        return outputs


class GRVSNModel:
    """Gated Recursice Variable Selection Network"""

    def __init__(self) -> None:
        # loading configurations
        self.CONF = Config()
        # initializing preprocessors for features
        self.init_feature_space()

    @tf.function
    def negloglik(self, y, p_y) -> float:
        """Helper loss function which works well with NormalLog distribution."""
        y = tf.cast(y, tf.float32)
        return -tf.cast(p_y.log_prob(y), dtype=tf.float32)

    def init_feature_space(self) -> None:
        """Setting up the feature space for the data"""
        logger.info("Initializing Feature Space")
        self.feature_space = tf.keras.utils.FeatureSpace(
            features=self.CONF.fsc.FEATURE_SPACE_CONFIG,
            crosses=self.CONF.fsc.FEATURE_SPACE_CROSSES,
            crossing_dim=self.CONF.fsc.FEATURE_SPACE_CROSSING_DIMENSION,
            output_mode=self.CONF.fsc.FEATURE_SPACE_OUTPUT_MODE,
        )
        logger.info("Feature Space successfully initialized ✅")

    def create_model(self) -> tf.keras.models.Model:
        """Creating the model"""
        # define inputs
        logger.info("Initializing model preprocessing layer 🏗️")
        _dict_inputs = self.feature_space.get_inputs()
        _encoded_features = self.feature_space.get_encoded_features()
        nr_features = len(_encoded_features)
        logger.info(f"Number of features: {nr_features}")

        # define backbone
        logger.info("Initializing model backbone layer 🥑")
        features = VariableSelection(
            nr_features=nr_features,
            units=self.CONF.model.MODEL_ENCODING_SIZE,
            dropout_rate=self.CONF.model.MODEL_DROPOUT,
        )(_encoded_features.values())

        # defining model MLP backend
        _backend_dims = self.CONF.model.MODEL_BACKEND_DIMENSIONS
        if _backend_dims:
            logger.info("Initializing model backend layer 🧠")
            for idx, dim in enumerate(_backend_dims.split("-")):
                logger.info(f"Adding layer of: {dim} neurons ☊")
                features = tf.keras.layers.Dense(
                    units=dim,
                    activation=self.CONF.model.MODEL_BACKEND_ACTIVATION,
                    name=f"mlp_{idx}",
                )(features)

        # define outputs
        logger.info("Defining model outputs 🐙")
        # classification
        logger.info(f"Initializing {self.CONF.model.MODEL_MODE} output")
        if self.CONF.model.MODEL_MODE == self.CONF.mmo.DISTRIBUTION:
            # setting up distribution parameters
            outputs = tf.keras.layers.Dense(1 + 1, name="output")(features)
            # distribution outputs
            outputs = tfp.layers.DistributionLambda(
                lambda t: tfd.LogNormal(
                    loc=t[..., :1],
                    scale=tf.math.softplus(0.05 * t[..., 1:]),
                ),
                dtype=tf.float32,
            )(outputs)
            logger.info("LogNormal Distribution ready to be used !")
        elif self.CONF.model.MODEL_MODE == self.CONF.mmo.CLASSIFICATION:
            outputs = tf.keras.layers.Dense(1, name="output", activation="sigmoid")(
                features,
            )
        else:
            outputs = tf.keras.layers.Dense(units=1, name="output")(features)
        # defining model
        logger.info("Defining model ⚛️")
        self.model = tf.keras.Model(inputs=_dict_inputs, outputs=outputs)

        logger.info("Model successfully initialized ✅")
        return self.model

    def define_loss(self) -> tf.keras.losses.Loss:
        """Defining loss function"""
        _mode = self.CONF.model.MODEL_MODE
        logger.info(f"Defining model loss for: {_mode}")
        if _mode == self.CONF.mmo.CLASSIFICATION:
            self.loss = tf.keras.losses.BinaryCrossentropy()
        elif _mode == self.CONF.mmo.DISTRIBUTION:
            self.loss = self.negloglik
        else:
            self.loss = tf.keras.losses.MeanSquaredError()

        logger.info(f"Model loss successfully defined: {self.loss} ✅")
        return self.loss

    def define_metrics(self) -> list[tf.keras.metrics.Metric]:
        """Define corresponding model metrics"""
        _mode = self.CONF.model.MODEL_MODE
        logger.info(f"Defining model metrics for: {_mode}")
        if _mode == self.CONF.mmo.CLASSIFICATION:
            self.metrics = [
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.FalsePositives(name="fp"),
                tf.keras.metrics.FalseNegatives(name="fn"),
                tf.keras.metrics.TruePositives(name="tp"),
            ]
        else:
            self.metrics = [
                tf.keras.metrics.MeanAbsoluteError(name="mae"),
                tf.keras.metrics.MeanSquaredError(name="mse"),
                tf.keras.metrics.RootMeanSquaredError(name="rmse"),
                tf.keras.metrics.MeanAbsolutePercentageError(name="mape"),
                tf.keras.metrics.MeanSquaredLogarithmicError(name="msle"),
            ]
        logger.info(f"Model metrics successfully defined: {self.metrics} ✅")
        return self.metrics

    def compile_model(self) -> None:
        """Compaling model with appropiate loss and metrics"""
        logger.info("Compiling model")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.CONF.model.LEARNING_RATE,
            ),
            loss=self.define_loss(),
            metrics=self.define_metrics(),
        )
        logger.info("Model successfully compiled ✅")

    def setup_callbacks(self):
        """
        Sets up the callbacks for keras model.
        """
        # settin gup callbacks
        manager = CallbackManager(
            model_name=self.CONF.model.MODEL_NAME,
            root_folder=self.CONF.data.TRAIN__UPLOAD__SYNC__ARTIFACTS,
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

    def setup_model(self) -> tf.keras.models.Model:
        logger.info("Setting up model")
        self.create_model()
        self.compile_model()
        logger.info("Model redy to be trained ✅")
        return self.model

    def save_model(self, path: str) -> None:
        """
        Saves the model to the given path.

        Args:
            path(str): Path to save the model.
        """
        logger.info(f"Saving model to: {path}")
        self.model.save(path)
        logger.info("Model successfully saved ✅")

    def load_model(self, path: str) -> tf.keras.models.Model:
        """
        Loads the model from the given path.

        Args:
            path(str): Path to load the model.
        """
        _mode = self.CONF.model.MODEL_MODE
        logger.info(f"Loading model for: {_mode}")
        logger.info(f"Loading model from: {path}")
        if _mode == self.CONF.mmo.CLASSIFICATION or _mode == self.CONF.mmo.REGRESSION:
            self.model = tf.keras.models.load_model(path)
        else:
            logger.info("Loading model with custom loss")
            self.model = tf.keras.models.load_model(
                path,
                custom_objects={
                    "negloglik": self.negloglik,
                },
            )
        logger.info("Model successfully loaded ✅")
        return self.model

    def predict_on_batch(self, data, model, batch_size: int = 1000) -> pd.DataFrame:
        """
        Maging predictions using given model for a given data by batches.

        Args:
            data (tf.dataset.Dataset) the raw data to be predicted.
            model (tf.keras.models.Model): model to be used for predictions.
            batch_size (int): batch size to be used for predictions

        Returns:
            data_batch_out (pd.DataFrame): dataframe containing predictions of the model.
        """
        logger.info(f"Starting batch prediction by {batch_size}")
        data_batch_out = pd.DataFrame()
        if self.CONF.model.MODEL_MODE == self.CONF.mmo.DISTRIBUTION:
            # fetching distributions for the batch
            yhat = model(data)
            logger.debug("yhat: ", yhat)
            # fetching data form the distribution
            data_batch_out["mean"] = yhat.mean().numpy().ravel()
            data_batch_out["q25"] = yhat.quantile(0.25).numpy().ravel()
            data_batch_out["median"] = yhat.quantile(0.5).numpy().ravel()
            data_batch_out["q75"] = yhat.quantile(0.75).numpy().ravel()
            data_batch_out["stddev"] = yhat.stddev().numpy().ravel()
            logger.debug("data_batch_out: ", data_batch_out)
        else:
            y_hat = model.predict(data)
            data_batch_out["y_hat"] = y_hat.numpy().ravel()
            # adding data
        return data_batch_out
