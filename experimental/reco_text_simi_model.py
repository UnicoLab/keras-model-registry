
# -*- coding: utf-8 -*-
# pypi/conda library
import tensorflow as tf
import tensorflow_recommenders as tfrs

# project plugin
from config import Config
from loguru import logger
from theparrot.tf import CallbackManager


class TextSimilarityModel:
    """A class used to represent a text similarity model using TFRS.

    Notebook: https://colab.research.google.com/drive/1KjFIxi-GoUZ9XyzeX2n1V6d_ktfSUxJ7#scrollTo=KpOdYDown4u5

    Attributes:
        embedding (tf.keras.Sequential): A sequence of layers for converting text to embeddings.
        query_model (tf.keras.Sequential): A sequence of layers for processing query text.
        candidate_model (tf.keras.Sequential): A sequence of layers for processing candidate text.
        similarity_model (tfrs.tasks.Retrieval): A TFRS retrieval model for similarity.
        index (tfrs.layers.factorized_top_k.ScaNN): A ScaNN index for retrieving top K similar items.

    """

    def __init__(self, text_vectorizer: object = None) -> None:
        logger.info("Initializing Text Similarity model ⚛️")
        self.CONF = Config()

        self.text_vectorizer = text_vectorizer or tf.keras.layers.TextVectorization(
            max_tokens=self.CONF.model.MAX_TOKENS,
            output_mode=self.CONF.model.VECTORIZER_OUTPUT_MODEL,
            output_sequence_length=self.CONF.model.EMBEDDING_DIM,
        )

    def build_models(self) -> None:
        """Create embedding, query and candidates model."""
        self.embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(
                    input_dim=self.CONF.model.MAX_TOKENS,
                    output_dim=self.CONF.model.EMBEDDING_DIM,
                ),
                tf.keras.layers.GlobalAveragePooling1D(),
            ]
        )

        self.query_model = tf.keras.Sequential(
            [
                self.text_vectorizer,
                self.embedding,
                tf.keras.layers.Dense(
                    units=self.CONF.model.QUERY_MODEL_DIM,
                    activation=self.CONF.model.QUERY_MODEL_ACTIVATION,
                ),
            ]
        )

        self.candidate_model = tf.keras.Sequential(
            [
                self.text_vectorizer,
                self.embedding,
                tf.keras.layers.Dense(
                    units=self.CONF.model.CANDIDATE_MODEL_DIM,
                    activation=self.CONF.model.CANDIDATE_MODEL_ACTIVATION,
                ),
            ]
        )

    def init_simi_model(self, raw_data):
        """Initialize the similarity model with the given raw_data.

        Args:
            raw_data (tf.data.Dataset): The raw dataset.

        """
        # defining data
        data_simi = raw_data.batch(self.CONF.model.SIMI_MODEL_BATCH_SIZE).map(self.candidate_model)

        # defining retrieval model
        self.similarity_model = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=data_simi,
            )
        )

    def build_index(self, raw_data):
        """Build the ScaNN index with the given raw_data.

        Args:
            raw_data (tf.data.Dataset): The raw dataset.

        """
        # building dataset
        data_index = raw_data.batch(self.CONF.model.SIMI_MODEL_BATCH_SIZE).map(lambda x: (x, self.candidate_model(x)))
        # building index
        self.index = tfrs.layers.factorized_top_k.ScaNN(
            query_model=self.query_model,
            k=self.CONF.model.TOP_K,
            distance_measure=self.CONF.model.DISTANCE_MEASURE,
            num_leaves=self.CONF.model.NR_LEAVES,  # this is the min data size for the index
            name="top_k_scann",
        )
        self.index.index_from_dataset(data_index)

    def compile(self):
        logger.info("Preparing custom optimizers: RADAM")
        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.CONF.model.LEARNING_RATE),
        )

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

    def compute_loss(self, features, training=False):
        """Compute the loss for the given features.

        Args:
            features (dict): The features as a dictionary.
            training (bool, optional): A flag indicating if the model is in training mode. Defaults to False.

        Returns:
            tf.Tensor: The computed loss.

        """
        query_embeddings = self.query_model(features["query"])
        positive_candidate_embeddings = self.candidate_model(features["candidate"])

        return self.similarity_model(query_embeddings, positive_candidate_embeddings)

    def call(self, queries):
        """Call the ScaNN index with the given queries and Get the top K similar items for the given query.

        Args:
            queries (Any): The input queries.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple containing the top K scores and top K candidates.

        """
        top_k_scores, top_k_candidates = self.index(queries)
        return top_k_scores, top_k_candidates

    def main(self, raw_data):
        """Prepare the model by initializing the similarity model and building the index.

        Args:
            raw_data (tf.data.Dataset): The raw dataset.

        Note:
            we first need to fit the text vectorizer.

        """
        self.build_models()
        self.init_simi_model(raw_data=raw_data)
        self.build_index(raw_data=raw_data)
        self.compile()
        return self
