# -*- coding: utf-8 -*-
# project plugin
from pathlib import Path

import tensorflow as tf

# project plugin
from config import Config
from loguru import logger
from tensorflow_decision_forests import tuner as tfdf_tuner
from tensorflow_decision_forests.keras import (
    FeatureSemantic,
    FeatureUsage,
    GradientBoostedTreesModel,
    Task,
)
from theparrot.tf.preprocessing import PreprocessModel
from theparrot.tools import log_method_calls


class MaxProbaIndexLayer(tf.keras.layers.Layer):
    """Custom Layer to post-process results from probability estimations into a given class directly."""

    def __init__(
        self,
        threshold: float,
        num_classes: int,
        label_classes: list[str],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.num_classes = num_classes
        self.label_classes = label_classes

    def call(self, scores):
        if self.num_classes == 2:
            return tf.where(scores <= self.threshold, 0, 1)
        else:
            return tf.argmax(scores, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "threshold": self.threshold,
                "num_classes": self.num_classes,
                "label_classes": self.label_classes,
            },
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Model:
    """Main class containing model architecture (only the architecture, no related fit nor predict !)."""

    def __init__(self):
        self.CONF = Config()

    @log_method_calls(
        start_msg="Preparing model features 📝",
        end_msg="Model Features Ready ✅",
    )
    def model_features(self) -> None:
        """Specify categorical and numerical data for model training.

        Returns:
            features (list(FeaturesUsage)): list and semantic of the input features of the model

        """
        logger.info("Preparing Features Definition")
        self.features = []
        for col_num in self.CONF.model.NUM_COLS:
            self.features.append(
                FeatureUsage(
                    name=col_num,
                    semantic=FeatureSemantic.NUMERICAL,
                ),
            )
            logger.info(f"Added {col_num} as numerical feature")
        for cat_col in self.CONF.model.CAT_COLS:
            self.features.append(
                FeatureUsage(
                    name=cat_col,
                    semantic=FeatureSemantic.CATEGORICAL,
                ),
            )
            logger.info(f"Added {cat_col} as categorical feature")
        logger.info("Features Definition Completed ✅")

    def get_csv_file_pattern(self, path) -> str:
        """Get the csv file pattern that will handle directories and file paths.

        Args:
            path (str): Path to the csv file (can be a directory or a file)

        Returns:
            str: File pattern that always has *.csv at the end

        """
        file_path = Path(path)
        # Check if the path is a directory
        if file_path.suffix:
            # Get the parent directory if the path is a file
            csv_pattern = file_path.parent
        else:
            csv_pattern = file_path
        logger.info(f"Correcting path from: {path} to: {csv_pattern}")
        return str(csv_pattern)

    @log_method_calls
    def train_preprocessing_model(self) -> None:
        """Training the feature space to preprocess the raw data."""
        logger.info("🏗️ Preparing training dataset using Preprocessing model")
        # we are skipping the label for the training
        _ppr_path = self.get_csv_file_pattern(
            path=self.CONF.paths.TRAIN__DOWNLOAD__SYNC__TRAIN_DATA,
        )
        logger.info(f"🔍 Looking for data in : {_ppr_path}")
        self.pm = PreprocessModel(
            path_data=_ppr_path,
            numeric_features=self.CONF.model.NUM_COLS,
            categorical_features=self.CONF.model.CAT_COLS,
            output_mode="dict",  # we want a dict as output for XGB
        )
        logger.info("Training preprocessing model 🔍")
        self.preprocessor_output = self.pm.build_preprocessor()
        self.preprocessing_model = self.preprocessor_output["model"]
        self.inputs_dim = self.preprocessor_output["output_dims"]
        self.signature = self.preprocessor_output["signature"]
        logger.info(f"Embeddings vector dims: {self.inputs_dim}")
        logger.info("preprocessing model trained ✅")

    @log_method_calls(
        start_msg="Building Post-Processing Model 🏋🏻",
        end_msg="Post-Processing Model Ready ✅",
    )
    def train_postprocessing_model(self) -> None:
        """Post-processing model predictions definition, to output a label instead of its probability or index.

        Note:
            We need to checl number classes since for binary classification we have single output (probability)
            while for multiclass we have a vector of outputs > 2.

        """
        _nr_classes = len(self.CONF.model.LABEL_CLASSES)

        # Replace the prediction_layer with the CustomPredictionLayer
        prediction_layer = MaxProbaIndexLayer(
            threshold=self.CONF.model.THRESHOLD,
            num_classes=_nr_classes,
            label_classes=self.CONF.model.LABEL_CLASSES,
        )

        # Define the input layer
        input_scores = tf.keras.layers.Input(shape=(None,), dtype=tf.float32)

        # Apply the custom prediction logic
        predictions = prediction_layer(input_scores)

        # Create the inverse lookup layer
        inverse_lookup = tf.keras.layers.StringLookup(
            vocabulary=self.CONF.model.LABEL_CLASSES,
            invert=True,
            # default is 1 at the beginning so we will have shifted results here
            num_oov_indices=0,
        )

        # Apply the inverse lookup to the predictions
        output_labels = inverse_lookup(predictions)

        # Build the model
        self.postprocess_model = tf.keras.models.Model(
            inputs=input_scores,
            outputs=output_labels,
        )

    @log_method_calls
    def prepare_tuner(self):
        """Prepare integrated model tuner for hyperparameter tuning.

        Note:
            Tuner is not compatible with multi-outputs models.

        """
        if self.CONF.model.TUNER_NR_TRIALS:
            logger.info("🪛 Preparing model tuner")
            tuner = tfdf_tuner.RandomSearch(num_trials=self.CONF.model.TUNER_NR_TRIALS)
            for key, value in self.CONF.model.TUNER_PARAMS.items():
                logger.info(f"Adding tuner parameter: {key} = {value}")
                tuner.choice(key, value)

            if self.CONF.model.TUNER_NR_TRIALS:
                logger.info(
                    f"(TUNER_NR_TRIALS > 0) -> Adding tuner to: {self.CONF.model.XGB_CONFIG}",
                )
                self.CONF.model.XGB_CONFIG["tuner"] = tuner
            logger.info("Tuner ready")
        else:
            logger.warning(
                "Tuner is not compatible with multi-outputs models (desactivating)",
            )

    @log_method_calls(
        start_msg="Initializing model 🤖",
        end_msg="Model Initialized ✅",
    )
    def init_model(self) -> None:
        """Initialize the model as a classification GradientBoostedTreesModel.

        Notes:
            In the initializaiton, we specify the list of features to use (strict)

        """
        self.model = GradientBoostedTreesModel(
            preprocessing=self.preprocessing_model,
            postprocessing=self.postprocess_model,
            task=Task.CLASSIFICATION,
            features=self.features,
            num_trees=self.CONF.model.NR_TREES,
            **self.CONF.model.XGB_CONFIG,
        )

    @log_method_calls(
        start_msg="Compiling model ⚙️",
        end_msg="Model Compiled ✅",
    )
    def compile_model(self) -> None:
        """Compile the model with classification metrics.

        Returns:
            tf.keras.Model: Keras model, ready to train

        """
        _metrics = [
            tf.keras.metrics.Accuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.TrueNegatives(name="true_negatives"),
            tf.keras.metrics.TruePositives(name="true_positives"),
        ]
        if len(self.CONF.model.LABEL_CLASSES) == 2:
            logger.info("adding Bi-Accuracy metric")
            _metrics.append(
                tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            )
        # compiling the model
        self.model.compile(
            metrics=_metrics,
        )

    @log_method_calls
    def export_model_signature(self) -> tf.function:
        """Define signature definition for model save."""

        @tf.function(input_signature=[self.signature])
        def serving_fn(inputs):
            """Serving function to serve prediction from imput data."""
            scores = self.model(inputs)
            return scores

        return serving_fn

    def main(self):
        """Main pipeline for the model initialization."""
        self.model_features()
        self.train_preprocessing_model()
        self.train_postprocessing_model()
        self.prepare_tuner()
        self.init_model()
        self.compile_model()
        return self.model
