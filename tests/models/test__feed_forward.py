"""Unit tests for feed forward model."""
import keras
from pathlib import Path
import tensorflow as tf
import numpy as np
from loguru import logger

from kmr.models.feed_forward import BaseFeedForwardModel
from ._base import BaseModelTest


class TestFeedForward(BaseModelTest):
    """Test cases for BaseFeedForwardModel."""

    def setUp(self) -> None:
        """Set up test case with sample data."""
        super().setUp()
        self.feature_names = ["feature1", "feature2", "feature3"]
        self.hidden_units = [64, 32]
        self.output_units = 1

        # Create sample dataset
        self.batch_size = 32
        self.num_samples = 100
        self.x = {
            name: tf.random.normal((self.num_samples, 1)) for name in self.feature_names
        }
        self.y = tf.random.uniform((self.num_samples, self.output_units))

        # Create dataset
        self.dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y)).batch(
            self.batch_size,
        )

    def test_model_creation(self) -> None:
        """Test if model can be created with various configurations."""
        logger.info("🧪 Testing model creation")

        # Test basic model creation
        model = BaseFeedForwardModel(
            feature_names=self.feature_names,
            hidden_units=self.hidden_units,
            output_units=self.output_units,
        )
        self.assertIsInstance(model, BaseFeedForwardModel)

        # Test model with dropout
        model_with_dropout = BaseFeedForwardModel(
            feature_names=self.feature_names,
            hidden_units=self.hidden_units,
            output_units=self.output_units,
            dropout_rate=0.5,
        )
        self.assertIsInstance(model_with_dropout, BaseFeedForwardModel)

    def test_model_compile_and_fit(self) -> None:
        """Test if model can be compiled and trained."""
        logger.info("🧪 Testing model compilation and training")

        model = BaseFeedForwardModel(
            feature_names=self.feature_names,
            hidden_units=self.hidden_units,
            output_units=self.output_units,
        )

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        # Train for 1 epoch
        history = model.fit(self.dataset, epochs=1, verbose=0)

        self.assertIn("loss", history.history)
        self.assertIn("mae", history.history)

    def test_model_predict(self) -> None:
        """Test if model can make predictions."""
        logger.info("🧪 Testing model predictions")

        model = BaseFeedForwardModel(
            feature_names=self.feature_names,
            hidden_units=self.hidden_units,
            output_units=self.output_units,
        )

        model.compile(optimizer="adam", loss="mse")

        # Make predictions
        predictions = model.predict(self.dataset, verbose=0)

        self.assertEqual(predictions.shape, (self.num_samples, self.output_units))

    def test_model_serialization(self) -> None:
        """Test model serialization and deserialization."""
        logger.info("🧪 Testing model serialization")

        original_model = BaseFeedForwardModel(
            feature_names=self.feature_names,
            hidden_units=self.hidden_units,
            output_units=self.output_units,
            dropout_rate=0.2,
        )

        # Get model config
        config = original_model.get_config()

        # Create new model from config
        restored_model = BaseFeedForwardModel.from_config(config)

        # Verify configurations match
        self.assertEqual(original_model.feature_names, restored_model.feature_names)
        self.assertEqual(original_model.hidden_units, restored_model.hidden_units)
        self.assertEqual(original_model.output_units, restored_model.output_units)
        self.assertEqual(original_model.dropout_rate, restored_model.dropout_rate)

    def test_model_save_and_load_keras(self) -> None:
        """Test saving and loading model."""
        logger.info("🧪 Testing model save and load (Keras format)")

        original_model = BaseFeedForwardModel(
            feature_names=self.feature_names,
            hidden_units=self.hidden_units,
            output_units=self.output_units,
        )

        # Compile and train the model
        original_model.compile(optimizer="adam", loss="mse")
        original_model.fit(self.dataset, epochs=1, verbose=0)

        # Get predictions from original model
        original_predictions = original_model.predict(self.dataset, verbose=0)

        # using base class temp directory
        model_path = str(Path(self.temp_file) / "_model.keras")

        # Save the model
        original_model.save(model_path)

        # Load the model
        loaded_model = keras.models.load_model(model_path)

        # Get predictions from loaded model
        loaded_predictions = loaded_model.predict(self.dataset, verbose=0)

        # Verify predictions are the same
        np.testing.assert_allclose(
            original_predictions,
            loaded_predictions,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_model_export_load_tf(self) -> None:
        """Test saving and loading model."""
        logger.info("🧪 Testing model save and load (TF format)")
        original_model_tf = BaseFeedForwardModel(
            feature_names=self.feature_names,
            hidden_units=self.hidden_units,
            output_units=self.output_units,
        )

        # Compile and train the model
        original_model_tf.compile(optimizer="adam", loss="mse")
        original_model_tf.fit(self.dataset, epochs=1, verbose=0)

        # Get predictions from original model
        original_predictions = original_model_tf.predict(self.dataset, verbose=0)

        # using base class temp direftory
        model_path = str(Path(self.temp_file, "model_tf"))

        # exporting to tf format
        original_model_tf.export(
            filepath=model_path,
            format="tf_saved_model",
        )

        # Load the serve function in a different process/environment
        reloaded_artifact = tf.saved_model.load(model_path)

        # loagin serving function
        serving_fn = reloaded_artifact.signatures["serving_default"]

        # serving the model# 1) Strip out labels and structure the features
        features_only_ds = self.dataset.map(
            lambda x, y: tuple(x[feature_name] for feature_name in self.feature_names),
        )

        # 2) Call the signature, matching the expected argument names
        loaded_predictions = []
        for x1_batch, x2_batch, x3_batch in features_only_ds:
            # 3) Call the signature, matching the expected argument names
            output_dict = serving_fn(
                args_0=x1_batch,
                args_0_1=x2_batch,
                args_0_2=x3_batch,
            )
            loaded_predictions.append(output_dict["output_0"].numpy())

        # 3) Get the predictions
        loaded_predictions = np.concatenate(loaded_predictions, axis=0)

        # Verify predictions are the same
        np.testing.assert_allclose(
            original_predictions,
            loaded_predictions,
            rtol=1e-5,
            atol=1e-5,
        )
