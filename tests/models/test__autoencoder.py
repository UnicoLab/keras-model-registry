"""Unit tests for autoencoder models."""
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from loguru import logger

from kmr.models.autoencoder import Autoencoder
from ._base import BaseModelTest


class TestAutoencoder(BaseModelTest):
    """Test cases for Autoencoder model."""

    def setUp(self) -> None:
        """Set up test case with sample data."""
        super().setUp()
        self.input_dim = 100
        self.encoding_dim = 32
        self.intermediate_dim = 64
        self.threshold = 2.0

        # Create sample dataset
        self.batch_size = 32
        self.num_samples = 100
        self.x = tf.random.normal((self.num_samples, self.input_dim))
        self.y = self.x  # Autoencoder target is the same as input

        # Create dataset
        self.dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y)).batch(
            self.batch_size,
        )

    def test_model_creation(self) -> None:
        """Test if model can be created with various configurations."""
        logger.info("🧪 Testing Autoencoder model creation")

        # Test basic model creation
        model = Autoencoder(
            input_dim=self.input_dim,
            encoding_dim=self.encoding_dim,
            intermediate_dim=self.intermediate_dim,
        )
        self.assertIsInstance(model, Autoencoder)
        self.assertEqual(model.input_dim, self.input_dim)
        self.assertEqual(model.encoding_dim, self.encoding_dim)
        self.assertEqual(model.intermediate_dim, self.intermediate_dim)

        # Test model with custom threshold
        model_with_threshold = Autoencoder(
            input_dim=self.input_dim,
            encoding_dim=self.encoding_dim,
            intermediate_dim=self.intermediate_dim,
            threshold=3.0,
        )
        self.assertEqual(model_with_threshold.threshold, 3.0)

    def test_model_creation_invalid_params(self) -> None:
        """Test model creation with invalid parameters."""
        logger.info("🧪 Testing Autoencoder invalid parameters")

        # Test negative input_dim
        with self.assertRaises(ValueError):
            Autoencoder(input_dim=-1, encoding_dim=self.encoding_dim)

        # Test negative encoding_dim
        with self.assertRaises(ValueError):
            Autoencoder(input_dim=self.input_dim, encoding_dim=-1)

        # Test negative intermediate_dim
        with self.assertRaises(ValueError):
            Autoencoder(
                input_dim=self.input_dim,
                encoding_dim=self.encoding_dim,
                intermediate_dim=-1,
            )

        # Test negative threshold
        with self.assertRaises(ValueError):
            Autoencoder(
                input_dim=self.input_dim,
                encoding_dim=self.encoding_dim,
                threshold=-1.0,
            )

    def test_model_compile_and_fit(self) -> None:
        """Test if model can be compiled and trained."""
        logger.info("🧪 Testing Autoencoder compilation and training")

        model = Autoencoder(
            input_dim=self.input_dim,
            encoding_dim=self.encoding_dim,
            intermediate_dim=self.intermediate_dim,
        )

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        # Train for 1 epoch
        history = model.fit(self.dataset, epochs=1, verbose=0)

        self.assertIn("loss", history.history)
        self.assertIn("mae", history.history)

    def test_model_predict(self) -> None:
        """Test if model can make predictions."""
        logger.info("🧪 Testing Autoencoder predictions")

        model = Autoencoder(
            input_dim=self.input_dim,
            encoding_dim=self.encoding_dim,
            intermediate_dim=self.intermediate_dim,
        )

        model.compile(optimizer="adam", loss="mse")

        # Make predictions
        predictions = model.predict(self.dataset, verbose=0)

        self.assertEqual(predictions.shape, (self.num_samples, self.input_dim))

    def test_model_setup_threshold(self) -> None:
        """Test threshold setup functionality."""
        logger.info("🧪 Testing Autoencoder threshold setup")

        model = Autoencoder(
            input_dim=self.input_dim,
            encoding_dim=self.encoding_dim,
            intermediate_dim=self.intermediate_dim,
        )

        model.compile(optimizer="adam", loss="mse")
        model.fit(self.dataset, epochs=1, verbose=0, auto_setup_threshold=False)

        # Setup threshold
        model.setup_threshold(self.x)

        # Check that threshold variables are set
        self.assertGreater(model.median, 0)
        self.assertGreater(model.std, 0)

    def test_model_auto_configure_threshold(self) -> None:
        """Test auto threshold configuration functionality."""
        logger.info("🧪 Testing Autoencoder auto threshold configuration")

        model = Autoencoder(
            input_dim=self.input_dim,
            encoding_dim=self.encoding_dim,
            intermediate_dim=self.intermediate_dim,
        )

        model.compile(optimizer="adam", loss="mse")
        model.fit(self.dataset, epochs=1, verbose=0, auto_setup_threshold=False)

        # Test different threshold methods
        for method in ["iqr", "percentile", "zscore"]:
            model.auto_configure_threshold(self.x, method=method)
            self.assertGreater(model.threshold, 0)
            self.assertGreater(model.median, 0)
            self.assertGreater(model.std, 0)

    def test_model_fit_with_auto_threshold(self) -> None:
        """Test fit method with automatic threshold setup."""
        logger.info("🧪 Testing Autoencoder fit with auto threshold")

        model = Autoencoder(
            input_dim=self.input_dim,
            encoding_dim=self.encoding_dim,
            intermediate_dim=self.intermediate_dim,
        )

        model.compile(optimizer="adam", loss="mse")

        # Fit with auto threshold setup
        history = model.fit(
            self.dataset,
            epochs=1,
            verbose=0,
            auto_setup_threshold=True,
        )

        # Check that threshold was automatically set
        self.assertGreater(model.median, 0)
        self.assertGreater(model.std, 0)
        self.assertIn("loss", history.history)

    def test_model_anomaly_detection(self) -> None:
        """Test anomaly detection functionality."""
        logger.info("🧪 Testing Autoencoder anomaly detection")

        model = Autoencoder(
            input_dim=self.input_dim,
            encoding_dim=self.encoding_dim,
            intermediate_dim=self.intermediate_dim,
        )

        model.compile(optimizer="adam", loss="mse")
        model.fit(self.dataset, epochs=1, verbose=0, auto_setup_threshold=False)
        model.setup_threshold(self.x)

        # Test anomaly detection
        results = model.is_anomaly(self.x)

        self.assertIn("score", results)
        self.assertIn("anomaly", results)
        self.assertIn("std", results)
        self.assertIn("threshold", results)
        self.assertIn("median", results)

        # Check shapes
        self.assertEqual(len(results["score"]), self.num_samples)
        self.assertEqual(len(results["anomaly"]), self.num_samples)

    def test_model_anomaly_detection_with_dataset(self) -> None:
        """Test anomaly detection with dataset input."""
        logger.info("🧪 Testing Autoencoder anomaly detection with dataset")

        model = Autoencoder(
            input_dim=self.input_dim,
            encoding_dim=self.encoding_dim,
            intermediate_dim=self.intermediate_dim,
        )

        model.compile(optimizer="adam", loss="mse")
        model.fit(self.dataset, epochs=1, verbose=0, auto_setup_threshold=False)
        model.setup_threshold(self.x)

        # Test anomaly detection with dataset
        results = model.is_anomaly(self.dataset)

        self.assertIn("score", results)
        self.assertIn("anomaly", results)
        self.assertIn("std", results)
        self.assertIn("threshold", results)
        self.assertIn("median", results)

        # Check shapes
        self.assertEqual(len(results["score"]), self.num_samples)
        self.assertEqual(len(results["anomaly"]), self.num_samples)

    def test_model_with_preprocessing(self) -> None:
        """Test autoencoder with preprocessing model integration."""
        logger.info("🧪 Testing Autoencoder with preprocessing model")

        # Create a custom preprocessing model that can handle multiple inputs
        class MultiInputPreprocessingModel(keras.Model):
            def __init__(self):
                super().__init__()
                self.dense1 = keras.layers.Dense(16, activation="relu")
                self.dense2 = keras.layers.Dense(16, activation="relu")
                self.dense3 = keras.layers.Dense(16, activation="relu")
                self.concat = keras.layers.Concatenate()
                self.final_dense = keras.layers.Dense(32, activation="relu")
                self.dropout = keras.layers.Dropout(0.1)

            def call(self, inputs) -> tf.Tensor:
                # Process each input separately
                feat1 = self.dense1(inputs["feature1"])
                feat2 = self.dense2(inputs["feature2"])
                feat3 = self.dense3(inputs["feature3"])

                # Concatenate and final processing
                combined = self.concat([feat1, feat2, feat3])
                output = self.final_dense(combined)
                output = self.dropout(output)
                return output

        preprocessing_model = MultiInputPreprocessingModel()

        # Create autoencoder with preprocessing
        model = Autoencoder(
            input_dim=32,  # Output size of preprocessing model
            encoding_dim=self.encoding_dim,
            intermediate_dim=self.intermediate_dim,
            preprocessing_model=preprocessing_model,
            inputs={"feature1": (10,), "feature2": (15,), "feature3": (25,)},
        )

        # Test with dictionary inputs
        test_inputs = {
            "feature1": tf.random.normal((10, 10)),
            "feature2": tf.random.normal((10, 15)),
            "feature3": tf.random.normal((10, 25)),
        }

        # Test forward pass
        results = model(test_inputs)

        self.assertIn("reconstruction", results)
        self.assertIn("score", results)
        self.assertIn("anomaly", results)
        self.assertIn("median", results)
        self.assertIn("std", results)
        self.assertIn("threshold", results)

        # Test anomaly detection
        anomaly_results = model.is_anomaly(test_inputs)
        self.assertIn("score", anomaly_results)
        self.assertIn("anomaly", anomaly_results)

    def test_model_predict_anomaly_scores(self) -> None:
        """Test anomaly score prediction."""
        logger.info("🧪 Testing Autoencoder anomaly score prediction")

        model = Autoencoder(
            input_dim=self.input_dim,
            encoding_dim=self.encoding_dim,
            intermediate_dim=self.intermediate_dim,
        )

        model.compile(optimizer="adam", loss="mse")
        model.fit(self.dataset, epochs=1, verbose=0)

        # Predict anomaly scores
        scores = model.predict_anomaly_scores(self.x)

        self.assertEqual(len(scores), self.num_samples)
        self.assertTrue(np.all(scores >= 0))  # Scores should be non-negative

    def test_model_serialization(self) -> None:
        """Test model serialization and deserialization."""
        logger.info("🧪 Testing Autoencoder serialization")

        original_model = Autoencoder(
            input_dim=self.input_dim,
            encoding_dim=self.encoding_dim,
            intermediate_dim=self.intermediate_dim,
            threshold=2.5,
        )

        # Get model config
        config = original_model.get_config()

        # Create new model from config
        restored_model = Autoencoder.from_config(config)

        # Verify configurations match
        self.assertEqual(original_model.input_dim, restored_model.input_dim)
        self.assertEqual(original_model.encoding_dim, restored_model.encoding_dim)
        self.assertEqual(
            original_model.intermediate_dim,
            restored_model.intermediate_dim,
        )
        self.assertEqual(original_model.threshold, restored_model.threshold)

    def test_model_save_and_load_keras(self) -> None:
        """Test saving and loading model."""
        logger.info("🧪 Testing Autoencoder save and load (Keras format)")

        original_model = Autoencoder(
            input_dim=self.input_dim,
            encoding_dim=self.encoding_dim,
            intermediate_dim=self.intermediate_dim,
        )

        # Compile and train the model
        original_model.compile(optimizer="adam", loss="mse")
        original_model.fit(self.dataset, epochs=1, verbose=0)

        # Get predictions from original model
        original_predictions = original_model.predict(self.dataset, verbose=0)

        # using base class temp directory
        model_path = str(Path(self.temp_file) / "autoencoder_model.keras")

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
        logger.info("🧪 Testing Autoencoder save and load (TF format)")
        original_model_tf = Autoencoder(
            input_dim=self.input_dim,
            encoding_dim=self.encoding_dim,
            intermediate_dim=self.intermediate_dim,
        )

        # Compile and train the model
        original_model_tf.compile(optimizer="adam", loss="mse")
        original_model_tf.fit(self.dataset, epochs=1, verbose=0)

        # Get predictions from original model
        original_predictions = original_model_tf.predict(self.dataset, verbose=0)

        # using base class temp directory
        model_path = str(Path(self.temp_file, "autoencoder_model_tf"))

        # exporting to tf format
        original_model_tf.export(
            filepath=model_path,
            format="tf_saved_model",
        )

        # Load the serve function in a different process/environment
        reloaded_artifact = tf.saved_model.load(model_path)

        # loading serving function
        serving_fn = reloaded_artifact.signatures["serving_default"]

        # serving the model
        loaded_predictions = []
        for batch_x, _ in self.dataset:
            # Call the signature - use positional argument as expected by the signature
            output_dict = serving_fn(batch_x)
            loaded_predictions.append(output_dict["output_0"].numpy())

        # Get the predictions
        loaded_predictions = np.concatenate(loaded_predictions, axis=0)

        # Verify predictions are the same
        np.testing.assert_allclose(
            original_predictions,
            loaded_predictions,
            rtol=1e-5,
            atol=1e-5,
        )
