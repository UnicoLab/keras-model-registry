"""Base test classes for model testing."""
import unittest
import tempfile
import keras
import tensorflow as tf
from pathlib import Path
import numpy as np
import pandas as pd

from loguru import logger


class FakeDataGenerator:
    """Class for generating fake CSV data based on feature specifications.

    This class maps each feature name to a feature type (represented as a string)
    and creates a synthetic dataset of the specified number of rows. It then returns
    the dataset as a CSV-formatted string.

    Example:
        ```python
        from fake_data_generator import FakeDataGenerator

        features = {
            "feat_float": "float",
            "feat_int": "integer_categorical",
            "feat_str_cat": "string_categorical",
            "feat_text": "text",
            "feat_date": "date",
        }

        generator = FakeDataGenerator()
        csv_data = generator.generate_csv(features_specs=features, nr_of_rows=5)
        print(csv_data)
        ```
    """

    def __init__(self):
        """
        Initializes the FakeDataGenerator. 
        
        Currently, no specific parameters are stored in the constructor.
        In future expansions, you can add parameters like random seeds, custom 
        distributions, or user-defined dictionaries of generation rules.
        """
        logger.debug("Initialized FakeDataGenerator.")

    def generate_csv(self, path: str, features_specs: dict[str, str], nr_of_rows: int = 100) -> str:
        """Generate a CSV string with synthetic data for the specified features.

        Args:
            features_specs (dict[str, str]): A dictionary mapping each feature name to a
                string describing the type of data to generate. Supported types include:
                'float', 'float_normalized', 'float_discretized', 'float_rescaled',
                'integer_categorical', 'string_categorical', 'text', and 'date'.
            nr_of_rows (int, optional): Number of rows in the generated dataset. Defaults to 10.

        Returns:
            str: A CSV-formatted string containing the synthetic dataset.

        Example:
            ```python
            features_specs = {
                "price": "float",
                "category_id": "integer_categorical",
                "animal_type": "string_categorical",
                "description": "text",
                "record_date": "date",
            }
            generator = FakeDataGenerator()
            csv_result = generator.generate_csv(features_specs, nr_of_rows=5)
            print(csv_result)
            ```
        """
        logger.debug("Starting CSV data generation.")
        df = self._generate_dataframe(features_specs, nr_of_rows)
        csv_data = df.to_csv(path_or_buf=path, index=False)
        logger.debug("CSV data generation completed.")
        return csv_data

    def _generate_dataframe(self, features_specs: dict[str, str], nr_of_rows: int) -> pd.DataFrame:
        """Generate a pandas DataFrame based on the given feature specifications.

        Args:
            features_specs (dict[str, str]): A dictionary mapping feature name to feature type.
            nr_of_rows (int): Number of rows to generate.

        Returns:
            pd.DataFrame: A dataframe containing synthetic data for the specified feature specs.
        """
        data = {}
        for feature_name, feature_type in features_specs.items():
            feature_type_lower = feature_type.lower().strip()
            logger.debug(
                f"Generating data for feature '{feature_name}' with type '{feature_type_lower}'."
            )

            if feature_type_lower in (
                "float",
                "float_normalized",
                "float_discretized",
                "float_rescaled",
            ):
                data[feature_name] = np.random.randn(nr_of_rows)
            elif feature_type_lower == "integer_categorical":
                data[feature_name] = np.random.randint(0, 5, size=nr_of_rows)
            elif feature_type_lower == "string_categorical":
                categories = ["cat", "dog", "fish", "bird"]
                data[feature_name] = np.random.choice(categories, size=nr_of_rows)
            elif feature_type_lower == "text":
                sentences = [
                    "Lorem ipsum dolor sit amet.",
                    "Consectetur adipiscing elit.",
                    "Vivamus eleifend augue quis velit.",
                    "Ut viverra neque at sem.",
                ]
                data[feature_name] = np.random.choice(sentences, size=nr_of_rows)
            elif feature_type_lower == "date":
                start_date = pd.Timestamp("2020-01-01")
                end_date = pd.Timestamp("2023-01-01")
                date_range = pd.date_range(start=start_date, end=end_date, freq="D")
                random_dates = np.random.choice(date_range, size=nr_of_rows)
                data[feature_name] = pd.Series(random_dates).dt.strftime("%Y-%m-%d")
            else:
                # Fallback for unsupported or custom types
                logger.warning(
                    f"Feature type '{feature_type_lower}' not recognized. "
                    "Generating random float data as fallback."
                )
                data[feature_name] = np.random.randn(nr_of_rows)

        return pd.DataFrame(data)


class BaseModelTest(unittest.TestCase):
    """Base test class for model testing."""

    @classmethod
    def setUpClass(cls):
        logger.info("🛎 Setting up test class")
        # create the temp file in setUp method if you want a fresh directory for each test.
        # This is useful if you don't want to share state between tests.
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_file = Path(cls.temp_dir.name)

        # prepare the PATH_LOCAL_TRAIN_DATA
        cls._path_data = Path("data/rawdata.csv")
        cls._path_data = cls.temp_file / cls._path_data
        cls._path_data.parent.mkdir(exist_ok=True, parents=True)

        # generating fake data
        _data = FakeDataGenerator()
        features_specs = {
            "feat1": "float",
            "feat2": "integer_categorical",
            "feat3": "string_categorical",
            "feat4": "text",
            "feat5": "date",
        }
        _data.generate_csv(
            path=cls._path_data,
            features_specs=features_specs,
            nr_of_rows=100,
        )

        # The start method is called on the patcher to apply the patch.
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        logger.info("🛎 Tearing down test class")
        super().tearDownClass()
        
        # Remove the temporary file after the test is done
        cls.temp_dir.cleanup()

    def compile_and_fit_model(
        self,
        model: tf.keras.Model,
        dataset: tf.data.Dataset,
        optimizer: str | tf.keras.optimizers.Optimizer = "adam",
        loss: str | tf.keras.losses.Loss = "mse",
        metrics: list[str | tf.keras.metrics.Metric] = None,
        epochs: int = 1,
    ) -> tf.keras.callbacks.History:
        """Compile and fit a Keras model on a given dataset.

        Args:
            model (tf.keras.Model): The model to compile and fit.
            dataset (tf.data.Dataset): A tf.data.Dataset containing (features, labels).
            optimizer (str | tf.keras.optimizers.Optimizer, optional): Keras optimizer. Defaults to "adam".
            loss (str | tf.keras.losses.Loss, optional): Loss function. Defaults to "mse".
            metrics (list[str | tf.keras.metrics.Metric], optional): List of metrics. Defaults to ["mae"].
            epochs (int, optional): Number of epochs to train. Defaults to 1.

        Returns:
            tf.keras.callbacks.History: The training history object.

        Example:
            ```python
            model = MyKerasModel(...)
            dataset = get_dataset(...)
            self.compile_and_fit_model(model, dataset, epochs=2)
            ```
        """
        if metrics is None:
            metrics = ["mae"]
        logger.debug("Compiling the model.")
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        logger.debug("Fitting the model.")
        history = model.fit(dataset, epochs=epochs, verbose=0)
        return history

    def save_and_load_model_keras(
        self,
        model: tf.keras.Model,
        dataset: tf.data.Dataset,
        model_filename: str = "model.keras",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Save a Keras model in the `.keras` format and load it to verify predictions.

        Args:
            model (tf.keras.Model): A trained Keras model.
            dataset (tf.data.Dataset): The dataset used to generate predictions.
            model_filename (str, optional): Filename for the saved model. Defaults to "model.keras".

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple of (original_predictions, loaded_predictions).

        Example:
            ```python
            model = MyKerasModel(...)
            self.compile_and_fit_model(model, dataset)
            orig_preds, loaded_preds = self.save_and_load_model_keras(model, dataset, "test_model.keras")
            np.testing.assert_allclose(orig_preds, loaded_preds, rtol=1e-5, atol=1e-5)
            ```
        """
        model_path = str(self.temp_file / model_filename)
        logger.debug(f"Saving the model to {model_path}.")
        model.save(model_path)
        
        logger.debug("Generating predictions from the original model.")
        original_predictions = model.predict(dataset, verbose=0)

        logger.debug("Loading the saved model.")
        loaded_model = tf.keras.models.load_model(model_path)

        logger.debug("Generating predictions from the loaded model.")
        loaded_predictions = loaded_model.predict(dataset, verbose=0)

        return original_predictions, loaded_predictions

    def export_and_load_tf_saved_model(
        self,
        model: tf.keras.Model,
        dataset: tf.data.Dataset,
        export_dir: str = "model_tf",
        serving_fn_name: str = "serve",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Export a model to TF SavedModel format and load it to verify predictions.

        Note:
            This method assumes your custom model has an `export` method
            that saves the model in TF SavedModel format with a `serve` 
            signature, or a built-in `tf.keras.Model.save` (if handling 
            signatures is done manually).

        Args:
            model (tf.keras.Model): A trained Keras model that supports exporting via `.export()`
                                    or `.save()` with a signature.
            dataset (tf.data.Dataset): A dataset used to generate predictions.
            export_dir (str, optional): The directory name for the saved model. Defaults to "model_tf".
            serving_fn_name (str, optional): The signature function name for serving. Defaults to "serve".

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple of (original_predictions, loaded_predictions).

        Example:
            ```python
            model = MyKerasModel(...)
            self.compile_and_fit_model(model, dataset)
            orig_preds, loaded_preds = self.export_and_load_tf_saved_model(model, dataset, "my_tf_export")
            np.testing.assert_allclose(orig_preds, loaded_preds, rtol=1e-5, atol=1e-5)
            ```
        """
        export_path = str(self.temp_file / export_dir)
        logger.debug(f"Exporting the model to {export_path}.")

        # If your model has a custom `export` method:
        # model.export(filepath=export_path, format="tf_saved_model")
        #
        # If using tf.keras built-in:
        # model.save(export_path, save_format="tf")  # For SavedModel format

        # For demonstration, let's assume the custom export method is used:
        model.export(filepath=export_path, format="tf_saved_model")

        logger.debug("Generating predictions from the original model.")
        original_predictions = model.predict(dataset, verbose=0)

        logger.debug("Loading the saved TF model via `tf.saved_model.load`.")
        reloaded_artifact = tf.saved_model.load(export_path)

        # If your model is expected to have a serving signature:
        # see if there's a function named 'serve' (by default) or a custom one
        serving_fn = reloaded_artifact.signatures.get(serving_fn_name, None)
        if serving_fn is None:
            raise ValueError(
                f"No serving function named '{serving_fn_name}' found in the loaded artifact."
            )

        # Convert dataset to a dictionary of Tensors for signatures
        # This depends on how your serve function expects inputs.
        inputs = list(dataset.take(1))[0][0]  # (features, labels)
        
        # You may need to adapt this to match your signature's input keys
        loaded_predictions = []
        for batch_features, _ in dataset:
            # Signature expects a dict of features
            result = serving_fn(**batch_features)
            # The returned dict might have a "predictions" or "output_0" key
            # depending on how you define your serving signature
            result_key = list(result.keys())[0]
            loaded_predictions.append(result[result_key].numpy())

        loaded_predictions = np.concatenate(loaded_predictions, axis=0)

        return original_predictions, loaded_predictions