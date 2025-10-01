"""Tests for the CategoricalAnomalyDetectionLayer."""

import unittest
import numpy as np
import tensorflow as tf
from keras import Model, Input
from kmr.layers.CategoricalAnomalyDetectionLayer import CategoricalAnomalyDetectionLayer


class TestCategoricalAnomalyDetectionLayer(unittest.TestCase):
    """Test cases for CategoricalAnomalyDetectionLayer."""

    def setUp(self) -> None:
        """Set up test cases."""
        self.batch_size = 32
        self.input_dim = 1
        self.valid_categories = ["red", "green", "blue"]

    def test_initialization(self) -> None:
        """Test layer initialization."""
        # Test string dtype
        layer = CategoricalAnomalyDetectionLayer(dtype="string")
        self.assertEqual(layer.dtype, "string")

        # Test int dtype
        layer = CategoricalAnomalyDetectionLayer(dtype="int")
        self.assertEqual(layer.dtype, "int")

        # Test invalid dtype
        with self.assertRaises(ValueError):
            CategoricalAnomalyDetectionLayer(dtype="invalid")

    def test_string_anomaly_detection(self) -> None:
        """Test anomaly detection with string inputs."""
        layer = CategoricalAnomalyDetectionLayer(dtype="string")
        layer.initialize_from_stats(self.valid_categories)

        # Test valid and invalid values
        inputs = tf.constant([["red"], ["purple"], ["green"]], dtype=tf.string)
        outputs = layer(inputs)

        # Check output shapes
        self.assertEqual(outputs["score"].shape, (3, 1))
        self.assertEqual(outputs["proba"].shape, (3, 1))
        self.assertEqual(outputs["anomaly"].shape, (3, 1))
        self.assertEqual(outputs["reason"].shape, (3, 1))

        # Check anomaly detection
        anomalies = outputs["anomaly"].numpy()
        self.assertFalse(anomalies[0][0])  # "red" is valid
        self.assertTrue(anomalies[1][0])   # "purple" is invalid
        self.assertFalse(anomalies[2][0])  # "green" is valid

    def test_int_anomaly_detection(self) -> None:
        """Test anomaly detection with integer inputs."""
        layer = CategoricalAnomalyDetectionLayer(dtype="int")
        layer.initialize_from_stats([1, 2, 3])

        # Test valid and invalid values
        inputs = tf.constant([[1], [4], [2]], dtype=tf.int32)
        outputs = layer(inputs)

        # Check anomaly detection
        anomalies = outputs["anomaly"].numpy()
        self.assertFalse(anomalies[0][0])  # 1 is valid
        self.assertTrue(anomalies[1][0])   # 4 is invalid
        self.assertFalse(anomalies[2][0])  # 2 is valid

    def test_empty_vocabulary(self) -> None:
        """Test behavior with empty vocabulary."""
        layer = CategoricalAnomalyDetectionLayer(dtype="string")
        layer.initialize_from_stats([])

        inputs = tf.constant([["any"]], dtype=tf.string)
        outputs = layer(inputs)

        # Everything should be anomalous with empty vocabulary
        self.assertTrue(outputs["anomaly"].numpy()[0][0])

    def test_serialization(self) -> None:
        """Test layer serialization."""
        # Create and save model with layer
        inputs = Input(shape=(1,), dtype=tf.string)
        layer = CategoricalAnomalyDetectionLayer(dtype="string")
        layer.initialize_from_stats(self.valid_categories)
        outputs = layer(inputs)
        model = Model(inputs=inputs, outputs=outputs)

        # Save and load model
        model_json = model.to_json()
        loaded_model = tf.keras.models.model_from_json(
            model_json,
            custom_objects={"CategoricalAnomalyDetectionLayer": CategoricalAnomalyDetectionLayer}
        )

        # Test loaded model
        test_input = tf.constant([["red"]], dtype=tf.string)
        original_output = model(test_input)
        loaded_output = loaded_model(test_input)

        # Compare outputs
        for key in original_output:
            if key in ["reason", "value"]:
                # Compare strings directly
                self.assertEqual(
                    original_output[key].numpy().tolist(),
                    loaded_output[key].numpy().tolist()
                )
            else:
                # Compare numerical values
                self.assertTrue(
                    np.allclose(
                        original_output[key].numpy(),
                        loaded_output[key].numpy()
                    )
                )


if __name__ == "__main__":
    unittest.main()
