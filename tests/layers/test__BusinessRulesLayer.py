"""Tests for the BusinessRulesLayer."""

import unittest
import numpy as np
import tensorflow as tf
from keras import Model, Input
from kmr.layers import BusinessRulesLayer


class TestBusinessRulesLayer(unittest.TestCase):
    """Test cases for BusinessRulesLayer."""

    def setUp(self) -> None:
        """Set up test cases."""
        self.batch_size = 32
        self.input_dim = 10
        self.numeric_rules = [(">", 0.0), ("<", 1.0)]
        self.categorical_rules = [
            ("in", ["red", "green", "blue"]),
            ("not in", ["invalid", "error"]),
        ]

    def test_initialization(self) -> None:
        """Test layer initialization."""
        # Test numeric rules
        layer = BusinessRulesLayer(rules=self.numeric_rules, feature_type="numerical")
        self.assertEqual(layer.rules, self.numeric_rules)
        self.assertEqual(layer.feature_type, "numerical")
        self.assertTrue(layer.weights_trainable)

        # Test categorical rules
        layer = BusinessRulesLayer(
            rules=self.categorical_rules,
            feature_type="categorical",
            trainable_weights=False,
        )
        self.assertEqual(layer.rules, self.categorical_rules)
        self.assertEqual(layer.feature_type, "categorical")
        self.assertFalse(layer.weights_trainable)

    def test_invalid_initialization(self) -> None:
        """Test invalid initialization parameters."""
        # Invalid feature type
        with self.assertRaises(ValueError):
            BusinessRulesLayer(rules=self.numeric_rules, feature_type="invalid")

        # Invalid numeric operator
        with self.assertRaises(ValueError):
            layer = BusinessRulesLayer(
                rules=[("invalid", 0.0)],
                feature_type="numerical",
            )
            inputs = tf.random.uniform((self.batch_size, 1))
            layer(inputs)

        # Invalid categorical operator
        with self.assertRaises(ValueError):
            layer = BusinessRulesLayer(
                rules=[("invalid", ["red"])],
                feature_type="categorical",
            )
            inputs = tf.constant([["red"]], dtype=tf.string)
            layer(inputs)

    def test_numerical_rules(self) -> None:
        """Test numerical rule application."""
        layer = BusinessRulesLayer(rules=self.numeric_rules, feature_type="numerical")

        # Test values within range
        inputs = tf.random.uniform((self.batch_size, 1))
        outputs = layer(inputs)

        self.assertIn("business_score", outputs)
        self.assertIn("business_proba", outputs)
        self.assertIn("business_anomaly", outputs)
        self.assertIn("business_reason", outputs)
        self.assertIn("business_value", outputs)

        # Test values outside range
        inputs = tf.constant([[-1.0], [0.5], [1.5]], dtype=tf.float32)
        outputs = layer(inputs)

        anomalies = outputs["business_anomaly"].numpy()
        self.assertTrue(anomalies[0])  # -1.0 < 0
        self.assertFalse(anomalies[1])  # 0.5 is within range
        self.assertTrue(anomalies[2])  # 1.5 > 1

    def test_categorical_rules(self) -> None:
        """Test categorical rule application."""
        layer = BusinessRulesLayer(
            rules=self.categorical_rules,
            feature_type="categorical",
        )

        # Test valid and invalid values
        inputs = tf.constant([["red"], ["invalid"], ["yellow"]], dtype=tf.string)
        outputs = layer(inputs)

        anomalies = outputs["business_anomaly"].numpy()
        self.assertFalse(anomalies[0])  # "red" is allowed
        self.assertTrue(anomalies[1])  # "invalid" is disallowed
        self.assertTrue(anomalies[2])  # "yellow" not in allowed set

    def test_trainable_weights(self) -> None:
        """Test trainable weights behavior."""
        # Test with trainable weights
        layer = BusinessRulesLayer(
            rules=self.numeric_rules,
            feature_type="numerical",
            trainable_weights=True,
        )
        self.assertTrue(layer.weights_trainable)

        # Test with non-trainable weights
        layer = BusinessRulesLayer(
            rules=self.numeric_rules,
            feature_type="numerical",
            trainable_weights=False,
        )
        self.assertFalse(layer.weights_trainable)

    def test_serialization(self) -> None:
        """Test layer serialization."""
        # Create and save model with layer
        inputs = Input(shape=(1,))
        layer = BusinessRulesLayer(rules=self.numeric_rules, feature_type="numerical")
        outputs = layer(inputs)
        model = Model(inputs=inputs, outputs=outputs)

        # Save and load model
        model_json = model.to_json()
        loaded_model = tf.keras.models.model_from_json(
            model_json,
            custom_objects={"BusinessRulesLayer": BusinessRulesLayer},
        )

        # Test loaded model
        test_input = tf.random.uniform((1, 1))
        original_output = model(test_input)
        loaded_output = loaded_model(test_input)

        # Compare outputs
        for key in original_output:
            if key == "business_reason":
                # Compare strings directly
                self.assertEqual(
                    original_output[key].numpy().tolist(),
                    loaded_output[key].numpy().tolist(),
                )
            else:
                # Compare numerical values
                self.assertTrue(
                    np.allclose(
                        original_output[key].numpy(),
                        loaded_output[key].numpy(),
                    ),
                )


if __name__ == "__main__":
    unittest.main()
