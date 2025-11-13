import unittest
import numpy as np
from keras import ops
from kerasfactory.layers.NumericalAnomalyDetection import NumericalAnomalyDetection


class TestNumericalAnomalyDetection(unittest.TestCase):
    """Test suite for NumericalAnomalyDetection layer."""

    def setUp(self) -> None:
        """Set up test cases."""
        self.batch_size = 32
        self.num_features = 5
        self.hidden_dims = [8, 4]
        self.layer = NumericalAnomalyDetection(
            hidden_dims=self.hidden_dims,
            reconstruction_weight=0.5,
            distribution_weight=0.5,
        )

    def test_initialization(self) -> None:
        """Test layer initialization."""
        self.assertEqual(self.layer.hidden_dims, self.hidden_dims)
        self.assertEqual(self.layer.reconstruction_weight, 0.5)
        self.assertEqual(self.layer.distribution_weight, 0.5)

    def test_build(self) -> None:
        """Test layer building."""
        input_shape = (self.batch_size, self.num_features)
        self.layer.build(input_shape)

        # Check encoder layers
        self.assertEqual(len(self.layer.encoder_layers), len(self.hidden_dims))
        for i, layer in enumerate(self.layer.encoder_layers):
            self.assertEqual(layer.units, self.hidden_dims[i])

        # Check decoder layers
        expected_decoder_dims = list(reversed(self.hidden_dims[:-1])) + [
            self.num_features,
        ]
        self.assertEqual(len(self.layer.decoder_layers), len(expected_decoder_dims))
        for i, layer in enumerate(self.layer.decoder_layers):
            self.assertEqual(layer.units, expected_decoder_dims[i])

        # Check distribution layers
        self.assertEqual(self.layer.mean_layer.units, self.num_features)
        self.assertEqual(self.layer.var_layer.units, self.num_features)

    def test_call(self) -> None:
        """Test forward pass."""
        inputs = ops.convert_to_tensor(
            np.random.normal(size=(self.batch_size, self.num_features)).astype(
                np.float32,
            ),
        )
        self.layer.build(inputs.shape)
        outputs = self.layer(inputs)

        # Check output shape
        self.assertEqual(outputs.shape, inputs.shape)

        # Check output values are finite
        self.assertTrue(ops.all(ops.isfinite(outputs)))

        # Check output values are non-negative (anomaly scores)
        self.assertTrue(ops.all(outputs >= 0))

    def test_compute_output_shape(self) -> None:
        """Test output shape computation."""
        input_shape = (self.batch_size, self.num_features)
        output_shape = self.layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, input_shape)

    def test_get_config(self) -> None:
        """Test layer serialization."""
        config = self.layer.get_config()
        self.assertEqual(config["hidden_dims"], self.hidden_dims)
        self.assertEqual(config["reconstruction_weight"], 0.5)
        self.assertEqual(config["distribution_weight"], 0.5)

        # Test reconstruction
        reconstructed_layer = NumericalAnomalyDetection.from_config(config)
        self.assertEqual(reconstructed_layer.hidden_dims, self.hidden_dims)
        self.assertEqual(reconstructed_layer.reconstruction_weight, 0.5)
        self.assertEqual(reconstructed_layer.distribution_weight, 0.5)

    def test_edge_cases(self) -> None:
        """Test edge cases and error conditions."""
        # Test with single feature
        layer = NumericalAnomalyDetection(hidden_dims=[2, 1])
        inputs = ops.convert_to_tensor(
            np.random.normal(size=(self.batch_size, 1)).astype(np.float32),
        )
        layer.build(inputs.shape)
        outputs = layer(inputs)
        self.assertEqual(outputs.shape, inputs.shape)

        # Test with large number of features
        layer = NumericalAnomalyDetection(hidden_dims=[64, 32])
        inputs = ops.convert_to_tensor(
            np.random.normal(size=(self.batch_size, 50)).astype(np.float32),
        )
        layer.build(inputs.shape)
        outputs = layer(inputs)
        self.assertEqual(outputs.shape, inputs.shape)


if __name__ == "__main__":
    unittest.main()
