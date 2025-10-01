import unittest
from keras import ops
from kmr.layers.DifferentialPreprocssing import DifferentialPreprocssingLayer


class TestDifferentialPreprocssingLayer(unittest.TestCase):
    """Test suite for DifferentialPreprocssingLayer.

    Tests initialization, shape handling, build validation,
    serialization, training mode behavior, and layer-specific
    functionality like imputation and transformations.
    """

    def setUp(self) -> None:
        """Initialize test variables and layer instance."""
        self.batch_size = 32
        self.num_features = 10
        self.mlp_hidden_units = 8
        self.layer = DifferentialPreprocssingLayer(
            num_features=self.num_features,
            mlp_hidden_units=self.mlp_hidden_units,
        )

    def test_initialization(self) -> None:
        """Test layer initialization and parameter validation."""
        layer = DifferentialPreprocssingLayer(num_features=5, mlp_hidden_units=4)
        self.assertEqual(layer.num_features, 5)
        self.assertEqual(layer.mlp_hidden_units, 4)
        self.assertEqual(layer.num_candidates, 4)

    def test_output_shape(self) -> None:
        """Test that output shape matches input shape after transformation."""
        input_data = ops.ones((self.batch_size, self.num_features))
        output = self.layer(input_data)
        self.assertEqual(output.shape, (self.batch_size, self.num_features))

    def test_build_validation(self) -> None:
        """Test layer building with various input shapes."""
        # Test valid input shape
        input_shape = (None, self.num_features)
        self.layer.build(input_shape)
        self.assertTrue(self.layer.built)

        # Test invalid input shape
        layer = DifferentialPreprocssingLayer(
            num_features=self.num_features,
            mlp_hidden_units=self.mlp_hidden_units,
        )
        with self.assertRaises(ValueError):
            layer.build((None, self.num_features + 1))

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        config = self.layer.get_config()
        new_layer = DifferentialPreprocssingLayer.from_config(config)

        self.assertEqual(new_layer.num_features, self.layer.num_features)
        self.assertEqual(new_layer.mlp_hidden_units, self.layer.mlp_hidden_units)
        self.assertEqual(new_layer.num_candidates, self.layer.num_candidates)

    def test_imputation(self) -> None:
        """Test missing value imputation functionality."""
        # Create data with NaN values
        x = ops.convert_to_tensor(
            [
                [1.0, float("nan"), 3.0],
                [float("nan"), 2.0, float("nan")],
                [1.0, 2.0, 3.0],
            ],
        )

        layer = DifferentialPreprocssingLayer(num_features=3)
        output = layer(x)

        # Check output shape and no NaN values
        self.assertEqual(output.shape, (3, 3))
        self.assertFalse(ops.any(ops.isnan(output)))

    def test_transformations(self) -> None:
        """Test all transformation branches (identity, affine, nonlinear, log)."""
        x = ops.ones((self.batch_size, self.num_features))
        output = self.layer(x)

        # Check that output values are finite and have correct shape
        self.assertTrue(ops.all(ops.isfinite(output)))
        self.assertEqual(output.shape, (self.batch_size, self.num_features))

    def test_training_mode(self) -> None:
        """Test layer behavior in training vs inference modes."""
        x = ops.ones((self.batch_size, self.num_features))

        # Training mode
        training_output = self.layer(x, training=True)

        # Inference mode
        inference_output = self.layer(x, training=False)

        # Outputs should have same shape in both modes
        self.assertEqual(training_output.shape, inference_output.shape)


if __name__ == "__main__":
    unittest.main()
