"""Unit tests for BaseLayer."""
from typing import Any
import unittest

import keras
from keras import ops

from kmr.layers._base_layer import BaseLayer


class TestCustomLayer(BaseLayer):
    """Test layer implementation for testing BaseLayer functionality."""

    def __init__(self, units: int, activation: str = 'linear', **kwargs) -> None:
        """Initialize test layer.

        Args:
            units: Number of output units.
            activation: Activation function. Default: 'linear'.
            **kwargs: Additional layer arguments.
        """
        # Set attributes before parent initialization for get_config
        self.units = units
        self.activation = activation
        super().__init__(**kwargs)
        self._validate_params()

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if self.units <= 0:
            raise ValueError(f"units must be positive, got {self.units}")
        if self.activation not in {'linear', 'relu'}:
            raise ValueError(
                f"activation must be 'linear' or 'relu', got {self.activation}"
            )

    def call(self, inputs):
        """Forward pass."""
        return inputs

    def get_config(self) -> dict[str, Any]:
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': self.activation,
        })
        return config


class TestBaseLayer(unittest.TestCase):
    """Test cases for BaseLayer."""

    def setUp(self) -> None:
        """Set up test cases."""
        self.valid_params = {
            'units': 10,
            'activation': 'relu'
        }
        self.layer = TestCustomLayer(**self.valid_params)

    def test_initialization(self) -> None:
        """Test layer initialization."""
        self.assertEqual(self.layer.units, 10)
        self.assertEqual(self.layer.activation, 'relu')

    def test_required_params(self) -> None:
        """Test required parameter validation."""
        with self.assertRaises(TypeError):
            TestCustomLayer(activation='relu')

    def test_invalid_params(self) -> None:
        """Test invalid parameter validation."""
        with self.assertRaises(ValueError):
            TestCustomLayer(units=-1)
        with self.assertRaises(ValueError):
            TestCustomLayer(units=10, activation='invalid')

    def test_dtype_validation(self) -> None:
        """Test tensor dtype validation."""
        # Valid dtype
        valid_tensor = ops.ones((10, 10), dtype='float32')
        self.layer._validate_dtype(valid_tensor, 'valid_tensor')

        # Invalid dtype
        invalid_tensor = ops.ones((10, 10), dtype='int32')
        with self.assertRaises(ValueError) as ctx:
            self.layer._validate_dtype(invalid_tensor, 'invalid_tensor')
        self.assertIn("Unsupported dtype", str(ctx.exception))

    def test_parameter_validation(self) -> None:
        """Test parameter validation."""
        # Test validation method directly
        layer = TestCustomLayer(units=10)
        self.assertIsNone(layer._validate_params())

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        config = self.layer.get_config()
        reconstructed_layer = TestCustomLayer.from_config(config)

        self.assertEqual(self.layer.units, reconstructed_layer.units)
        self.assertEqual(self.layer.activation, reconstructed_layer.activation)

    def test_model_integration(self) -> None:
        """Test layer works in a Keras model."""
        model = keras.Sequential([self.layer])
        
        # Test forward pass
        inputs = ops.ones((1, 10))
        outputs = model(inputs)
        self.assertEqual(outputs.shape, (1, 10))


if __name__ == '__main__':
    unittest.main()
