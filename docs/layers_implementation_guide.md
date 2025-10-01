# Layer Implementation Guide for Keras Model Registry (KMR)

This document outlines the standard patterns and best practices for implementing new layers in the Keras Model Registry project.

## Key Requirements

1. **Keras 3 Only**: All layer implementations MUST use only Keras 3 operations. NO TensorFlow dependencies are allowed in layer implementations.
2. **TensorFlow Usage**: TensorFlow can ONLY be used in test files for validation purposes.

## Layer Structure

All layers in the KMR project should follow this structure:

1. **Module Docstring**: Describe the purpose and functionality of the layer.
2. **Imports**: Import necessary dependencies (Keras only, no TensorFlow).
3. **Class Definition**: Define the layer class inheriting from `BaseLayer`.
4. **Class Docstring**: Comprehensive documentation including:
   - General description
   - Parameters with types and defaults
   - Input/output shapes
   - Usage examples
5. **Implementation**: The actual layer implementation using only Keras 3 operations.

## Required Components

Every layer must include:

1. **Keras Serialization**: Use the `@register_keras_serializable(package="kmr.layers")` decorator.
2. **BaseLayer Inheritance**: Inherit from `kmr.layers._base_layer.BaseLayer`.
3. **Type Annotations**: Use proper type hints for all methods and parameters.
4. **Parameter Validation**: Validate input parameters in `__init__` or `_validate_params`.
5. **Logging**: Use loguru for logging important information.
6. **Serialization Support**: Implement `get_config()` method properly.

## Implementation Pattern

Follow this pattern for implementing layers:

```python
"""
Module docstring describing the layer's purpose and functionality.
"""

from typing import Any
from loguru import logger
from keras import layers, ops
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kmr.layers._base_layer import BaseLayer

@register_keras_serializable(package="kmr.layers")
class MyCustomLayer(BaseLayer):
    """Detailed class docstring with description, parameters, and examples.

    Args:
        param1: Description of param1.
        param2: Description of param2.
        ...

    Input shape:
        Description of input shape.

    Output shape:
        Description of output shape.

    Example:
        ```python
        import keras
        # Usage example with Keras operations only
        ```
    """

    def __init__(
        self,
        param1: type = default,
        param2: type = default,
        name: str | None = None,
        **kwargs: Any
    ) -> None:
        # Set private attributes before calling parent's __init__
        self._param1 = param1
        self._param2 = param2

        # Validate parameters
        if not valid_condition:
            raise ValueError("Error message")

        # IMPORTANT: Set public attributes BEFORE calling parent's __init__
        # This is necessary because BaseLayer._log_initialization() calls get_config()
        # which accesses these public attributes
        self.param1 = self._param1
        self.param2 = self._param2
        
        # Initialize any other instance variables
        self.some_variable = None

        # Call parent's __init__ after setting public attributes
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if not valid_condition:
            raise ValueError("Error message")

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Builds the layer with the given input shape.

        Args:
            input_shape: Tuple of integers defining the input shape.
        """
        # Create weights and sublayers
        
        logger.debug(f"Layer built with params: {self.param1}, {self.param2}")
        super().build(input_shape)

    def call(self, inputs: KerasTensor, training: bool | None = None) -> KerasTensor:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Output tensor.
        """
        # Implement forward pass using ONLY Keras operations
        # Use ops.xxx instead of tf.xxx
        return output

    def get_config(self) -> dict[str, Any]:
        """Returns the config of the layer.

        Returns:
            Python dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "param1": self.param1,
            "param2": self.param2,
        })
        return config
```

## Attribute Initialization Order

Pay special attention to the order of operations in the `__init__` method:

1. Set private attributes first (`self._param1 = param1`)
2. Validate parameters
3. Set public attributes (`self.param1 = self._param1`)
4. Initialize any other instance variables
5. Call `super().__init__(name=name, **kwargs)`

This order is critical because `BaseLayer._log_initialization()` is called during `super().__init__()` and it calls `get_config()`, which accesses the public attributes. If the public attributes are not set before calling `super().__init__()`, you'll get an `AttributeError`.

## Keras 3 Operations Reference

When implementing layers, use Keras 3 operations instead of TensorFlow operations:

| TensorFlow | Keras 3 |
|------------|---------|
| tf.stack | keras.ops.stack |
| tf.reshape | keras.ops.reshape |
| tf.reduce_sum | keras.ops.sum |
| tf.reduce_mean | keras.ops.mean |
| tf.reduce_max | keras.ops.max |
| tf.reduce_min | keras.ops.min |
| tf.nn.softmax | keras.ops.softmax |
| tf.concat | keras.ops.concatenate |
| tf.math.pow | keras.ops.power |
| tf.abs | keras.ops.absolute |

For a more comprehensive list, refer to the KERAS_DICT.md file.

## Testing

Each layer should have a corresponding test file in `tests/layers/` with the naming pattern `test__LayerName.py`. Tests should include:

1. **Initialization Tests**: Test default and custom initialization.
2. **Invalid Parameter Tests**: Test error handling for invalid parameters.
3. **Build Tests**: Test layer building with different configurations.
4. **Output Shape Tests**: Test that output shape matches expectations.
5. **Training Mode Tests**: Test behavior in training vs inference modes.
6. **Serialization Tests**: Test serialization and deserialization.
7. **Functional Tests**: Test specific functionality of the layer.
8. **Integration Tests**: Test integration with a simple model.

Note: TensorFlow can be used in test files for validation purposes, but should be clearly marked as such.

## Registration

After implementing a new layer:

1. Add an import statement in `kmr/layers/__init__.py`
2. Add the layer name to the `__all__` list in the same file.

## Common Pitfalls

1. **TensorFlow Dependencies**: NEVER use TensorFlow operations in layer implementations.
2. **Incorrect Attribute Initialization Order**: Always set public attributes BEFORE calling `super().__init__()`.
3. **Missing Imports**: Ensure all necessary imports are included.
4. **Incomplete Serialization**: Make sure all parameters are included in `get_config()`.
5. **Missing Type Hints**: Always include proper type annotations.
6. **Insufficient Documentation**: Always provide comprehensive docstrings.
7. **Improper Validation**: Always validate input parameters. 