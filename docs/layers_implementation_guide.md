# 🧩 Layer Implementation Guide for Keras Model Registry (KMR)

This guide outlines the complete process and best practices for implementing new layers in the Keras Model Registry project. Follow the checklists to ensure your implementation meets all KMR standards.

## 📋 Layer Implementation Checklist

Use this checklist when implementing a new layer. Check off each item as you complete it.

### Phase 1: Planning & Design
- [ ] **Define Purpose**: Clearly document what the layer does and when to use it
- [ ] **Review Existing Layers**: Check if similar functionality already exists in `kmr/layers/`
- [ ] **Plan Architecture**: Design the layer's interface (parameters, inputs, outputs)
- [ ] **Review Keras 3 APIs**: Ensure all operations are available in Keras 3
- [ ] **Check Dependencies**: Verify no TensorFlow-specific code is needed

### Phase 2: Implementation - Core Layer Code
- [ ] **Create File**: Create `kmr/layers/YourLayerName.py` following naming conventions
- [ ] **Add Module Docstring**: Document the module's purpose
- [ ] **Implement Pure Keras 3**: Use only Keras operations (no TensorFlow)
- [ ] **Apply @register_keras_serializable**: Decorate class with `@register_keras_serializable(package="kmr.layers")`
- [ ] **Inherit from BaseLayer**: Extend `kmr.layers._base_layer.BaseLayer`
- [ ] **Implement __init__**: 
  - [ ] Set private attributes first (`self._param = param`)
  - [ ] Validate parameters (in __init__ or _validate_params)
  - [ ] Set public attributes (`self.param = self._param`)
  - [ ] Call `super().__init__(name=name, **kwargs)` AFTER setting public attributes
- [ ] **Implement _validate_params**: Add parameter validation logic
- [ ] **Implement build()**: Initialize weights and sublayers
- [ ] **Implement call()**: Implement forward pass with Keras operations only
- [ ] **Implement get_config()**: Return all constructor parameters
- [ ] **Add Type Hints**: All methods and parameters have proper type annotations
- [ ] **Add Logging**: Use `loguru` for debug messages
- [ ] **Add Docstrings**: Comprehensive Google-style docstrings for all methods

### Phase 3: Unit Tests
- [ ] **Create Test File**: Create `tests/layers/test__YourLayerName.py`
- [ ] **Test Initialization**: 
  - [ ] Default parameters
  - [ ] Custom parameters
  - [ ] Invalid parameters (should raise errors)
- [ ] **Test Layer Building**: Build with different input shapes
- [ ] **Test Output Shape**: Verify output shapes match expected values
- [ ] **Test Output Type**: Verify output is correct dtype
- [ ] **Test Different Batch Sizes**: Test with various batch dimensions
- [ ] **Test Serialization**:
  - [ ] `get_config()` returns correct dict
  - [ ] `from_config()` recreates layer correctly
  - [ ] `keras.saving.serialize_keras_object()` works
  - [ ] `keras.saving.deserialize_keras_object()` works
  - [ ] Model with layer can be saved/loaded (`.keras` format)
  - [ ] Weights can be saved/loaded (`.h5` format)
- [ ] **Test Deterministic Output**: Same input produces same output
- [ ] **Test Training Mode**: Layer behaves differently in training vs inference (if applicable)
- [ ] **All Tests Pass**: Run `pytest tests/layers/test__YourLayerName.py -v`

### Phase 4: Documentation
- [ ] **Create Documentation File**: Create `docs/layers/your-layer-name.md`
- [ ] **Follow Template**: Use structure from similar layer in `docs/layers/`
- [ ] **Include Sections**:
  - [ ] Overview and purpose
  - [ ] How it works (with Mermaid diagram if helpful)
  - [ ] Why use it and use cases
  - [ ] Quick start example
  - [ ] Advanced usage
  - [ ] Parameter guide
  - [ ] Performance characteristics
  - [ ] Testing section
  - [ ] Common issues & troubleshooting
  - [ ] Related layers
  - [ ] References
- [ ] **Add Code Examples**: Real, working examples
- [ ] **Include Mathematical Details**: If applicable
- [ ] **Add Visual Aids**: Diagrams, flowcharts, or Mermaid diagrams

### Phase 5: Integration & Updates
- [ ] **Update Imports**: Add to `kmr/layers/__init__.py`
  - [ ] Add import statement
  - [ ] Add layer name to `__all__` list
- [ ] **Update API Documentation**: Add entry to `docs/api/layers.md`
  - [ ] Add layer name and description
  - [ ] Include autodoc reference (`::: kmr.layers.YourLayerName`)
- [ ] **Update Layers Overview**: Add to `docs/layers_overview.md`
  - [ ] Add to appropriate category
  - [ ] Add API reference card
- [ ] **Update Data Analyzer**: If applicable, add to `kmr/utils/data_analyzer.py`
  - [ ] Add to appropriate data characteristic
  - [ ] Update layer recommendations
- [ ] **Update Contributing Guide**: If introducing new patterns

### Phase 6: Quality Assurance
- [ ] **Run All Tests**: `pytest tests/ -v` - ensure no regressions
- [ ] **Pre-commit Hooks**: Run `pre-commit run --all-files`
  - [ ] Black formatting passes
  - [ ] Ruff linting passes
  - [ ] No unused imports or variables
  - [ ] Proper type hints
- [ ] **Documentation Build**: `mkdocs serve` builds without errors
- [ ] **Code Review**: Request code review from team
- [ ] **Integration Test**: Test layer in a complete model

---

## Key Requirements

### ✅ Keras 3 Only
All layer implementations MUST use only Keras 3 operations. NO TensorFlow dependencies are allowed in layer implementations.
- **Allowed**: `keras.layers`, `keras.ops`, `keras.activations`
- **NOT Allowed**: `tensorflow.python.*`, `tf.nn.*` (use `keras.ops.*` instead)
- **Exception**: TensorFlow can ONLY be used in test files for validation purposes

### ✅ Type Annotations (Python 3.12+)
Use modern type hints with the union operator:
```python
param: int | float = 0.1  # Instead of Union[int, float]
```

### ✅ Google-Style Docstrings
Use Google-style docstrings for all classes and methods:
```python
def my_method(self, param: str) -> int:
    """Short description.
    
    Longer description if needed.
    
    Args:
        param: Description of parameter.
        
    Returns:
        Description of return value.
        
    Raises:
        ValueError: When something is invalid.
    """
```

---

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
    """Short description.
    
    Longer description of what this layer does and when to use it.

    Args:
        param1: Description of param1 with type.
        param2: Description of param2 with type.
        name: Optional name for the layer.

    Input shape:
        `(batch_size, ...)` - Description of input tensor.

    Output shape:
        `(batch_size, ...)` - Description of output tensor.

    Example:
        ```python
        import keras
        from kmr.layers import MyCustomLayer
        
        # Create layer
        layer = MyCustomLayer(param1=value1, param2=value2)
        
        # Use in a model
        inputs = keras.Input(shape=(10,))
        outputs = layer(inputs)
        model = keras.Model(inputs, outputs)
        ```
    """

    def __init__(
        self,
        param1: int = 10,
        param2: float = 0.1,
        name: str | None = None,
        **kwargs: Any
    ) -> None:
        # Set private attributes first
        self._param1 = param1
        self._param2 = param2

        # Validate parameters
        self._validate_params()

        # Set public attributes BEFORE calling super().__init__()
        # This is required because BaseLayer._log_initialization() calls get_config()
        self.param1 = self._param1
        self.param2 = self._param2

        # Initialize any other instance variables
        self.some_variable = None

        # Call parent's __init__ last
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if self._param1 < 0:
            raise ValueError(f"param1 must be non-negative, got {self._param1}")
        if not (0 <= self._param2 <= 1):
            raise ValueError(f"param2 must be in [0, 1], got {self._param2}")

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Build layer with given input shape.

        Args:
            input_shape: Tuple of integers defining the input shape.
        """
        # Create weights and sublayers here
        # Example:
        # self.dense = layers.Dense(self._param1)
        
        logger.debug(f"Building {self.__class__.__name__} with params: param1={self.param1}, param2={self.param2}")
        super().build(input_shape)

    def call(self, inputs: KerasTensor, training: bool | None = None) -> KerasTensor:
        """Forward pass.

        Args:
            inputs: Input tensor.
            training: Boolean or None, whether the layer should behave in training mode or inference mode.

        Returns:
            Output tensor.
        """
        # Implement forward pass using ONLY Keras operations
        # Use keras.ops.* instead of tf.*
        output = inputs  # Replace with actual implementation
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

## Keras 3 Operations Reference

When implementing layers, use Keras 3 operations instead of TensorFlow operations:

| Operation | TensorFlow | Keras 3 |
|-----------|------------|---------|
| Stacking | `tf.stack` | `keras.ops.stack` |
| Reshape | `tf.reshape` | `keras.ops.reshape` |
| Sum | `tf.reduce_sum` | `keras.ops.sum` |
| Mean | `tf.reduce_mean` | `keras.ops.mean` |
| Max | `tf.reduce_max` | `keras.ops.max` |
| Min | `tf.reduce_min` | `keras.ops.min` |
| Softmax | `tf.nn.softmax` | `keras.ops.softmax` |
| Concatenate | `tf.concat` | `keras.ops.concatenate` |
| Power | `tf.math.pow` | `keras.ops.power` |
| Absolute | `tf.abs` | `keras.ops.absolute` |
| Cast | `tf.cast` | `keras.ops.cast` |
| Transpose | `tf.transpose` | `keras.ops.transpose` |
| Squeeze | `tf.squeeze` | `keras.ops.squeeze` |
| Expand dims | `tf.expand_dims` | `keras.ops.expand_dims` |
| Gather | `tf.gather` | `keras.ops.take` |
| Slice | `tf.slice` | `keras.ops.slice` |
| Pad | `tf.pad` | `keras.ops.pad` |

## Common Pitfalls & Solutions

| Pitfall | Problem | Solution |
|---------|---------|----------|
| TensorFlow Dependencies | Using `tf.*` operations | Use `keras.ops.*` instead |
| Wrong Attribute Order | `AttributeError` during initialization | Set public attributes BEFORE `super().__init__()` |
| Missing Imports | `ImportError` | Check all imports are included |
| Incomplete Serialization | Layer cannot be loaded | Include all parameters in `get_config()` |
| Missing Type Hints | Code quality issues | Add type annotations to all methods |
| Insufficient Documentation | Users can't use the layer | Write comprehensive docstrings |
| Improper Validation | Invalid parameters accepted | Validate in `__init__()` or `_validate_params()` |
| No Pre-commit Checks | Code style issues | Run `pre-commit run --all-files` |
| Untested Code | Bugs in production | Write comprehensive unit tests |
| Missing Tests | Serialization breaks | Add serialization tests |

---

## Testing Template

Create comprehensive tests following this template:

```python
import unittest
import numpy as np
import tensorflow as tf
import keras

from kmr.layers import MyCustomLayer

class TestMyCustomLayer(unittest.TestCase):
    """Test suite for MyCustomLayer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.layer = MyCustomLayer(param1=10, param2=0.1)
        self.input_shape = (32, 20)  # batch_size, feature_dim
        self.input_data = np.random.randn(*self.input_shape).astype(np.float32)

    def test_initialization(self) -> None:
        """Test layer initialization."""
        self.assertEqual(self.layer.param1, 10)
        self.assertEqual(self.layer.param2, 0.1)

    def test_invalid_parameters(self) -> None:
        """Test invalid parameter handling."""
        with self.assertRaises(ValueError):
            MyCustomLayer(param1=-1)

    def test_output_shape(self) -> None:
        """Test output shape."""
        output = self.layer(self.input_data)
        self.assertEqual(output.shape, self.input_shape)

    def test_serialization(self) -> None:
        """Test layer serialization."""
        config = self.layer.get_config()
        new_layer = MyCustomLayer.from_config(config)
        
        output1 = self.layer(self.input_data)
        output2 = new_layer(self.input_data)
        
        np.testing.assert_allclose(output1, output2, rtol=1e-5)

    def test_model_save_load(self) -> None:
        """Test model with layer can be saved and loaded."""
        import tempfile
        
        inputs = keras.Input(shape=(20,))
        outputs = self.layer(inputs)
        model = keras.Model(inputs, outputs)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = f"{tmpdir}/model.keras"
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)
            
            pred1 = model.predict(self.input_data, verbose=0)
            pred2 = loaded_model.predict(self.input_data, verbose=0)
            
            np.testing.assert_allclose(pred1, pred2, rtol=1e-5)

if __name__ == "__main__":
    unittest.main()
```

---

## Next Steps

After implementing and testing your layer:

1. **Submit for Review**: Create a pull request with your implementation
2. **Address Feedback**: Update based on review comments
3. **Merge**: Once approved, merge to main branch
4. **Announce**: Notify team about new layer availability
5. **Update README**: If it's a major layer, update main README 