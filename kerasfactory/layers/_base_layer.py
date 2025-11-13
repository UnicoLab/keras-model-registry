"""Base layer implementation for the Keras Model Registry."""
from typing import Any, ClassVar, TypeVar

from keras import Layer
from keras.saving import register_keras_serializable
from loguru import logger

T = TypeVar("T", bound="BaseLayer")


@register_keras_serializable(package="kerasfactory.layers")
class BaseLayer(Layer):
    """Base class for all layers in the Keras Model Registry.

    This class provides common functionality and patterns that should be used
    across all layers in the package. It includes:
    1. Standard initialization and configuration handling
    2. Logging support with loguru
    3. Type hints and validation
    4. Common utility methods

    Example:
        ```python
        from kerasfactory.layers import BaseLayer

        class CustomLayer(BaseLayer):
            def __init__(self, units: int, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self.units = units
                self._validate_params()

            def _validate_params(self) -> None:
                if self.units <= 0:
                    raise ValueError(f"units must be positive, got {self.units}")
        ```
    """

    # Class-level configuration
    _required_kwargs: ClassVar[set[str]] = set()
    _optional_kwargs: ClassVar[set[str]] = set()
    _valid_dtypes: ClassVar[set[str]] = {"float32", "float64"}

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the base layer.

        Args:
            **kwargs: Additional keyword arguments passed to the parent Layer.
        """
        # Extract layer-specific kwargs
        layer_kwargs = {}
        for key in ["trainable", "name", "dtype", "dynamic"]:
            if key in kwargs:
                layer_kwargs[key] = kwargs.pop(key)

        super().__init__(**layer_kwargs)
        self._log_initialization()

    def _validate_params(self) -> None:
        """Validate layer parameters.

        This method should be overridden by child classes to implement
        parameter validation logic.

        Raises:
            NotImplementedError: If the child class doesn't implement this method.
        """
        raise NotImplementedError(
            f"_validate_params not implemented for {self.__class__.__name__}",
        )

    def _log_initialization(self) -> None:
        """Log layer initialization details using loguru."""
        logger.debug(
            f"Initialized {self.__class__.__name__} with parameters: {self.get_config()}",
        )

    def _validate_dtype(self, tensor: Any, name: str) -> None:
        """Validate tensor dtype.

        Args:
            tensor: Input tensor to validate.
            name: Name of the tensor for error messages.

        Raises:
            ValueError: If tensor dtype is not supported.
        """
        dtype = getattr(tensor.dtype, "name", str(tensor.dtype))
        if dtype not in self._valid_dtypes:
            raise ValueError(
                f"Unsupported dtype {dtype} for {name}. Supported dtypes: {self._valid_dtypes}",
            )

    def get_config(self) -> dict[str, Any]:
        """Return the config dictionary for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        return super().get_config()

    @classmethod
    def from_config(cls: type[T], config: dict[str, Any]) -> T:
        """Create a layer instance from its config.

        Args:
            config: Layer configuration dictionary.

        Returns:
            A new instance of the layer.
        """
        return cls(**config)
