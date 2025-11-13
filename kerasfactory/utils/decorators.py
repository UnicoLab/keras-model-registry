import functools
import inspect
from typing import Any, TypeVar
from collections.abc import Callable
from loguru import logger
from keras.saving import register_keras_serializable

T = TypeVar("T", bound=type[Any])


def log_init(cls: type[T]) -> type[T]:
    """Class decorator to log initialization arguments."""
    original_init = cls.__init__  # type: ignore

    @functools.wraps(original_init)
    def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
        # Convert input_schema to regular dict if present
        if "input_schema" in kwargs:
            kwargs["input_schema"] = dict(kwargs["input_schema"])

        # Get the signature of the original __init__
        sig = inspect.signature(original_init)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()

        # Remove 'self' from the arguments
        init_args = dict(bound_args.arguments)
        init_args.pop("self", None)

        # Store arguments for potential later use
        self._init_args = init_args

        # Separate args and kwargs based on parameter kinds
        required_args = []
        optional_kwargs = {}

        for name, param in sig.parameters.items():
            if name == "self":
                continue

            value = init_args.get(name)
            if param.default == inspect.Parameter.empty:
                required_args.append(f"{name}={value}")
            else:
                # Only include kwargs that differ from their defaults
                if value != param.default:
                    optional_kwargs[name] = value

        # Format and log the initialization message
        class_name = cls.__name__
        args_str = ", ".join(required_args)
        kwargs_str = ", ".join([f"{k}={v}" for k, v in optional_kwargs.items()])

        if kwargs_str:
            logger.info(
                f"Initializing {class_name} with args: ({args_str}) and kwargs: ({kwargs_str})",
            )
        else:
            logger.info(f"Initializing {class_name} with args: ({args_str})")

        # Call the original __init__
        original_init(self, *args, **kwargs)

    cls.__init__ = new_init  # type: ignore
    return cls


def log_method(func: Callable) -> Callable:
    """Method decorator to log method calls with their arguments."""

    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Convert input dictionaries to regular dicts
        new_args = []
        for arg in args:
            if isinstance(arg, dict):
                new_args.append(dict(arg))
            else:
                new_args.append(arg)

        new_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, dict):
                new_kwargs[key] = dict(value)
            else:
                new_kwargs[key] = value

        # Get the signature of the function
        sig = inspect.signature(func)
        bound_args = sig.bind(self, *new_args, **new_kwargs)
        bound_args.apply_defaults()

        # Remove 'self' from the arguments
        call_args = dict(bound_args.arguments)
        call_args.pop("self", None)

        # Format the log message
        method_name = func.__name__
        args_str = ", ".join([f"args={new_args}"] if new_args else [])
        kwargs_str = ", ".join([f"{k}={v}" for k, v in new_kwargs.items()])

        if args_str and kwargs_str:
            logger.info(f"Calling {method_name} with {args_str}, {kwargs_str}")
        elif args_str:
            logger.info(f"Calling {method_name} with {args_str}")
        elif kwargs_str:
            logger.info(f"Calling {method_name} with {kwargs_str}")
        else:
            logger.info(f"Calling {method_name} with ()")

        # Call the original function
        result = func(self, *new_args, **new_kwargs)
        return result

    return wrapper


def log_property(func: Callable) -> Callable:
    """Property decorator to log property access."""

    @functools.wraps(func)
    def wrapper(self: Any) -> Any:
        property_name = func.__name__
        logger.debug(f"Accessing property {property_name}")
        return func(self)

    return wrapper


def add_serialization(cls: T) -> T:
    """Decorator to add serialization methods to a Keras model class.

    Args:
        cls: The class to decorate.

    Returns:
        The decorated class.
    """
    # Register the class for Keras serialization
    cls = register_keras_serializable()(cls)

    original_init = cls.__init__

    @functools.wraps(original_init)
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the decorator.

        Args:
            self: The instance being initialized.
            *args: Provided class arguments.
            **kwargs: Provided kwargs for the class.

        """
        # Bind the arguments to get a dictionary of the parameters
        sig = inspect.signature(original_init)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        init_args = dict(bound_args.arguments)
        init_args.pop("self", None)

        # Store the initialization arguments
        self._init_args = init_args

        # Call the original __init__ method
        original_init(self, *args, **kwargs)

    def get_config(self) -> dict[str, Any]:
        """Return the configuration of the model.

        Returns:
            dict: serializable configuration of the class.
        """
        base_config = super().get_config()  # type: ignore
        return {**base_config, **self._init_args}

    @classmethod  # type: ignore
    def from_config(cls, config: dict[str, Any]) -> Any:
        """Create an instance from a configuration dictionary.

        Args:
            cls: The class being instantiated.
            config: Configuration dictionary for deserialization.
        """
        return cls(**config)

    # Assign the new methods to the class
    cls.__init__ = __init__
    cls.get_config = get_config
    cls.from_config = from_config

    return cls
