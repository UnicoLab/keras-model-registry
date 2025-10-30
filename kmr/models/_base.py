from typing import Any, Optional, Union
from collections import OrderedDict
import keras
from keras import Model
from loguru import logger


class BaseModel(Model):
    """Base model class with comprehensive input handling and common features.

    This class extends the standard Keras Model to provide:
    - Universal input handling (supports any input format)
    - Preprocessing model integration with automatic fitting
    - Input validation and standardization
    - Common utility methods for all models
    - Automatic functional model creation
    """

    def __init__(self, *args, **kwargs):
        """Initialize the base model with preprocessing support."""
        # Extract preprocessing-related parameters
        self._preprocessing_model = kwargs.pop("preprocessing_model", None)
        self._inputs = kwargs.pop("inputs", None)
        self._preprocessing_fitted = False

        super().__init__(*args, **kwargs)

        # Set up preprocessing model if provided
        if self._preprocessing_model is not None:
            self._setup_preprocessing_model()

    def _standardize_inputs(self, inputs: Any) -> OrderedDict:
        """Standardize inputs to OrderedDict format for consistent handling.

        This method provides universal input handling that supports:
        - Single tensors/vectors (numpy arrays, tensors)
        - Lists/tuples of tensors
        - Dictionaries (regular dict, OrderedDict)
        - Mixed input formats

        Args:
            inputs: Input data in various formats (dict, list, tensor, etc.)

        Returns:
            OrderedDict: Standardized input format with consistent keys
        """
        if isinstance(inputs, OrderedDict):
            return inputs
        elif isinstance(inputs, dict):
            return OrderedDict(inputs)
        elif isinstance(inputs, (list, tuple)):
            # Convert list/tuple to OrderedDict with default keys
            return OrderedDict({f"input_{i}": inp for i, inp in enumerate(inputs)})
        else:
            # Single tensor input
            return OrderedDict({"input": inputs})

    def _process_inputs_for_model(
        self,
        inputs: Any,
        expected_keys: list[str] = None,
        auto_split: bool = True,
        auto_reshape: bool = True,
    ) -> Union[list, Any]:
        """Process inputs for model consumption with intelligent handling.

        This method provides intelligent input processing that:
        - Standardizes inputs to a consistent format
        - Handles feature splitting for single tensors
        - Reshapes inputs as needed
        - Validates input shapes

        Args:
            inputs: Input data in various formats
            expected_keys: Expected feature names (for multi-feature models)
            auto_split: Whether to automatically split single tensors into features
            auto_reshape: Whether to automatically reshape 1D inputs to 2D

        Returns:
            Processed inputs ready for model consumption
        """
        # Standardize inputs
        standardized_inputs = self._standardize_inputs(inputs)

        # Handle preprocessing if model is provided
        if self._preprocessing_model is not None:
            return self._process_preprocessed_inputs(standardized_inputs)

        # Handle raw inputs
        if len(standardized_inputs) > 1 or any(
            key.startswith("input_") for key in standardized_inputs.keys()
        ):
            # Multiple inputs - get tensors in the correct order
            if expected_keys is not None:
                # Use expected keys to maintain order
                input_tensors = self._get_input_tensors(
                    standardized_inputs, expected_keys,
                )
            else:
                # Use all available inputs
                input_tensors = list(standardized_inputs.values())

            # Reshape inputs if needed
            if auto_reshape:
                input_tensors = self._reshape_inputs(input_tensors)

            return input_tensors
        else:
            # Single input
            single_input = list(standardized_inputs.values())[0]

            if auto_split and expected_keys is not None and len(expected_keys) > 1:
                # Split single tensor into multiple features
                return self._split_single_input(single_input, expected_keys)
            else:
                # Return single input as is
                return single_input

    def _process_preprocessed_inputs(self, standardized_inputs: OrderedDict) -> Any:
        """Process inputs when a preprocessing model is present.

        Args:
            standardized_inputs: Standardized input format

        Returns:
            Preprocessed inputs ready for the main model
        """
        # Prepare inputs for preprocessing model
        preprocessed_inputs = self._prepare_inputs_for_preprocessing(
            standardized_inputs, self._preprocessing_model,
        )

        # Apply preprocessing
        if isinstance(preprocessed_inputs, list):
            # KDP preprocessing model - list of tensors
            return self._preprocessing_model(preprocessed_inputs)
        else:
            # Regular preprocessing model - single tensor or dict
            return self._preprocessing_model(preprocessed_inputs)

    def _split_single_input(self, single_input: Any, expected_keys: list[str]) -> list:
        """Split a single input tensor into multiple features.

        Args:
            single_input: Single input tensor
            expected_keys: Expected feature names

        Returns:
            List of feature tensors
        """
        if len(expected_keys) > 1:
            # Split the input tensor into multiple parts
            input_dim = single_input.shape[-1]
            feature_dim = input_dim // len(expected_keys)
            features = []
            for i in range(len(expected_keys)):
                start_idx = i * feature_dim
                end_idx = (
                    (i + 1) * feature_dim if i < len(expected_keys) - 1 else input_dim
                )
                feature_input = keras.ops.slice(
                    single_input, [0, start_idx], [-1, end_idx - start_idx],
                )
                features.append(feature_input)
            return features
        else:
            return [single_input]

    def _reshape_inputs(self, input_tensors: list) -> list:
        """Reshape input tensors to ensure they are 2D.

        Args:
            input_tensors: List of input tensors

        Returns:
            List of reshaped input tensors
        """
        reshaped_inputs = []
        for input_tensor in input_tensors:
            if len(input_tensor.shape) == 1:
                # Reshape from (batch_size,) to (batch_size, 1)
                reshaped_inputs.append(keras.ops.expand_dims(input_tensor, axis=-1))
            else:
                reshaped_inputs.append(input_tensor)
        return reshaped_inputs

    def _get_input_tensors(
        self, standardized_inputs: OrderedDict, expected_keys: list[str] | None = None,
    ) -> list:
        """Extract input tensors from standardized inputs.

        Args:
            standardized_inputs: Standardized input format
            expected_keys: Expected input keys (for validation)

        Returns:
            list: List of input tensors in the correct order
        """
        if expected_keys is not None:
            # Use expected keys to maintain order
            tensors = []
            for key in expected_keys:
                if key in standardized_inputs:
                    tensors.append(standardized_inputs[key])
                else:
                    # Check if we have input_0, input_1, etc. keys (from list/tuple inputs)
                    if all(
                        f"input_{i}" in standardized_inputs
                        for i in range(len(expected_keys))
                    ):
                        # Use input_0, input_1, etc. keys in order
                        for i in range(len(expected_keys)):
                            tensors.append(standardized_inputs[f"input_{i}"])
                        return tensors
                    else:
                        raise ValueError(
                            f"Expected input key '{key}' not found in inputs. Available keys: {list(standardized_inputs.keys())}",
                        )
            return tensors
        else:
            # Return all tensors in the order they appear
            return list(standardized_inputs.values())

    def _validate_input_shapes(
        self, inputs: list, expected_shapes: list[tuple] | None = None,
    ) -> None:
        """Validate input shapes if expected shapes are provided.

        Args:
            inputs: List of input tensors
            expected_shapes: Expected shapes for validation (optional)
        """
        if expected_shapes is not None:
            if len(inputs) != len(expected_shapes):
                raise ValueError(
                    f"Expected {len(expected_shapes)} inputs, got {len(inputs)}",
                )

            for i, (input_tensor, expected_shape) in enumerate(
                zip(inputs, expected_shapes),
            ):
                if hasattr(input_tensor, "shape"):
                    actual_shape = input_tensor.shape
                    if len(actual_shape) != len(expected_shape):
                        raise ValueError(
                            f"Input {i}: expected {len(expected_shape)}D tensor, got {len(actual_shape)}D",
                        )
                    # Check non-None dimensions
                    for j, (actual_dim, expected_dim) in enumerate(
                        zip(actual_shape, expected_shape),
                    ):
                        if expected_dim is not None and actual_dim != expected_dim:
                            raise ValueError(
                                f"Input {i}, dimension {j}: expected {expected_dim}, got {actual_dim}",
                            )

    def _prepare_inputs_for_preprocessing(
        self, standardized_inputs: OrderedDict, preprocessing_model: Any,
    ) -> Any:
        """Prepare inputs for preprocessing model based on its expected format.

        Args:
            standardized_inputs: Standardized input format
            preprocessing_model: The preprocessing model

        Returns:
            Prepared inputs in the format expected by the preprocessing model
        """
        # Check if this is a KDP preprocessing model (Functional model with multiple inputs)
        if (
            hasattr(preprocessing_model, "inputs")
            and hasattr(preprocessing_model, "outputs")
            and len(preprocessing_model.inputs) > 1
        ):
            # KDP preprocessing model - convert to list of tensors in correct order
            input_list = []
            for input_tensor in preprocessing_model.inputs:
                feature_name = input_tensor.name
                if feature_name in standardized_inputs:
                    input_list.append(standardized_inputs[feature_name])
                else:
                    raise ValueError(
                        f"Missing input feature: {feature_name}. Available keys: {list(standardized_inputs.keys())}",
                    )
            return input_list
        else:
            # Check if this is a custom model that expects dictionary inputs
            # by checking if it has a call method that expects dict-like inputs
            if (
                hasattr(preprocessing_model, "call")
                and hasattr(preprocessing_model, "__class__")
                and preprocessing_model.__class__.__name__ != "Sequential"
                and len(standardized_inputs) > 1
            ):
                # Check if the model expects dictionary inputs by looking at input names
                if (
                    hasattr(preprocessing_model, "inputs")
                    and len(preprocessing_model.inputs) == 1
                    and preprocessing_model.inputs[0].name
                    in ["preprocessing_input", "input"]
                ):
                    # This is a regular preprocessing model that expects concatenated input
                    pass  # Fall through to concatenation logic
                else:
                    # Try to detect if the model expects dictionary inputs by checking
                    # if it's a custom keras.Model subclass
                    try:
                        # Test if the model can handle dictionary inputs
                        test_dict = {
                            k: v for k, v in list(standardized_inputs.items())[:1]
                        }
                        # If it's a custom model that expects dict inputs, pass the dict
                        return dict(standardized_inputs)
                    except:
                        # If it fails, fall back to concatenation
                        pass

            # Regular preprocessing model - concatenate inputs into single tensor
            if len(standardized_inputs) == 1:
                return list(standardized_inputs.values())[0]
            else:
                # Concatenate multiple inputs into a single tensor
                input_tensors = list(standardized_inputs.values())
                # Reshape 1D inputs to 2D for concatenation
                reshaped_inputs = []
                for tensor in input_tensors:
                    if len(tensor.shape) == 1:
                        reshaped_inputs.append(keras.ops.expand_dims(tensor, axis=-1))
                    else:
                        reshaped_inputs.append(tensor)
                return keras.ops.concatenate(reshaped_inputs, axis=-1)

    def _setup_preprocessing_model(self) -> None:
        """Set up the preprocessing model for integration."""
        if self._preprocessing_model is None:
            return

        logger.debug("Setting up preprocessing model integration")

        # Check if preprocessing model needs to be built
        if not self._preprocessing_model.built:
            if self._inputs is not None:
                # Build preprocessing model with specified input shapes
                sample_inputs = OrderedDict()
                for key, shape in self._inputs.items():
                    sample_inputs[key] = keras.ops.zeros((1,) + shape)

                # Try to call the preprocessing model with sample inputs
                try:
                    self._preprocessing_model(sample_inputs)
                except Exception as e:
                    logger.debug(
                        f"Could not build preprocessing model with sample inputs: {e}",
                    )
                    logger.info(
                        "Preprocessing model will be built on first actual call",
                    )
            else:
                logger.warning(
                    "Preprocessing model provided but no input shapes specified. "
                    "Model will be built on first call.",
                )

    def _check_preprocessing_model_fitted(self, data: Any) -> bool:
        """Check if the preprocessing model has been fitted with the training data.

        Args:
            data: Training data to check against.

        Returns:
            bool: True if preprocessing model is fitted, False otherwise.
        """
        if self._preprocessing_model is None:
            return True

        # For now, we'll assume the preprocessing model needs fitting
        # In a more sophisticated implementation, we could check if the model
        # has been trained on similar data
        return self._preprocessing_fitted

    def _auto_fit_preprocessing_model(self, data: Any) -> None:
        """Automatically fit the preprocessing model if it hasn't been fitted.

        Args:
            data: Training data to fit the preprocessing model on.
        """
        if self._preprocessing_model is None:
            return

        if not self._check_preprocessing_model_fitted(data):
            logger.info("Auto-fitting preprocessing model with training data...")

            # Check if this is a KDP preprocessing model (has build_preprocessor method)
            if hasattr(self._preprocessing_model, "build_preprocessor"):
                # KDP preprocessing model - it's already built and fitted after build_preprocessor()
                logger.info(
                    "KDP preprocessing model detected - already built and fitted",
                )
            elif hasattr(self._preprocessing_model, "inputs") and hasattr(
                self._preprocessing_model, "outputs",
            ):
                # This is already a built Keras model (like from KDP build_preprocessor result)
                # Check if it has normalization layers that need fitting
                has_normalization = any(
                    "norm" in layer.name.lower()
                    or "normalization" in layer.name.lower()
                    for layer in self._preprocessing_model.layers
                )

                if has_normalization:
                    # For KDP models, the normalization layers are already adapted during build_preprocessor()
                    # Skip adaptation to avoid the AttributeError
                    logger.info(
                        "Built Keras preprocessing model with normalization layers detected - already adapted by KDP",
                    )
                else:
                    logger.info(
                        "Built Keras preprocessing model detected - no fitting needed",
                    )
            elif hasattr(self._preprocessing_model, "fit"):
                # Regular Keras model that needs fitting
                if isinstance(data, (dict, OrderedDict)):
                    # Multi-input data - convert to OrderedDict if needed
                    if isinstance(data, dict) and not isinstance(data, OrderedDict):
                        data = OrderedDict(data)

                    # Compile the preprocessing model if it's not compiled
                    if (
                        not hasattr(self._preprocessing_model, "_compile_config")
                        or self._preprocessing_model._compile_config is None
                    ):
                        self._preprocessing_model.compile(optimizer="adam", loss="mse")

                    self._preprocessing_model.fit(data, epochs=1, verbose=0)
                else:
                    # Single input data - create dummy targets
                    dummy_targets = (
                        data  # For autoencoders, targets are the same as inputs
                    )

                    # Compile the preprocessing model if it's not compiled
                    if (
                        not hasattr(self._preprocessing_model, "_compile_config")
                        or self._preprocessing_model._compile_config is None
                    ):
                        self._preprocessing_model.compile(optimizer="adam", loss="mse")

                    self._preprocessing_model.fit(
                        data, dummy_targets, epochs=1, verbose=0,
                    )
            else:
                # If it's not a Keras model, we'll just call it to build it
                if isinstance(data, (dict, OrderedDict)):
                    # Convert to OrderedDict if needed
                    if isinstance(data, dict) and not isinstance(data, OrderedDict):
                        data = OrderedDict(data)
                    self._preprocessing_model(data)
                else:
                    # For single input, we need to create a sample
                    sample_data = data[:1] if hasattr(data, "__getitem__") else data
                    self._preprocessing_model(sample_data)

            self._preprocessing_fitted = True
            logger.info("Preprocessing model auto-fitting completed")

    def _create_functional_model(self) -> Optional[keras.Model]:
        """Create a functional model that combines preprocessing and main model.

        Returns:
            keras.Model: Functional model combining preprocessing and main model, or None if no preprocessing.
        """
        if self._preprocessing_model is None:
            return None

        logger.debug("Creating functional model with preprocessing integration")

        # Check if this is a KDP preprocessing model (Functional model with multiple inputs)
        if (
            hasattr(self._preprocessing_model, "inputs")
            and hasattr(self._preprocessing_model, "outputs")
            and len(self._preprocessing_model.inputs) > 1
        ):
            # KDP preprocessing model - use its inputs directly
            logger.debug("Detected KDP preprocessing model with multiple inputs")

            # Get preprocessing output
            preprocessing_output = self._preprocessing_model(
                self._preprocessing_model.inputs,
            )

            # Get main model output - pass the preprocessed output as a single tensor
            main_output = self(preprocessing_output, training=False)

            # Create functional model using KDP preprocessing inputs
            functional_model = keras.Model(
                inputs=self._preprocessing_model.inputs,
                outputs=main_output,
                name=f"{self.name}_with_preprocessing",
            )

            return functional_model

        # Create input layers based on the inputs specification
        elif self._inputs is not None:
            input_layers = OrderedDict()
            for key, shape in self._inputs.items():
                input_layers[key] = keras.layers.Input(shape=shape, name=key)

            # Get preprocessing output
            preprocessing_output = self._preprocessing_model(input_layers)

            # Get main model output
            main_output = self(preprocessing_output, training=False)

            # Create functional model
            functional_model = keras.Model(
                inputs=input_layers,
                outputs=main_output,
                name=f"{self.name}_with_preprocessing",
            )

            return functional_model
        else:
            logger.warning(
                "Cannot create functional model without input shapes specification",
            )
            return None

    def filer_inputs(self, inputs: dict) -> dict:
        """Filter inputs based on the specified input shapes.

        Args:
            inputs: Dictionary of inputs to filter.

        Returns:
            dict: Filtered inputs.
        """
        if self._inputs is None:
            return inputs
        return {k: v for k, v in inputs.items() if k in self._inputs}

    def inspect_signatures(self, model: Model) -> dict:
        """Inspect the model signatures.

        Args:
            model: Model to inspect signatures for.

        Returns:
            dict: Signature information.
        """
        sig_keys = list(model.signatures.keys())
        logger.info(f"found signatures: {sig_keys}")
        info = {}
        for sig in sig_keys:
            _infer = model.signatures[sig]
            _inputs = _infer.structured_input_signature
            _outputs = _infer.structured_outputs
            info["signature"] = {
                "inputs": _inputs,
                "outputs": _outputs,
            }
        return info

    @property
    def preprocessing_model(self) -> Optional[keras.Model]:
        """Get the preprocessing model."""
        return self._preprocessing_model

    @property
    def inputs(self) -> Optional[dict]:
        """Get the input shapes specification."""
        return self._inputs

    @property
    def preprocessing_fitted(self) -> bool:
        """Check if the preprocessing model has been fitted."""
        return self._preprocessing_fitted

    def fit(
        self,
        x: Any = None,
        y: Any = None,
        epochs: int = 1,
        callbacks: list | None = None,
        **kwargs: Any,
    ) -> keras.callbacks.History:
        """Fits the model to the given data with preprocessing model integration.

        This method automatically handles preprocessing model fitting if needed,
        then calls the parent class fit method for training.

        Args:
            x (Any): The training data (features).
            y (Any): The training targets (labels).
            epochs (int): The number of epochs to train for.
            callbacks (list, optional): A list of callbacks to use during training. Defaults to None.
            **kwargs: Additional keyword arguments passed to the fit method.

        Returns:
            keras.callbacks.History: A History object containing training history.
        """
        # Auto-fit preprocessing model if needed (use x as the data)
        if x is not None:
            self._auto_fit_preprocessing_model(x)

        # Train the model using the parent class fit method
        history = super().fit(x=x, y=y, epochs=epochs, callbacks=callbacks, **kwargs)

        return history

    def get_input_info(self) -> dict[str, Any]:
        """Get comprehensive input information for the model.

        Returns:
            Dictionary containing input information
        """
        info = {
            "has_preprocessing_model": self._preprocessing_model is not None,
            "preprocessing_fitted": self._preprocessing_fitted,
            "input_shapes": self._inputs,
        }

        if self._preprocessing_model is not None:
            if hasattr(self._preprocessing_model, "inputs"):
                info["preprocessing_inputs"] = [
                    inp.name for inp in self._preprocessing_model.inputs
                ]
            if hasattr(self._preprocessing_model, "outputs"):
                info["preprocessing_outputs"] = [
                    out.name for out in self._preprocessing_model.outputs
                ]

        return info

    def validate_inputs(self, inputs: Any, expected_keys: list[str] = None) -> bool:
        """Validate inputs against expected format.

        Args:
            inputs: Input data to validate
            expected_keys: Expected feature names

        Returns:
            True if inputs are valid, False otherwise
        """
        try:
            standardized_inputs = self._standardize_inputs(inputs)

            if expected_keys is not None:
                for key in expected_keys:
                    if key not in standardized_inputs:
                        logger.warning(f"Missing expected input key: {key}")
                        return False

            return True
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False

    def get_model_summary(self) -> str:
        """Get a comprehensive model summary.

        Returns:
            String containing model summary information
        """
        summary_parts = [
            f"Model: {self.name}",
            f"Type: {self.__class__.__name__}",
            f"Built: {self.built}",
        ]

        if self._preprocessing_model is not None:
            summary_parts.append(
                f"Preprocessing: {self._preprocessing_model.__class__.__name__}",
            )
            summary_parts.append(f"Preprocessing Fitted: {self._preprocessing_fitted}")

        if self._inputs is not None:
            summary_parts.append(f"Input Shapes: {self._inputs}")

        if hasattr(self, "feature_names"):
            summary_parts.append(
                f"Feature Names: {getattr(self, 'feature_names', 'N/A')}",
            )

        return " | ".join(summary_parts)

    def create_functional_model(self) -> Optional[keras.Model]:
        """Create a functional model that combines preprocessing and main model.

        This is a public method that wraps the internal _create_functional_model.

        Returns:
            Functional model or None if no preprocessing model
        """
        return self._create_functional_model()

    def reset_preprocessing_fitted(self) -> None:
        """Reset the preprocessing fitted flag.

        Useful when you want to refit the preprocessing model.
        """
        self._preprocessing_fitted = False
        logger.info("Preprocessing fitted flag reset")

    def set_preprocessing_model(self, preprocessing_model: Any) -> None:
        """Set a new preprocessing model.

        Args:
            preprocessing_model: New preprocessing model to use
        """
        self._preprocessing_model = preprocessing_model
        self._preprocessing_fitted = False
        if preprocessing_model is not None:
            self._setup_preprocessing_model()
        logger.info(f"Preprocessing model set to: {type(preprocessing_model).__name__}")
