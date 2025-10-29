"""Feed forward neural network model implementation."""
from typing import Any
import keras
from keras import layers, Model
from keras import KerasTensor
from keras.saving import register_keras_serializable
from loguru import logger

from ._base import BaseModel


@register_keras_serializable(package="kmr")
class BaseFeedForwardModel(BaseModel):
    """Base feed forward neural network model.

    This model implements a basic feed forward neural network with configurable
    hidden layers, activations, and regularization options.

    Example:
        ```python
        # Create a simple feed forward model
        model = BaseFeedForwardModel(
            feature_names=['feature1', 'feature2'],
            hidden_units=[64, 32],
            output_units=1
        )

        # Compile and train the model
        model.compile(optimizer='adam', loss='mse')
        model.fit(train_dataset, epochs=10)
        ```
    """

    def __init__(
        self,
        feature_names: list[str],
        hidden_units: list[int],
        output_units: int = 1,
        dropout_rate: float = 0.0,
        activation: str = "relu",
        preprocessing_model: Model | None = None,
        kernel_initializer: str | Any | None = "glorot_uniform",
        bias_initializer: str | Any | None = "zeros",
        kernel_regularizer: str | Any | None = None,
        bias_regularizer: str | Any | None = None,
        activity_regularizer: str | Any | None = None,
        kernel_constraint: str | Any | None = None,
        bias_constraint: str | Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Feed Forward Neural Network.

        Args:
            feature_names: list of feature names.
            hidden_units: list of hidden layer units.
            output_units: Number of output units.
            dropout_rate: Dropout rate.
            activation: Activation function.
            preprocessing_model: Optional preprocessing model.
            kernel_initializer: Weight initializer.
            bias_initializer: Bias initializer.
            kernel_regularizer: Weight regularizer.
            bias_regularizer: Bias regularizer.
            activity_regularizer: Activity regularizer.
            kernel_constraint: Weight constraint.
            bias_constraint: Bias constraint.
            **kwargs: Additional arguments.
        """
        super().__init__(preprocessing_model=preprocessing_model, **kwargs)

        # Store model parameters
        self.feature_names = feature_names
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        logger.info("🏗️ Initializing Feed Forward Neural Network")
        logger.info(f"📊 Model Architecture: {hidden_units} -> {output_units}")
        logger.info(f"🔄 Input Features: {feature_names}")

        # Create input layers
        self.input_layers = {}
        for name in feature_names:
            self.input_layers[name] = layers.Input(shape=(1,), name=name)
        logger.debug(f"✨ Created input layers for features: {feature_names}")

        # Build model layers
        self.concat_layer = layers.Concatenate(axis=1)
        logger.debug("✨ Created concatenation layer")

        # Add hidden layers
        self.hidden_layers = []
        for i, units in enumerate(hidden_units, 1):
            logger.debug(f"✨ Adding hidden layer {i} with {units} units")
            dense = layers.Dense(
                units=units,
                activation=activation,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                name=f"hidden_{i}",
            )
            self.hidden_layers.append(dense)

            # Add dropout if specified
            if dropout_rate > 0:
                dropout = layers.Dropout(rate=dropout_rate)
                self.hidden_layers.append(dropout)

        # Add output layer
        logger.debug(f"✨ Adding output layer with {output_units} units")
        self.output_layer = layers.Dense(
            units=output_units,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name="output",
        )

        # Build the model
        self.build_model()

    def build_model(self) -> None:
        """Build the model architecture."""
        # Create inputs
        inputs = [self.input_layers[name] for name in self.feature_names]

        # Apply preprocessing if available
        if self.preprocessing_model is not None:
            # Check if preprocessing model expects multiple inputs (like KDP)
            if hasattr(self.preprocessing_model, 'inputs') and len(self.preprocessing_model.inputs) > 1:
                # KDP-style preprocessing: pass individual inputs
                x = self.preprocessing_model(inputs)
            else:
                # Standard preprocessing: concatenate first, then apply
                x = self.concat_layer(inputs)
                x = self.preprocessing_model(x)
        else:
            # No preprocessing: just concatenate
            x = self.concat_layer(inputs)

        # Apply hidden layers
        for layer in self.hidden_layers:
            x = layer(x)

        # Apply output layer
        outputs = self.output_layer(x)

        # Create model with appropriate input structure
        if (self.preprocessing_model is not None and 
            hasattr(self.preprocessing_model, 'inputs') and 
            len(self.preprocessing_model.inputs) > 1):
            # For KDP-style preprocessing, create model with named inputs
            input_dict = {name: self.input_layers[name] for name in self.feature_names}
            self._model = Model(inputs=input_dict, outputs=outputs)
        else:
            # Standard model with list of inputs
            self._model = Model(inputs=inputs, outputs=outputs)

    def call(
        self,
        inputs: Any,
        training: bool = False,
    ) -> KerasTensor:
        """Forward pass of the model with universal input handling.

        This method supports various input formats:
        - Single tensors/vectors (numpy arrays, tensors)
        - Lists/tuples of tensors
        - Dictionaries (regular dict, OrderedDict)
        - Mixed input formats

        Args:
            inputs: Input data in various formats (dict, list, tensor, etc.)
            training: Whether in training mode.

        Returns:
            Model output tensor.
        """
        # Use BaseModel's intelligent input processing
        processed_inputs = self._process_inputs_for_model(
            inputs, 
            expected_keys=self.feature_names,
            auto_split=True,
            auto_reshape=True
        )
        
        # Pass through internal model
        if self.preprocessing_model is not None:
            # For preprocessed inputs, we need to check if inputs are already preprocessed
            # If inputs is a single tensor (preprocessed), apply hidden layers directly
            if isinstance(processed_inputs, (list, tuple)) and len(processed_inputs) == 1:
                # Single preprocessed tensor
                x = processed_inputs[0]
                for layer in self.hidden_layers:
                    x = layer(x)
                return self.output_layer(x)
            elif hasattr(processed_inputs, 'shape') and len(processed_inputs.shape) == 2:
                # Single preprocessed tensor (not in a list)
                x = processed_inputs
                for layer in self.hidden_layers:
                    x = layer(x)
                return self.output_layer(x)
            else:
                # Multiple inputs or dictionary - use internal model
                return self._model(processed_inputs, training=training)
        else:
            # Use the internal model for raw inputs
            return self._model(processed_inputs, training=training)

    def get_config(self) -> dict[str, Any]:
        """Get model configuration.

        Returns:
            Dict containing model configuration.
        """
        config = super().get_config()
        config.update(
            {
                "feature_names": self.feature_names,
                "hidden_units": self.hidden_units,
                "output_units": self.output_units,
                "dropout_rate": self.dropout_rate,
                "activation": self.activation,
                "preprocessing_model": self.preprocessing_model,
                "kernel_initializer": self.kernel_initializer,
                "bias_initializer": self.bias_initializer,
                "kernel_regularizer": self.kernel_regularizer,
                "bias_regularizer": self.bias_regularizer,
                "activity_regularizer": self.activity_regularizer,
                "kernel_constraint": self.kernel_constraint,
                "bias_constraint": self.bias_constraint,
            },
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BaseFeedForwardModel":
        """Create model from configuration.

        Args:
            config: Dict containing model configuration.

        Returns:
            Instantiated model.
        """
        # Extract preprocessing model if present
        preprocessing_model = config.pop("preprocessing_model", None)
        
        # Deserialize preprocessing model if it's a config dict
        if preprocessing_model is not None and isinstance(preprocessing_model, dict):
            from keras.saving import deserialize_keras_object
            preprocessing_model = deserialize_keras_object(preprocessing_model)

        # Create model instance
        return cls(preprocessing_model=preprocessing_model, **config)
