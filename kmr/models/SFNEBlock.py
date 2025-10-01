"""This module implements a SFNEBlock (Slow-Fast Neural Engine Block) model that combines
slow and fast processing paths for feature extraction. It's a building block for the Terminator model.
"""

from typing import Any
from loguru import logger
from keras import layers, ops, Model
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kmr.models._base import BaseModel
from kmr.layers.SlowNetwork import SlowNetwork
from kmr.layers.HyperZZWOperator import HyperZZWOperator


@register_keras_serializable(package="kmr.models")
class SFNEBlock(BaseModel):
    """Slow-Fast Neural Engine Block for feature processing.

    This model combines a slow network path and a fast processing path to extract
    features. It uses a SlowNetwork to generate hyper-kernels, which are then used
    by a HyperZZWOperator to compute context-dependent weights. These weights are
    further processed by global and local convolutions before being combined.

    Args:
        input_dim: Dimension of the input features.
        output_dim: Dimension of the output features. Default is same as input_dim.
        hidden_dim: Number of hidden units in the network. Default is 64.
        num_layers: Number of layers in the network. Default is 2.
        slow_network_layers: Number of layers in the slow network. Default is 3.
        slow_network_units: Number of units per layer in the slow network. Default is 128.
        preprocessing_model: Optional preprocessing model to apply before the main processing.
        name: Optional name for the model.

    Input shape:
        2D tensor with shape: `(batch_size, input_dim)` or a dictionary with feature inputs

    Output shape:
        2D tensor with shape: `(batch_size, output_dim)`

    Example:
        ```python
        import keras
        from kmr.models import SFNEBlock

        # Create sample input data
        x = keras.random.normal((32, 16))  # 32 samples, 16 features

        # Create the model
        sfne = SFNEBlock(input_dim=16, output_dim=8)
        y = sfne(x)
        print("Output shape:", y.shape)  # (32, 8)
        ```
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = None,
        hidden_dim: int = 64,
        num_layers: int = 2,
        slow_network_layers: int = 3,
        slow_network_units: int = 128,
        preprocessing_model: Model | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Extract our specific parameters before calling parent's __init__
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.slow_network_layers = slow_network_layers
        self.slow_network_units = slow_network_units
        self.preprocessing_model = preprocessing_model

        # Call parent's __init__ with only the parameters it accepts
        super().__init__(name=name, **kwargs)

        # Validate parameters
        self._validate_params()

        # Create layers
        self.input_layer = layers.Dense(self.hidden_dim, activation="relu")
        self.hidden_layers = [
            layers.Dense(self.hidden_dim, activation="relu")
            for _ in range(self.num_layers)
        ]
        self.slow_network = SlowNetwork(
            input_dim=input_dim,
            num_layers=slow_network_layers,
            units=slow_network_units,
        )
        self.hyper_zzw = HyperZZWOperator(input_dim=self.hidden_dim)
        self.global_conv = layers.Conv1D(input_dim, kernel_size=1, activation="relu")
        self.local_conv = layers.Conv1D(
            input_dim,
            kernel_size=3,
            padding="same",
            activation="relu",
        )
        self.bottleneck = layers.Dense(input_dim, activation="relu")
        self.output_layer = layers.Dense(self.output_dim, activation="linear")

    def _validate_params(self) -> None:
        """Validate model parameters."""
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {self.output_dim}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.slow_network_layers <= 0:
            raise ValueError(
                f"slow_network_layers must be positive, got {self.slow_network_layers}",
            )
        if self.slow_network_units <= 0:
            raise ValueError(
                f"slow_network_units must be positive, got {self.slow_network_units}",
            )

    def call(
        self,
        inputs: dict[str, KerasTensor] | KerasTensor,
        training: bool = False,
    ) -> KerasTensor:
        """Forward pass of the model.

        Args:
            inputs: Input tensor with shape (batch_size, input_dim) or a dictionary with feature inputs.
            training: Boolean indicating whether the model should behave in training mode.

        Returns:
            Output tensor with shape (batch_size, output_dim).
        """
        # Handle dictionary inputs if provided
        if isinstance(inputs, dict):
            # Filter inputs to only include those expected by the model
            inputs = self.filer_inputs(inputs)
            # Convert dictionary to a list of tensors
            x = ops.concatenate(list(inputs.values()), axis=-1)
        else:
            x = inputs

        # Apply preprocessing if available
        if self.preprocessing_model is not None:
            x = self.preprocessing_model(x, training=training)

        logger.debug(f"SFNEBlock input shape: {x.shape}")

        # Input layer
        x = self.input_layer(x)

        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x)

        # Generate hyper-kernels using the slow network
        slow_input = (
            ops.concatenate(list(inputs.values()), axis=-1)
            if isinstance(inputs, dict)
            else inputs
        )

        hyper_kernels = self.slow_network(slow_input, training=training)
        logger.debug(f"SFNEBlock hyper_kernels shape: {hyper_kernels.shape}")

        # Compute context-dependent weights
        context_weights = self.hyper_zzw([x, hyper_kernels])
        logger.debug(f"SFNEBlock context_weights shape: {context_weights.shape}")

        # Expand dimensions for convolution
        context_weights_expanded = ops.expand_dims(context_weights, axis=-1)
        logger.debug(
            f"SFNEBlock context_weights_expanded shape: {context_weights_expanded.shape}",
        )

        # Global convolution
        global_output = self.global_conv(context_weights_expanded)
        logger.debug(f"SFNEBlock global_output shape: {global_output.shape}")

        # Local convolution
        local_output = self.local_conv(context_weights_expanded)
        logger.debug(f"SFNEBlock local_output shape: {local_output.shape}")

        # Concatenate the outputs along the last axis
        combined_output = ops.concatenate([global_output, local_output], axis=-1)
        logger.debug(f"SFNEBlock combined_output shape: {combined_output.shape}")

        # Flatten the combined output
        combined_output_flat = ops.reshape(
            combined_output,
            [-1, combined_output.shape[1] * combined_output.shape[2]],
        )
        logger.debug(
            f"SFNEBlock combined_output_flat shape: {combined_output_flat.shape}",
        )

        # Bottleneck layer
        bottleneck_output = self.bottleneck(combined_output_flat)
        logger.debug(f"SFNEBlock bottleneck_output shape: {bottleneck_output.shape}")

        # Output layer
        output = self.output_layer(bottleneck_output)
        logger.debug(f"SFNEBlock output_layer output shape: {output.shape}")

        return output

    def get_config(self) -> dict[str, Any]:
        """Returns the config of the model.

        Returns:
            Python dictionary containing the model configuration.
        """
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "slow_network_layers": self.slow_network_layers,
                "slow_network_units": self.slow_network_units,
                "preprocessing_model": self.preprocessing_model,
            },
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SFNEBlock":
        """Creates a model from its configuration.

        Args:
            config: Dictionary containing the model configuration.

        Returns:
            A new instance of the model.
        """
        # Extract preprocessing model if present
        preprocessing_model = config.pop("preprocessing_model", None)

        # Create model instance
        return cls(preprocessing_model=preprocessing_model, **config)
