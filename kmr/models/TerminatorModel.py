"""This module implements a TerminatorModel that combines multiple SFNE blocks for
advanced feature processing. It's designed for complex tabular data modeling tasks.
"""

from typing import Any
from loguru import logger
from keras import layers, Model
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kmr.models._base import BaseModel
from kmr.models.SFNEBlock import SFNEBlock
from kmr.layers.HyperZZWOperator import HyperZZWOperator
from kmr.layers.SlowNetwork import SlowNetwork


@register_keras_serializable(package="kmr.models")
class TerminatorModel(BaseModel):
    """Terminator model for advanced feature processing.

    This model stacks multiple SFNE blocks to process features in a hierarchical manner.
    It's designed for complex tabular data modeling tasks where feature interactions
    are important.

    Args:
        input_dim: Dimension of the input features.
        context_dim: Dimension of the context features.
        output_dim: Dimension of the output.
        hidden_dim: Number of hidden units in the network. Default is 64.
        num_layers: Number of layers in the network. Default is 2.
        num_blocks: Number of SFNE blocks to stack. Default is 3.
        slow_network_layers: Number of layers in each slow network. Default is 3.
        slow_network_units: Number of units per layer in each slow network. Default is 128.
        preprocessing_model: Optional preprocessing model to apply before the main processing.
        name: Optional name for the model.

    Input shape:
        List of 2D tensors with shapes: `[(batch_size, input_dim), (batch_size, context_dim)]`

    Output shape:
        2D tensor with shape: `(batch_size, output_dim)`

    Example:
        ```python
        import keras
        from kmr.models import TerminatorModel

        # Create sample input data
        x = keras.random.normal((32, 16))  # 32 samples, 16 features
        context = keras.random.normal((32, 8))  # 32 samples, 8 context features

        # Create the model
        terminator = TerminatorModel(input_dim=16, context_dim=8, output_dim=1)
        y = terminator([x, context])
        print("Output shape:", y.shape)  # (32, 1)
        ```
    """

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_blocks: int = 3,
        slow_network_layers: int = 3,
        slow_network_units: int = 128,
        preprocessing_model: Model | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Extract our specific parameters before calling parent's __init__
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.slow_network_layers = slow_network_layers
        self.slow_network_units = slow_network_units
        self.preprocessing_model = preprocessing_model

        # Call parent's __init__ with only the parameters it accepts
        super().__init__(name=name, **kwargs)

        # Validate parameters
        self._validate_params()

        # Create layers
        self.input_layer = layers.Dense(input_dim, activation="relu")
        self.slow_network = SlowNetwork(
            input_dim=context_dim,
            num_layers=slow_network_layers,
            units=slow_network_units,
        )
        self.hyper_zzw = HyperZZWOperator(input_dim=input_dim, context_dim=context_dim)
        self.sfne_blocks = [
            SFNEBlock(
                input_dim=input_dim,
                output_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                slow_network_layers=slow_network_layers,
                slow_network_units=slow_network_units,
            )
            for _ in range(num_blocks)
        ]
        self.output_layer = layers.Dense(output_dim, activation="linear")

        # Add a context-dependent layer to ensure context affects output
        self.context_dense = layers.Dense(input_dim, activation="relu")

    def _validate_params(self) -> None:
        """Validate model parameters."""
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.context_dim <= 0:
            raise ValueError(f"context_dim must be positive, got {self.context_dim}")
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {self.output_dim}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {self.num_blocks}")
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
        inputs: list[KerasTensor] | KerasTensor,
        training: bool = False,
    ) -> KerasTensor:
        """Forward pass of the model.

        Args:
            inputs: List of input tensors [input_tensor, context_tensor] or a single input tensor.
            training: Boolean indicating whether the model should behave in training mode.

        Returns:
            Output tensor with shape (batch_size, output_dim).
        """
        # Handle inputs
        if isinstance(inputs, list):
            x, context = inputs
        else:
            raise ValueError(
                "TerminatorModel expects a list of inputs [input_tensor, context_tensor]",
            )

        # Apply preprocessing if available
        if self.preprocessing_model is not None:
            x = self.preprocessing_model(x, training=training)

        logger.debug(
            f"TerminatorModel input shape: {x.shape}, context shape: {context.shape}",
        )

        # Input layer
        x = self.input_layer(x)
        logger.debug(f"TerminatorModel input_layer output shape: {x.shape}")

        # Generate hyper-kernels using the slow network
        hyper_kernels = self.slow_network(context, training=training)
        logger.debug(f"TerminatorModel hyper_kernels shape: {hyper_kernels.shape}")

        # Compute context-dependent weights
        context_weights = self.hyper_zzw([x, hyper_kernels])
        logger.debug(f"TerminatorModel context_weights shape: {context_weights.shape}")

        # Process context to ensure it affects the output
        self.context_dense(context)

        # SFNE blocks
        for i, sfne_block in enumerate(self.sfne_blocks):
            x = sfne_block(x, training=training)
            logger.debug(f"TerminatorModel SFNEBlock {i} output shape: {x.shape}")

        # Combine with context features to ensure context dependency
        x = x * context_weights

        # Output layer
        output = self.output_layer(x)
        logger.debug(f"TerminatorModel output_layer output shape: {output.shape}")

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
                "context_dim": self.context_dim,
                "output_dim": self.output_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_blocks": self.num_blocks,
                "slow_network_layers": self.slow_network_layers,
                "slow_network_units": self.slow_network_units,
                "preprocessing_model": self.preprocessing_model,
            },
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TerminatorModel":
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
