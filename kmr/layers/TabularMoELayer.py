"""This module implements a TabularMoELayer (Mixture-of-Experts) that routes input features
through multiple expert sub-networks and aggregates their outputs via a learnable gating mechanism.
This approach is useful for tabular data where different experts can specialize in different feature patterns.
"""

from typing import Any
from loguru import logger
from keras import layers, ops
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class TabularMoELayer(BaseLayer):
    """Mixture-of-Experts layer for tabular data.

    This layer routes input features through multiple expert sub-networks and
    aggregates their outputs via a learnable gating mechanism. Each expert is a small
    MLP, and the gate learns to weight their contributions.

    Args:
        num_experts: Number of expert networks. Default is 4.
        expert_units: Number of hidden units in each expert network. Default is 16.
        name: Optional name for the layer.

    Input shape:
        2D tensor with shape: `(batch_size, num_features)`

    Output shape:
        2D tensor with shape: `(batch_size, num_features)` (same as input)

    Example:
        ```python
        import keras
        from kmr.layers import TabularMoELayer

        # Tabular data with 8 features
        x = keras.random.normal((32, 8))

        # Create the layer with 4 experts and 16 units per expert
        moe_layer = TabularMoELayer(num_experts=4, expert_units=16)
        y = moe_layer(x)
        print("MoE output shape:", y.shape)  # Expected: (32, 8)
        ```
    """

    def __init__(
        self,
        num_experts: int = 4,
        expert_units: int = 16,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the TabularMoELayer.

        Args:
            num_experts: Number of expert networks.
            expert_units: Number of units in each expert.
            name: Name of the layer.
            **kwargs: Additional keyword arguments.
        """
        # Set public attributes
        self.num_experts = num_experts
        self.expert_units = expert_units

        # Initialize instance variables
        self.experts: list[Any] | None = None
        self.expert_outputs: list[Any] | None = None
        self.gate: Any | None = None

        # Validate parameters during initialization
        self._validate_params()

        # Call parent's __init__
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if self.num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {self.num_experts}")
        if self.expert_units <= 0:
            raise ValueError(f"expert_units must be positive, got {self.expert_units}")

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Builds the layer with the given input shape.

        Args:
            input_shape: Tuple of integers defining the input shape.
        """
        # Validate parameters again during build
        self._validate_params()

        # Create expert networks
        self.experts = [
            layers.Dense(self.expert_units, activation="relu", name=f"expert_{i}")
            for i in range(self.num_experts)
        ]

        # Create expert output projections
        self.expert_outputs = [
            layers.Dense(input_shape[-1], activation=None, name=f"expert_out_{i}")
            for i in range(self.num_experts)
        ]

        # Create gating network
        self.gate = layers.Dense(self.num_experts, activation="softmax", name="gate")

        logger.debug(
            f"TabularMoELayer built with num_experts={self.num_experts}, expert_units={self.expert_units}",
        )
        super().build(input_shape)

    def call(self, inputs: KerasTensor, _: bool | None = None) -> KerasTensor:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor with shape (batch_size, num_features).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Output tensor with the same shape as input.
        """
        expert_outputs = []
        for i in range(self.num_experts):
            x = self.experts[i](inputs)
            x = self.expert_outputs[i](x)
            expert_outputs.append(x)

        # Stack outputs: shape (batch, num_experts, features)
        experts_stack = ops.stack(expert_outputs, axis=1)

        # Compute gating weights: shape (batch, num_experts)
        gate_weights = self.gate(inputs)

        # Reshape for broadcasting: (batch, num_experts, 1)
        gate_weights = ops.expand_dims(gate_weights, axis=-1)

        # Aggregate expert outputs with gating weights
        output = ops.sum(experts_stack * gate_weights, axis=1)

        return output

    def get_config(self) -> dict[str, Any]:
        """Returns the config of the layer.

        Returns:
            Python dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "num_experts": self.num_experts,
                "expert_units": self.expert_units,
            },
        )
        return config
