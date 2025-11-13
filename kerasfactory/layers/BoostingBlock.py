"""This module implements a BoostingBlock layer that simulates gradient boosting behavior in a neural network.
The layer computes a correction term via a configurable MLP and adds a scaled version to the input.
"""

from typing import Any
from loguru import logger
from keras import layers, initializers
from keras import KerasTensor
from kerasfactory.layers._base_layer import BaseLayer
from keras.saving import register_keras_serializable


@register_keras_serializable(package="kerasfactory.layers")
class BoostingBlock(BaseLayer):
    """A neural network layer that simulates gradient boosting behavior.

    This layer implements a weak learner that computes a correction term via a configurable MLP
    and adds a scaled version of this correction to the input. Stacking several such blocks
    can mimic the iterative residual-correction process of gradient boosting.

    The output is computed as:
        output = inputs + gamma * f(inputs)
    where:
        - f is a configurable MLP (default: two-layer network)
        - gamma is a learnable or fixed scaling factor

    Args:
        hidden_units: Number of units in the hidden layer(s). Can be an int for single hidden layer
            or a list of ints for multiple hidden layers. Default is 64.
        hidden_activation: Activation function for hidden layers. Default is 'relu'.
        output_activation: Activation function for the output layer. Default is None.
        gamma_trainable: Whether the scaling factor gamma is trainable. Default is True.
        gamma_initializer: Initializer for the gamma scaling factor. Default is 'ones'.
        use_bias: Whether to include bias terms in the dense layers. Default is True.
        kernel_initializer: Initializer for the dense layer kernels. Default is 'glorot_uniform'.
        bias_initializer: Initializer for the dense layer biases. Default is 'zeros'.
        dropout_rate: Optional dropout rate to apply after hidden layers. Default is None.
        name: Optional name for the layer.

    Input shape:
        N-D tensor with shape: (batch_size, ..., input_dim)

    Output shape:
        Same shape as input: (batch_size, ..., input_dim)

    Example:
        ```python
        import tensorflow as tf
        from kerasfactory.layers import BoostingBlock

        # Create sample input data
        x = tf.random.normal((32, 16))  # 32 samples, 16 features

        # Basic usage
        block = BoostingBlock(hidden_units=64)
        y = block(x)
        print("Output shape:", y.shape)  # (32, 16)

        # Advanced configuration
        block = BoostingBlock(
            hidden_units=[32, 16],  # Two hidden layers
            hidden_activation='selu',
            dropout_rate=0.1,
            gamma_trainable=False
        )
        y = block(x)
        ```
    """

    def __init__(
        self,
        hidden_units: int | list[int] = 64,
        hidden_activation: str = "relu",
        output_activation: str | None = None,
        gamma_trainable: bool = True,
        gamma_initializer: str | initializers.Initializer = "ones",
        use_bias: bool = True,
        kernel_initializer: str | initializers.Initializer = "glorot_uniform",
        bias_initializer: str | initializers.Initializer = "zeros",
        dropout_rate: float | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the BoostingBlock layer.

        Args:
            hidden_units: Number of hidden units or list of units per layer.
            hidden_activation: Activation function for hidden layers.
            output_activation: Activation function for output layer.
            gamma_trainable: Whether gamma parameter is trainable.
            gamma_initializer: Initializer for gamma parameter.
            use_bias: Whether to use bias.
            kernel_initializer: Initializer for kernel weights.
            bias_initializer: Initializer for bias weights.
            dropout_rate: Dropout rate.
            name: Name of the layer.
            **kwargs: Additional keyword arguments.
        """
        # Set attributes before calling parent's __init__
        self._hidden_units = (
            [hidden_units] if isinstance(hidden_units, int) else hidden_units
        )
        self._hidden_activation = hidden_activation
        self._output_activation = output_activation
        self._gamma_trainable = gamma_trainable
        self._gamma_initializer = initializers.get(gamma_initializer)
        self._use_bias = use_bias
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._bias_initializer = initializers.get(bias_initializer)
        self._dropout_rate = dropout_rate

        # Validate parameters
        if any(units <= 0 for units in self._hidden_units):
            raise ValueError("All hidden_units must be positive integers")
        if dropout_rate is not None and not 0 <= dropout_rate < 1:
            raise ValueError("dropout_rate must be between 0 and 1")

        super().__init__(name=name, **kwargs)

        # Now set public attributes
        self.hidden_units = self._hidden_units
        self.hidden_activation = self._hidden_activation
        self.output_activation = self._output_activation
        self.gamma_trainable = self._gamma_trainable
        self.gamma_initializer = self._gamma_initializer
        self.use_bias = self._use_bias
        self.kernel_initializer = self._kernel_initializer
        self.bias_initializer = self._bias_initializer
        self.dropout_rate = self._dropout_rate

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Builds the layer with the given input shape.

        Args:
            input_shape: Tuple of integers defining the input shape.
        """
        # Build hidden layers
        self.hidden_layers = []
        for i, units in enumerate(self.hidden_units):
            self.hidden_layers.append(
                layers.Dense(
                    units=units,
                    activation=self.hidden_activation,
                    use_bias=self.use_bias,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    name=f"hidden_{i+1}",
                ),
            )
            if self.dropout_rate is not None:
                self.hidden_layers.append(
                    layers.Dropout(rate=self.dropout_rate, name=f"dropout_{i+1}"),
                )

        # Build output layer
        self.output_layer = layers.Dense(
            units=input_shape[-1],
            activation=self.output_activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="output",
        )

        # Learnable scaling factor
        self.gamma = self.add_weight(
            name="gamma",
            shape=(1,),
            initializer=self.gamma_initializer,
            trainable=self.gamma_trainable,
        )

        logger.debug(
            f"BoostingBlock built with hidden_units={self.hidden_units}, "
            f"activation={self.hidden_activation}, dropout={self.dropout_rate}",
        )
        super().build(input_shape)

    def call(self, inputs: KerasTensor, training: bool | None = None) -> KerasTensor:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Only relevant when using dropout.

        Returns:
            Output tensor of same shape as input.
        """
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        correction = self.output_layer(x)
        return inputs + self.gamma * correction

    def get_config(self) -> dict[str, Any]:
        """Returns the config of the layer.

        Returns:
            Python dictionary containing the layer configuration.
        """
        config = super().get_config()

        # Use private attributes during initialization, public attributes after
        hidden_units = getattr(
            self,
            "hidden_units",
            getattr(self, "_hidden_units", None),
        )
        hidden_activation = getattr(
            self,
            "hidden_activation",
            getattr(self, "_hidden_activation", None),
        )
        output_activation = getattr(
            self,
            "output_activation",
            getattr(self, "_output_activation", None),
        )
        gamma_trainable = getattr(
            self,
            "gamma_trainable",
            getattr(self, "_gamma_trainable", None),
        )
        gamma_initializer = getattr(
            self,
            "gamma_initializer",
            getattr(self, "_gamma_initializer", None),
        )
        use_bias = getattr(self, "use_bias", getattr(self, "_use_bias", None))
        kernel_initializer = getattr(
            self,
            "kernel_initializer",
            getattr(self, "_kernel_initializer", None),
        )
        bias_initializer = getattr(
            self,
            "bias_initializer",
            getattr(self, "_bias_initializer", None),
        )
        dropout_rate = getattr(
            self,
            "dropout_rate",
            getattr(self, "_dropout_rate", None),
        )

        config.update(
            {
                "hidden_units": hidden_units,
                "hidden_activation": hidden_activation,
                "output_activation": output_activation,
                "gamma_trainable": gamma_trainable,
                "gamma_initializer": initializers.serialize(gamma_initializer)
                if gamma_initializer
                else None,
                "use_bias": use_bias,
                "kernel_initializer": initializers.serialize(kernel_initializer)
                if kernel_initializer
                else None,
                "bias_initializer": initializers.serialize(bias_initializer)
                if bias_initializer
                else None,
                "dropout_rate": dropout_rate,
            },
        )
        return config
