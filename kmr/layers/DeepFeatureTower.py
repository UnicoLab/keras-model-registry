"""Recommendation tower layer for feature processing.

Dense neural network tower for processing user or item features in
two-tower recommendation architectures.
"""

from typing import Any
from keras import layers
from keras import KerasTensor
from keras.saving import register_keras_serializable
from keras.regularizers import l2

from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class DeepFeatureTower(BaseLayer):
    """Dense feature tower for user/item feature processing.

    Implements a stack of dense layers with batch normalization and dropout
    for processing user or item features in a two-tower recommendation model.

    Args:
        units: Output dimension (default=32).
        hidden_layers: Number of hidden layers (default=2).
        dropout_rate: Dropout rate between layers (default=0.2).
        l2_reg: L2 regularization coefficient (default=1e-6).
        activation: Activation function (default='relu').
        name: Optional name for the layer.

    Input shape:
        (batch_size, input_dim) - Feature vectors

    Output shape:
        (batch_size, units) - Processed feature vectors

    Example:
        ```python
        import keras
        from kmr.layers import DeepFeatureTower

        features = keras.random.normal((32, 100))
        tower = DeepFeatureTower(units=32, hidden_layers=2)
        output = tower(features)
        print("Output shape:", output.shape)  # (32, 32)
        ```
    """

    def __init__(
        self,
        units: int = 32,
        hidden_layers: int = 2,
        dropout_rate: float = 0.2,
        l2_reg: float = 1e-6,
        activation: str = "relu",
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the DeepFeatureTower layer.

        Args:
            units: Output dimension.
            hidden_layers: Number of hidden layers.
            dropout_rate: Dropout rate.
            l2_reg: L2 regularization coefficient.
            activation: Activation function.
            name: Name of the layer.
            **kwargs: Additional keyword arguments.
        """
        self._units = units
        self._hidden_layers = hidden_layers
        self._dropout_rate = float(dropout_rate)
        self._l2_reg = float(l2_reg)
        self._activation = activation

        self._validate_params()

        self.units = self._units
        self.hidden_layers = self._hidden_layers
        self.dropout_rate = self._dropout_rate
        self.l2_reg = self._l2_reg
        self.activation = self._activation
        self.dense_layers = None
        self.batch_norms = None
        self.dropouts = None

        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if not isinstance(self._units, int) or self._units <= 0:
            raise ValueError(f"units must be positive integer, got {self._units}")
        if not isinstance(self._hidden_layers, int) or self._hidden_layers < 1:
            raise ValueError(f"hidden_layers must be >= 1, got {self._hidden_layers}")
        if not (0 <= self._dropout_rate < 1):
            raise ValueError(
                f"dropout_rate must be in [0, 1), got {self._dropout_rate}",
            )

    def build(self, input_shape: tuple) -> None:
        """Build layer with given input shape.

        Args:
            input_shape: Input shape tuple.
        """
        self.dense_layers = []
        self.batch_norms = []
        self.dropouts = []

        for _ in range(self._hidden_layers):
            dense = layers.Dense(
                self._units,
                activation=self._activation,
                kernel_regularizer=l2(self._l2_reg),
            )
            self.dense_layers.append(dense)
            self.batch_norms.append(layers.BatchNormalization())
            self.dropouts.append(layers.Dropout(self._dropout_rate))

        super().build(input_shape)

    def call(self, inputs: KerasTensor, training: bool | None = None) -> KerasTensor:
        """Process features through tower.

        Args:
            inputs: Input feature tensor.
            training: Whether in training mode.

        Returns:
            Processed feature tensor.
        """
        x = inputs
        for i in range(self._hidden_layers):
            x = self.dense_layers[i](x)
            x = self.batch_norms[i](x, training=training)
            x = self.dropouts[i](x, training=training)
        return x

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "hidden_layers": self.hidden_layers,
                "dropout_rate": self.dropout_rate,
                "l2_reg": self.l2_reg,
                "activation": self.activation,
            },
        )
        return config
