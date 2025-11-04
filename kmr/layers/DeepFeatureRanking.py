"""Deep feature ranking layer for recommendations."""

from typing import Any
from keras import layers
from keras import KerasTensor
from keras.saving import register_keras_serializable
from keras.regularizers import l2

from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class DeepFeatureRanking(BaseLayer):
    """Deep ranking network that scores items based on combined features.

    Implements deep neural network for ranking that processes combined
    user/item/context features to produce ranking scores.

    Args:
        hidden_dim: Hidden dimension (default=32).
        l2_reg: L2 regularization coefficient (default=1e-6).
        dropout_rate: Dropout rate (default=0.2).
        name: Optional name for the layer.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        l2_reg: float = 1e-6,
        dropout_rate: float = 0.2,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize layer."""
        self._hidden_dim = hidden_dim
        self._l2_reg = float(l2_reg)
        self._dropout_rate = float(dropout_rate)

        self._validate_params()

        self.hidden_dim = self._hidden_dim
        self.l2_reg = self._l2_reg
        self.dropout_rate = self._dropout_rate
        self.dense1 = None
        self.dense2 = None
        self.dropout = None
        self.batch_norm = None

        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate parameters."""
        if not isinstance(self._hidden_dim, int) or self._hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self._hidden_dim}")

    def build(self, input_shape: tuple) -> None:
        """Build layer."""
        self.dense1 = layers.Dense(
            self._hidden_dim,
            activation="relu",
            kernel_regularizer=l2(self._l2_reg),
        )
        self.batch_norm = layers.BatchNormalization()
        self.dropout = layers.Dropout(self._dropout_rate)
        self.dense2 = layers.Dense(1, activation="linear")

        super().build(input_shape)

    def call(self, inputs: KerasTensor, training: bool | None = None) -> KerasTensor:
        """Forward pass."""
        x = self.dense1(inputs)
        x = self.batch_norm(x, training=training)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x

    def get_config(self) -> dict[str, Any]:
        """Get configuration."""
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "l2_reg": self.l2_reg,
                "dropout_rate": self.dropout_rate,
            },
        )
        return config
