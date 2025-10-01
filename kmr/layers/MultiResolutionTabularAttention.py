"""This module implements a MultiResolutionTabularAttention layer that applies separate attention
mechanisms for numerical and categorical features, along with cross-attention between them.
It's particularly useful for mixed-type tabular data.
"""

from typing import Any
from loguru import logger
from keras import layers
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class MultiResolutionTabularAttention(BaseLayer):
    """Custom layer to apply multi-resolution attention for mixed-type tabular data.

    This layer implements separate attention mechanisms for numerical and categorical features,
    along with cross-attention between them. It's designed to handle the different characteristics
    of numerical and categorical features in tabular data.

    Args:
        num_heads (int): Number of attention heads
        d_model (int): Dimensionality of the attention model
        dropout_rate (float): Dropout rate for regularization
        name (str, optional): Name for the layer

    Input shape:
        List of two tensors:
        - Numerical features: `(batch_size, num_samples, num_numerical_features)`
        - Categorical features: `(batch_size, num_samples, num_categorical_features)`

    Output shape:
        List of two tensors with shapes:
        - `(batch_size, num_samples, d_model)` (numerical features)
        - `(batch_size, num_samples, d_model)` (categorical features)

    Example:
        ```python
        import keras
        from kmr.layers import MultiResolutionTabularAttention

        # Create sample input data
        numerical = keras.random.normal((32, 100, 10))  # 32 batches, 100 samples, 10 numerical features
        categorical = keras.random.normal((32, 100, 5))  # 32 batches, 100 samples, 5 categorical features

        # Apply multi-resolution attention
        attention = MultiResolutionTabularAttention(num_heads=4, d_model=32, dropout_rate=0.1)
        num_out, cat_out = attention([numerical, categorical])
        print("Numerical output shape:", num_out.shape)  # (32, 100, 32)
        print("Categorical output shape:", cat_out.shape)  # (32, 100, 32)
        ```
    """

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        dropout_rate: float = 0.1,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Set private attributes first
        self._num_heads = num_heads
        self._d_model = d_model
        self._dropout_rate = dropout_rate

        # Validate parameters
        self._validate_params()

        # Set public attributes BEFORE calling parent's __init__
        self.num_heads = self._num_heads
        self.d_model = self._d_model
        self.dropout_rate = self._dropout_rate

        # Initialize layers
        # Numerical features
        self.num_projection = None
        self.num_attention = None
        self.num_layernorm1 = None
        self.num_dropout1 = None
        self.num_layernorm2 = None
        self.num_dropout2 = None

        # Categorical features
        self.cat_projection = None
        self.cat_attention = None
        self.cat_layernorm1 = None
        self.cat_dropout1 = None
        self.cat_layernorm2 = None
        self.cat_dropout2 = None

        # Cross-attention
        self.num_cat_attention = None
        self.cat_num_attention = None
        self.cross_num_layernorm = None
        self.cross_num_dropout = None
        self.cross_cat_layernorm = None
        self.cross_cat_dropout = None

        # Feed-forward networks
        self.ffn_dense1 = None
        self.ffn_dense2 = None

        # Call parent's __init__ after setting public attributes
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if not isinstance(self._num_heads, int) or self._num_heads <= 0:
            raise ValueError(
                f"num_heads must be a positive integer, got {self._num_heads}",
            )

        if not isinstance(self._d_model, int) or self._d_model <= 0:
            raise ValueError(f"d_model must be a positive integer, got {self._d_model}")

        if not isinstance(self._dropout_rate, float) or not 0 <= self._dropout_rate < 1:
            raise ValueError(
                f"dropout_rate must be a float between 0 and 1, got {self._dropout_rate}",
            )

        # Check that d_model is divisible by num_heads
        if self._d_model % self._num_heads != 0:
            raise ValueError(
                f"d_model ({self._d_model}) must be divisible by num_heads ({self._num_heads})",
            )

    def build(self, input_shape: list[tuple[int, ...]]) -> None:
        """Builds the layer with the given input shape.

        Args:
            input_shape: List of two tuples defining the input shapes for numerical and categorical features.
        """
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError(
                f"Input must be a list of two tensors (numerical and categorical features), got {input_shape}",
            )

        num_shape, cat_shape = input_shape

        if len(num_shape) != 3 or len(cat_shape) != 3:
            raise ValueError(
                f"Both numerical and categorical inputs must be 3-dimensional "
                f"(batch_size, num_samples, num_features), got {num_shape} and {cat_shape}",
            )

        if num_shape[0] != cat_shape[0] or num_shape[1] != cat_shape[1]:
            raise ValueError(
                f"Batch size and number of samples must match for numerical and categorical inputs, "
                f"got shapes {num_shape} and {cat_shape}",
            )

        # Store input dimensions
        self.num_features = num_shape[-1]
        self.cat_features = cat_shape[-1]

        # Projection layers
        self.num_projection = layers.Dense(self.d_model)
        self.cat_projection = layers.Dense(self.d_model)

        # Self-attention layers
        self.num_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.dropout_rate,
        )
        self.cat_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.dropout_rate,
        )

        # Cross-attention layers
        self.num_cat_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.dropout_rate,
        )
        self.cat_num_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.dropout_rate,
        )

        # Feed-forward network
        self.ffn_dense1 = layers.Dense(self.d_model, activation="relu")
        self.ffn_dense2 = layers.Dense(self.d_model)

        # Normalization and dropout layers
        # For numerical features
        self.num_layernorm1 = layers.LayerNormalization()
        self.num_dropout1 = layers.Dropout(self.dropout_rate)
        self.num_layernorm2 = layers.LayerNormalization()
        self.num_dropout2 = layers.Dropout(self.dropout_rate)
        self.cross_num_layernorm = layers.LayerNormalization()
        self.cross_num_dropout = layers.Dropout(self.dropout_rate)

        # For categorical features
        self.cat_layernorm1 = layers.LayerNormalization()
        self.cat_dropout1 = layers.Dropout(self.dropout_rate)
        self.cat_layernorm2 = layers.LayerNormalization()
        self.cat_dropout2 = layers.Dropout(self.dropout_rate)
        self.cross_cat_layernorm = layers.LayerNormalization()
        self.cross_cat_dropout = layers.Dropout(self.dropout_rate)

        logger.debug(
            f"MultiResolutionTabularAttention built with d_model={self.d_model}, "
            f"num_heads={self.num_heads}, num_features={self.num_features}, "
            f"cat_features={self.cat_features}",
        )
        super().build(input_shape)

    def call(
        self,
        inputs: list[KerasTensor],
        training: bool = False,
    ) -> tuple[KerasTensor, KerasTensor]:
        """Forward pass of the layer.

        Args:
            inputs: List of two tensors [numerical_features, categorical_features]
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Tuple of two tensors (numerical_output, categorical_output)
        """
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError(
                "Input must be a list of two tensors (numerical and categorical features)",
            )

        numerical, categorical = inputs

        # Project inputs to d_model dimension
        num_projected = self.num_projection(numerical)
        cat_projected = self.cat_projection(categorical)

        # Self-attention for numerical features
        num_attention = self.num_attention(
            num_projected,
            num_projected,
            num_projected,
            training=training,
        )
        num_attention = self.num_layernorm1(
            num_projected + self.num_dropout1(num_attention, training=training),
        )

        # Self-attention for categorical features
        cat_attention = self.cat_attention(
            cat_projected,
            cat_projected,
            cat_projected,
            training=training,
        )
        cat_attention = self.cat_layernorm1(
            cat_projected + self.cat_dropout1(cat_attention, training=training),
        )

        # Cross-attention: numerical to categorical
        num_to_cat = self.num_cat_attention(
            query=cat_attention,
            key=num_attention,
            value=num_attention,
            training=training,
        )
        cat_cross = self.cross_cat_layernorm(
            cat_attention + self.cross_cat_dropout(num_to_cat, training=training),
        )

        # Cross-attention: categorical to numerical
        cat_to_num = self.cat_num_attention(
            query=num_attention,
            key=cat_attention,
            value=cat_attention,
            training=training,
        )
        num_cross = self.cross_num_layernorm(
            num_attention + self.cross_num_dropout(cat_to_num, training=training),
        )

        # Feed-forward network for numerical features
        num_ffn = self.ffn_dense2(self.ffn_dense1(num_cross))
        num_output = self.num_layernorm2(
            num_cross + self.num_dropout2(num_ffn, training=training),
        )

        # Feed-forward network for categorical features
        cat_ffn = self.ffn_dense2(self.ffn_dense1(cat_cross))
        cat_output = self.cat_layernorm2(
            cat_cross + self.cat_dropout2(cat_ffn, training=training),
        )

        return num_output, cat_output

    def compute_output_shape(
        self,
        input_shape: list[tuple[int, ...]],
    ) -> list[tuple[int, ...]]:
        """Compute the output shape of the layer.

        Args:
            input_shape: List of shapes of the input tensors.

        Returns:
            List of shapes of the output tensors.
        """
        num_shape, cat_shape = input_shape
        return [
            (num_shape[0], num_shape[1], self.d_model),
            (cat_shape[0], cat_shape[1], self.d_model),
        ]

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "d_model": self.d_model,
                "dropout_rate": self.dropout_rate,
            },
        )
        return config
