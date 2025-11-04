"""This module implements a VariableSelection layer that applies a gated residual network to each
feature independently and learns feature weights through a softmax layer. It's particularly
useful for dynamic feature selection in time series and tabular models.
"""

from typing import Any
from loguru import logger
from keras import layers, ops
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kmr.layers._base_layer import BaseLayer
from kmr.layers.GatedResidualNetwork import GatedResidualNetwork


@register_keras_serializable(package="kmr.layers")
class VariableSelection(BaseLayer):
    """Layer for dynamic feature selection using gated residual networks.

    This layer applies a gated residual network to each feature independently and learns
    feature weights through a softmax layer. It can optionally use a context vector to
    condition the feature selection.

    Args:
        nr_features (int): Number of input features
        units (int): Number of hidden units in the gated residual network
        dropout_rate (float): Dropout rate for regularization
        use_context (bool): Whether to use a context vector for conditioning
        name (str, optional): Name for the layer

    Input shape:
        If use_context is False:
            - Single tensor with shape: `(batch_size, nr_features, feature_dim)`
        If use_context is True:
            - List of two tensors:
                - Features tensor with shape: `(batch_size, nr_features, feature_dim)`
                - Context tensor with shape: `(batch_size, context_dim)`

    Output shape:
        Tuple of two tensors:
        - Selected features: `(batch_size, feature_dim)`
        - Feature weights: `(batch_size, nr_features)`

    Example:
        ```python
        import keras
        from kmr.layers import VariableSelection

        # Create sample input data
        x = keras.random.normal((32, 10, 16))  # 32 batches, 10 features, 16 dims per feature

        # Without context
        vs = VariableSelection(nr_features=10, units=32, dropout_rate=0.1)
        selected, weights = vs(x)
        print("Selected features shape:", selected.shape)  # (32, 16)
        print("Feature weights shape:", weights.shape)  # (32, 10)

        # With context
        context = keras.random.normal((32, 64))  # 32 batches, 64-dim context
        vs_context = VariableSelection(nr_features=10, units=32, dropout_rate=0.1, use_context=True)
        selected, weights = vs_context([x, context])
        ```
    """

    def __init__(
        self,
        nr_features: int,
        units: int,
        dropout_rate: float = 0.1,
        use_context: bool = False,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the VariableSelection layer.

        Args:
            nr_features: Number of input features.
            units: Number of units in the selection network.
            dropout_rate: Dropout rate.
            use_context: Whether to use context for selection.
            name: Name of the layer.
            **kwargs: Additional keyword arguments.
        """
        # Set private attributes first
        self._nr_features = nr_features
        self._units = units
        self._dropout_rate = dropout_rate
        self._use_context = use_context

        # Validate parameters
        self._validate_params()

        # Set public attributes BEFORE calling parent's __init__
        self.nr_features = self._nr_features
        self.units = self._units
        self.dropout_rate = self._dropout_rate
        self.use_context = self._use_context

        # Initialize layers
        self.feature_grns: list[GatedResidualNetwork] | None = None
        self.grn_var: GatedResidualNetwork | None = None
        self.softmax: layers.Dense | None = None

        # Call parent's __init__ after setting public attributes
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if not isinstance(self._nr_features, int) or self._nr_features <= 0:
            raise ValueError(
                f"nr_features must be a positive integer, got {self._nr_features}",
            )

        if not isinstance(self._units, int) or self._units <= 0:
            raise ValueError(f"units must be a positive integer, got {self._units}")

        if not isinstance(self._dropout_rate, float) or not 0 <= self._dropout_rate < 1:
            raise ValueError(
                f"dropout_rate must be a float between 0 and 1, got {self._dropout_rate}",
            )

        if not isinstance(self._use_context, bool):
            raise ValueError(f"use_context must be a boolean, got {self._use_context}")

    def build(self, input_shape: tuple[int, ...] | list[tuple[int, ...]]) -> None:
        """Builds the layer with the given input shape.

        Args:
            input_shape: Shape of the input tensor or list of shapes if using context.
        """
        if self.use_context:
            if not isinstance(input_shape, list) or len(input_shape) != 2:
                raise ValueError(
                    f"When use_context is True, input_shape must be a list of two shapes, got {input_shape}",
                )
            features_shape, context_shape = input_shape

            # Handle 2D feature input for backward compatibility
            if len(features_shape) == 2:
                if features_shape[1] != self.nr_features:
                    raise ValueError(
                        f"Number of features in input ({features_shape[1]}) does not match "
                        f"nr_features ({self.nr_features})",
                    )
                # Set a default feature dimension of 1 for 2D inputs
                self.feature_dim = 1
            elif len(features_shape) == 3:
                if features_shape[1] != self.nr_features:
                    raise ValueError(
                        f"Number of features in input ({features_shape[1]}) does not match "
                        f"nr_features ({self.nr_features})",
                    )
                self.feature_dim = features_shape[2]
            else:
                raise ValueError(
                    f"Features tensor must be 2-dimensional (batch_size, nr_features) or "
                    f"3-dimensional (batch_size, nr_features, feature_dim), "
                    f"got shape {features_shape}",
                )

            if len(context_shape) != 2:
                raise ValueError(
                    f"Context tensor must be 2-dimensional (batch_size, context_dim), got shape {context_shape}",
                )

            context_shape[-1]
        else:
            # Handle 2D input for backward compatibility
            if isinstance(input_shape, tuple) and len(input_shape) == 2:
                if input_shape[1] != self.nr_features:
                    raise ValueError(
                        f"Number of features in input ({input_shape[1]}) does not match "
                        f"nr_features ({self.nr_features})",
                    )
                # Set a default feature dimension of 1 for 2D inputs
                self.feature_dim = 1
            elif isinstance(input_shape, tuple) and len(input_shape) == 3:
                if input_shape[1] != self.nr_features:
                    raise ValueError(
                        f"Number of features in input ({input_shape[1]}) does not match "
                        f"nr_features ({self.nr_features})",
                    )
                self.feature_dim = input_shape[2]
            else:
                raise ValueError(
                    f"When use_context is False, input must be 2-dimensional "
                    f"(batch_size, nr_features) or 3-dimensional "
                    f"(batch_size, nr_features, feature_dim), got shape {input_shape}",
                )

        # Create GRN layers for each feature
        self.feature_grns = [
            GatedResidualNetwork(
                units=self.units,
                dropout_rate=self.dropout_rate,
                name=f"grn_{i}",
            )
            for i in range(self.nr_features)
        ]

        # Create variable selection network
        self.grn_var = GatedResidualNetwork(
            units=self.nr_features,
            dropout_rate=self.dropout_rate,
            name="weight_network",
        )

        # Create softmax layer
        self.softmax = layers.Activation("softmax")

        logger.debug(
            f"VariableSelection built with nr_features={self.nr_features}, "
            f"units={self.units}, use_context={self.use_context}",
        )
        super().build(input_shape)

    def call(
        self,
        inputs: KerasTensor | list[KerasTensor],
        training: bool = False,
    ) -> tuple:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor or list of tensors if using context
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Tuple of (selected_features, feature_weights)
        """
        # Ensure layer is built (Keras will auto-build on first call)
        if not self.built:
            # Determine input shape for building
            input_shape = inputs[0].shape if isinstance(inputs, list) else inputs.shape
            self.build(input_shape)

        if self.use_context:
            if not isinstance(inputs, list) or len(inputs) != 2:
                raise ValueError(
                    "When use_context is True, inputs must be a list of two tensors (features and context)",
                )
            features, context = inputs

            # Handle 2D features input
            if len(features.shape) == 2:
                features = ops.expand_dims(
                    features,
                    axis=-1,
                )  # [batch_size, nr_features, 1]
        else:
            # When use_context is False, we expect a single tensor, not a list
            if isinstance(inputs, list):
                raise ValueError(
                    "When use_context is False, inputs must be a single tensor, not a list of tensors",
                )

            features = inputs
            context = None

            # Handle 2D features input
            if len(features.shape) == 2:
                features = ops.expand_dims(
                    features,
                    axis=-1,
                )  # [batch_size, nr_features, 1]

        # Apply GRN to each feature
        transformed_features = []
        for i in range(self.nr_features):
            # Extract i-th feature
            feature = features[:, i, :]
            # Apply GRN
            transformed = self.feature_grns[i](feature, training=training)
            transformed_features.append(transformed)

        # Stack transformed features
        transformed_features = ops.stack(transformed_features, axis=1)

        # Compute feature weights
        if self.use_context:
            # Use context to compute weights
            weights = self.grn_var(context, training=training)
        else:
            # Compute weights from features directly
            weights = self.grn_var(ops.mean(features, axis=2))

        # Apply softmax to get normalized weights
        weights = self.softmax(weights)

        # Expand weights for broadcasting
        weights = ops.expand_dims(weights, axis=-1)

        # Compute weighted sum of features
        selected_features = ops.sum(transformed_features * weights, axis=1)

        # Remove the last dimension from weights for output
        weights = ops.squeeze(weights, axis=-1)

        return selected_features, weights

    def compute_output_shape(
        self,
        input_shape: tuple[int, ...] | list[tuple[int, ...]],
    ) -> list[tuple[int, ...]]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor or list of shapes if using context.

        Returns:
            List of shapes for the output tensors.
        """
        features_shape = input_shape[0] if self.use_context else input_shape

        # Handle different input shape types
        if isinstance(features_shape, list | tuple) and len(features_shape) > 0:
            batch_size = (
                int(features_shape[0])
                if isinstance(features_shape[0], int | float)
                else 1
            )
        else:
            batch_size = 1  # Default fallback

        return [
            (batch_size, self.units),  # Selected features
            (batch_size, self.nr_features),  # Feature weights
        ]

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "nr_features": self.nr_features,
                "units": self.units,
                "dropout_rate": self.dropout_rate,
                "use_context": self.use_context,
            },
        )
        return config
