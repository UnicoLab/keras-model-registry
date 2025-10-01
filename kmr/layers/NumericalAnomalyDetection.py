from keras import layers, ops
from keras import KerasTensor
from typing import Any
from loguru import logger
from keras.saving import register_keras_serializable
from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class NumericalAnomalyDetection(BaseLayer):
    """Numerical anomaly detection layer for identifying outliers in numerical features.

    This layer learns a distribution for each numerical feature and outputs an anomaly
    score for each feature based on how far it deviates from the learned distribution.
    The layer uses a combination of mean, variance, and autoencoder reconstruction error
    to detect anomalies.

    Example:
        ```python
        import tensorflow as tf
        from kmr.layers import NumericalAnomalyDetection

        # Suppose we have 5 numerical features
        x = tf.random.normal((32, 5))  # Batch of 32 samples
        # Create a NumericalAnomalyDetection layer
        anomaly_layer = NumericalAnomalyDetection(
            hidden_dims=[8, 4],
            reconstruction_weight=0.5,
            distribution_weight=0.5
        )
        anomaly_scores = anomaly_layer(x)
        print("Anomaly scores shape:", anomaly_scores.shape)  # Expected: (32, 5)
        ```
    """

    def __init__(
        self,
        hidden_dims: list[int],
        reconstruction_weight: float = 0.5,
        distribution_weight: float = 0.5,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the layer.

        Args:
            hidden_dims: List of hidden dimensions for the autoencoder.
            reconstruction_weight: Weight for reconstruction error in anomaly score.
            distribution_weight: Weight for distribution-based error in anomaly score.
            **kwargs: Additional keyword arguments.
        """
        self.hidden_dims = hidden_dims
        self.reconstruction_weight = reconstruction_weight
        self.distribution_weight = distribution_weight
        super().__init__(**kwargs)

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Build the layer.

        Args:
            input_shape: Input shape tuple.
        """
        self.num_features = input_shape[-1]

        # Encoder layers
        self.encoder_layers = []
        for dim in self.hidden_dims:
            self.encoder_layers.append(
                layers.Dense(dim, activation="relu", name=f"encoder_{dim}"),
            )

        # Decoder layers
        self.decoder_layers = []
        for dim in reversed(self.hidden_dims[:-1]):
            self.decoder_layers.append(
                layers.Dense(dim, activation="relu", name=f"decoder_{dim}"),
            )
        self.decoder_layers.append(
            layers.Dense(self.num_features, name="decoder_output"),
        )

        # Distribution parameters
        self.mean_layer = layers.Dense(self.num_features, name="mean")
        self.var_layer = layers.Dense(
            self.num_features,
            activation="softplus",
            name="variance",
        )

        logger.debug(
            "NumericalAnomalyDetection built with {} features and hidden dims {}.",
            self.num_features,
            self.hidden_dims,
        )
        super().build(input_shape)

    def call(self, inputs: KerasTensor) -> KerasTensor:
        """Forward pass.

        Args:
            inputs: Input tensor of shape (batch_size, num_features).

        Returns:
            Anomaly scores tensor of shape (batch_size, num_features).
        """
        # Autoencoder reconstruction
        x = inputs
        for layer in self.encoder_layers:
            x = layer(x)
        encoded = x

        x = encoded
        for layer in self.decoder_layers:
            x = layer(x)
        reconstructed = x

        # Distribution parameters
        mean = self.mean_layer(encoded)
        var = self.var_layer(encoded) + 1e-6  # Add epsilon for numerical stability

        # Compute reconstruction error (non-negative by design)
        reconstruction_error = ops.square(inputs - reconstructed)

        # Compute distribution-based error (ensure non-negative)
        distribution_error = ops.abs(ops.square(inputs - mean) / var + ops.log(var))

        # Combine errors (both components are non-negative)
        anomaly_scores = (
            self.reconstruction_weight * reconstruction_error
            + self.distribution_weight * distribution_error
        )

        return anomaly_scores

    def compute_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Compute output shape.

        Args:
            input_shape: Input shape tuple.

        Returns:
            Output shape tuple.
        """
        return input_shape

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Layer configuration dictionary.
        """
        config = super().get_config()
        config.update(
            {
                "hidden_dims": self.hidden_dims,
                "reconstruction_weight": self.reconstruction_weight,
                "distribution_weight": self.distribution_weight,
            },
        )
        return config
