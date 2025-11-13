"""This module implements a DistributionAwareEncoder layer that automatically detects
the distribution type of input data and applies appropriate transformations and encodings.
It builds upon the DistributionTransformLayer but adds more sophisticated distribution
detection and specialized encoding for different distribution types.
"""

from typing import Any
from loguru import logger
from keras import layers, ops
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kerasfactory.layers._base_layer import BaseLayer
from kerasfactory.layers.DistributionTransformLayer import DistributionTransformLayer


@register_keras_serializable(package="kerasfactory.layers")
class DistributionAwareEncoder(BaseLayer):
    """Layer that automatically detects and encodes data based on its distribution.

    This layer first detects the distribution type of the input data and then applies
    appropriate transformations and encodings. It builds upon the DistributionTransformLayer
    but adds more sophisticated distribution detection and specialized encoding for
    different distribution types.

    Args:
        embedding_dim: Dimension of the output embedding. If None, the output will have
            the same dimension as the input. Default is None.
        auto_detect: Whether to automatically detect the distribution type. If False,
            the layer will use the specified distribution_type. Default is True.
        distribution_type: The distribution type to use if auto_detect is False.
            Options are "normal", "exponential", "lognormal", "uniform", "beta",
            "bimodal", "heavy_tailed", "mixed", "bounded", "unknown". Default is "unknown".
        transform_type: The transformation type to use. If "auto", the layer will
            automatically select the best transformation based on the detected distribution.
            See DistributionTransformLayer for available options. Default is "auto".
        add_distribution_embedding: Whether to add a learned embedding of the distribution
            type to the output. Default is False.
        name: Optional name for the layer.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., features)`.

    Output shape:
        If embedding_dim is None, same shape as input: `(batch_size, ..., features)`.
        If embedding_dim is specified: `(batch_size, ..., embedding_dim)`.
        If add_distribution_embedding is True, the output will have an additional
        dimension for the distribution embedding.

    Example:
        ```python
        import keras
        import numpy as np
        from kerasfactory.layers import DistributionAwareEncoder

        # Create sample input data with different distributions
        # Normal distribution
        normal_data = keras.ops.convert_to_tensor(
            np.random.normal(0, 1, (100, 10)), dtype="float32"
        )

        # Exponential distribution
        exp_data = keras.ops.convert_to_tensor(
            np.random.exponential(1, (100, 10)), dtype="float32"
        )

        # Create the encoder
        encoder = DistributionAwareEncoder(embedding_dim=16, add_distribution_embedding=True)

        # Apply to normal data
        normal_encoded = encoder(normal_data)
        print("Normal encoded shape:", normal_encoded.shape)  # (100, 16)

        # Apply to exponential data
        exp_encoded = encoder(exp_data)
        print("Exponential encoded shape:", exp_encoded.shape)  # (100, 16)
        ```
    """

    def __init__(
        self,
        embedding_dim: int | None = None,
        auto_detect: bool = True,
        distribution_type: str = "unknown",
        transform_type: str = "auto",
        add_distribution_embedding: bool = False,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the DistributionAwareEncoder.

        Args:
            embedding_dim: Embedding dimension.
            auto_detect: Whether to auto-detect distribution type.
            distribution_type: Type of distribution.
            transform_type: Type of transformation to apply.
            add_distribution_embedding: Whether to add distribution embedding.
            name: Name of the layer.
            **kwargs: Additional keyword arguments.
        """
        # Set private attributes first
        self._embedding_dim = embedding_dim
        self._auto_detect = auto_detect
        self._distribution_type = distribution_type
        self._transform_type = transform_type
        self._add_distribution_embedding = add_distribution_embedding

        # Define valid distribution types
        self._valid_distributions = [
            "normal",
            "exponential",
            "lognormal",
            "uniform",
            "beta",
            "bimodal",
            "heavy_tailed",
            "mixed",
            "bounded",
            "unknown",
        ]

        # Validate parameters
        self._validate_params()

        # Set public attributes BEFORE calling parent's __init__
        self.embedding_dim = self._embedding_dim
        self.auto_detect = self._auto_detect
        self.distribution_type = self._distribution_type
        self.transform_type = self._transform_type
        self.add_distribution_embedding = self._add_distribution_embedding

        # Initialize instance variables
        self.distribution_transform: DistributionTransformLayer | None = None
        self.distribution_embedding: layers.Embedding | None = None
        self.projection: layers.Dense | None = None
        self.detected_distribution: layers.Variable | None = None
        self._is_initialized: bool = False

        # Call parent's __init__ after setting public attributes
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if self._embedding_dim is not None and self._embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be positive, got {self._embedding_dim}",
            )

        if not isinstance(self._auto_detect, bool):
            raise ValueError(f"auto_detect must be a boolean, got {self._auto_detect}")

        if self._distribution_type not in self._valid_distributions:
            raise ValueError(
                f"distribution_type must be one of {self._valid_distributions}, got {self._distribution_type}",
            )

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Builds the layer with the given input shape.

        Args:
            input_shape: Tuple of integers defining the input shape.
        """
        # Get the feature dimension
        input_shape[-1]

        # Create the distribution transform layer
        self.distribution_transform = DistributionTransformLayer(
            transform_type=self.transform_type,
            name="distribution_transform",
        )

        # Create the distribution embedding if needed
        if self.add_distribution_embedding:
            # Create a learned embedding for each distribution type
            num_distributions = len(self._valid_distributions)
            self.distribution_embedding = self.add_weight(
                name="distribution_embedding",
                shape=(
                    num_distributions,
                    8,
                ),  # 8-dimensional embedding for each distribution
                initializer="glorot_uniform",
                trainable=True,
            )

        # Create the projection layer if needed
        if self.embedding_dim is not None:
            self.projection = layers.Dense(
                self.embedding_dim,
                activation="relu",
                name="projection",
            )

        # Create a variable to store the detected distribution
        if self.auto_detect:
            self.detected_distribution = self.add_weight(
                name="detected_distribution",
                shape=(1,),
                dtype="int32",
                trainable=False,
                initializer="zeros",
            )

        logger.debug(
            f"DistributionAwareEncoder built with embedding_dim={self.embedding_dim}, "
            f"auto_detect={self.auto_detect}, distribution_type={self.distribution_type}, "
            f"transform_type={self.transform_type}, "
            f"add_distribution_embedding={self.add_distribution_embedding}",
        )
        super().build(input_shape)

    def _detect_distribution(self, x: KerasTensor) -> int:
        """Detect the distribution type of the input data.

        Args:
            x: Input tensor

        Returns:
            Index of the detected distribution type in self._valid_distributions
        """
        # Calculate statistics (once)
        mean = ops.mean(x, axis=0)
        std = ops.std(x, axis=0)
        avg_std = ops.mean(std)  # Average std across features

        # Calculate higher moments using the pre-calculated mean and std
        skewness = self._calculate_skewness(x, mean, std)
        kurtosis = self._calculate_kurtosis(x, mean, std)

        # Calculate average statistics across features
        avg_skewness = ops.mean(ops.abs(skewness))
        avg_kurtosis = ops.mean(kurtosis)

        # Check if data is bounded between 0 and 1
        min_val = ops.min(x)
        max_val = ops.max(x)
        is_bounded_01 = ops.logical_and(min_val >= 0, max_val <= 1)

        # Check if data has negative values
        has_negative = ops.any(x < 0)

        # Check if data is heavy-tailed
        is_heavy_tailed = avg_kurtosis > 3.0

        # Detect distribution type based on statistics
        distribution_idx = 0  # Default to unknown

        # Normal distribution: low skewness, kurtosis around 3
        if avg_skewness < 0.2 and ops.abs(avg_kurtosis - 3.0) < 0.5:
            distribution_idx = self._valid_distributions.index("normal")
        # Exponential distribution: high skewness, no negative values
        elif (
            avg_skewness > 1.5
            and ops.logical_not(has_negative)
            and ops.logical_not(is_bounded_01)
        ):
            distribution_idx = self._valid_distributions.index("exponential")
        # Lognormal: very high skewness, no negative values, some very large values
        elif (
            avg_skewness > 2.0
            and ops.logical_not(has_negative)
            and ops.any(x > 20 * ops.mean(mean))
        ):
            distribution_idx = self._valid_distributions.index("lognormal")
        # Beta: bounded in [0,1] with skewness
        elif is_bounded_01 and avg_skewness > 0.5:
            distribution_idx = self._valid_distributions.index("beta")
        # Uniform: bounded with low skewness
        elif is_bounded_01:
            distribution_idx = self._valid_distributions.index("uniform")
        # Heavy-tailed distribution
        elif is_heavy_tailed:
            distribution_idx = self._valid_distributions.index("heavy_tailed")
        # Bimodal distribution - check this after other distributions
        elif self._check_bimodality(x, mean, std, kurtosis):
            distribution_idx = self._valid_distributions.index("bimodal")
        # Mixed distribution: has negative values and moderate variance
        # Using pre-calculated avg_std instead of recalculating
        elif has_negative and avg_std > 1.5:
            distribution_idx = self._valid_distributions.index("mixed")
        # Unknown distribution - fallback
        else:
            distribution_idx = self._valid_distributions.index("unknown")

        logger.debug(
            f"Detected distribution: {self._valid_distributions[distribution_idx]}, "
            f"skewness={avg_skewness}, kurtosis={avg_kurtosis}, "
            f"is_bounded_01={is_bounded_01}, has_negative={has_negative}, "
            f"is_heavy_tailed={is_heavy_tailed}, avg_std={avg_std}",
        )

        return distribution_idx

    def _calculate_skewness(
        self,
        x: KerasTensor,
        mean: KerasTensor = None,
        std: KerasTensor = None,
    ) -> KerasTensor:
        """Calculate the skewness of the input tensor.

        Args:
            x: Input tensor
            mean: Pre-calculated mean (optional)
            std: Pre-calculated standard deviation (optional)

        Returns:
            Skewness value
        """
        # Calculate mean and standard deviation if not provided
        mean = (
            ops.mean(x, axis=0, keepdims=True)
            if mean is None
            else ops.expand_dims(mean, axis=0)
        )

        std = (
            ops.std(x, axis=0, keepdims=True)
            if std is None
            else ops.expand_dims(std, axis=0)
        )

        # Add small epsilon to std to avoid division by zero
        std = ops.maximum(std, 1e-10)

        # Calculate skewness
        skewness = ops.mean(ops.power((x - mean) / std, 3.0), axis=0)

        return skewness

    def _calculate_kurtosis(
        self,
        x: KerasTensor,
        mean: KerasTensor = None,
        std: KerasTensor = None,
    ) -> KerasTensor:
        """Calculate the kurtosis of the input tensor.

        Args:
            x: Input tensor
            mean: Pre-calculated mean (optional)
            std: Pre-calculated standard deviation (optional)

        Returns:
            Kurtosis value
        """
        # Calculate mean and standard deviation if not provided
        mean = (
            ops.mean(x, axis=0, keepdims=True)
            if mean is None
            else ops.expand_dims(mean, axis=0)
        )

        std = (
            ops.std(x, axis=0, keepdims=True)
            if std is None
            else ops.expand_dims(std, axis=0)
        )

        # Add small epsilon to std to avoid division by zero
        std = ops.maximum(std, 1e-10)

        # Calculate kurtosis
        kurtosis = ops.mean(ops.power((x - mean) / std, 4.0), axis=0)

        return kurtosis

    def _check_bimodality(
        self,
        x: KerasTensor,
        mean: KerasTensor = None,
        std: KerasTensor = None,
        precalculated_kurtosis: KerasTensor = None,
    ) -> KerasTensor:
        """Check if the distribution is bimodal.

        This is a simplified check that looks for a dip in the histogram.

        Args:
            x: Input tensor
            mean: Pre-calculated mean (optional)
            std: Pre-calculated standard deviation (optional)
            precalculated_kurtosis: Pre-calculated kurtosis (optional)

        Returns:
            Boolean tensor indicating if the distribution is likely bimodal
        """
        # For graph mode compatibility, we need a simpler approach
        # Use pre-calculated kurtosis if provided, otherwise calculate it
        if precalculated_kurtosis is not None:
            kurtosis = ops.mean(precalculated_kurtosis)
        else:
            kurtosis = ops.mean(self._calculate_kurtosis(x, mean, std))

        # Bimodal distributions often have lower kurtosis than normal distributions
        # This is a simplified heuristic that should work in graph mode
        is_bimodal = ops.logical_and(
            kurtosis < 2.5,  # Lower than normal distribution
            ops.logical_not(ops.abs(kurtosis - 3.0) < 0.5),  # Not too close to normal
        )

        return is_bimodal

    def call(self, inputs: KerasTensor, training: bool | None = None) -> KerasTensor:
        """Apply distribution-aware encoding to the inputs.

        Args:
            inputs: Input tensor
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode

        Returns:
            Encoded tensor
        """
        # Ensure layer is built (Keras will auto-build on first call)
        if not self.built:
            self.build(inputs.shape)

        # Ensure inputs are cast to float32
        x = ops.cast(inputs, dtype="float32")

        # Detect distribution type if auto_detect is True
        if self.auto_detect:
            if training or not self._is_initialized:
                # During training or first call, detect the distribution
                distribution_idx = self._detect_distribution(x)

                # Store the detected distribution
                if self.detected_distribution is not None:
                    self.detected_distribution.assign(ops.array([distribution_idx]))

                # Set the distribution type for this forward pass
                self.distribution_type = self._valid_distributions[distribution_idx]

                # Mark as initialized
                self._is_initialized = True
            else:
                # During inference, use the stored distribution
                if self.detected_distribution is not None:
                    distribution_idx = int(
                        ops.convert_to_numpy(self.detected_distribution)[0],
                    )
                    self.distribution_type = self._valid_distributions[distribution_idx]

        # Apply distribution transform
        transformed = self.distribution_transform(x, training=training)

        # Apply projection if needed
        encoded = (
            self.projection(transformed) if self.projection is not None else transformed
        )

        # Add distribution embedding if needed
        if self.add_distribution_embedding and hasattr(self, "distribution_embedding"):
            # Get the distribution index
            if self.auto_detect:
                distribution_idx = int(
                    ops.convert_to_numpy(self.detected_distribution)[0],
                )
            else:
                distribution_idx = self._valid_distributions.index(
                    self.distribution_type,
                )

            # Get the embedding for this distribution
            dist_embedding = self.distribution_embedding[distribution_idx]

            # Reshape for broadcasting
            dist_embedding = ops.reshape(
                dist_embedding,
                (1,) * (len(encoded.shape) - 1) + (8,),
            )

            # Repeat the embedding for each item in the batch
            batch_size = ops.shape(encoded)[0]
            dist_embedding = ops.repeat(dist_embedding, batch_size, axis=0)

            # Concatenate with the encoded tensor
            encoded = ops.concatenate([encoded, dist_embedding], axis=-1)

        return encoded

    def get_config(self) -> dict[str, Any]:
        """Returns the config of the layer.

        Returns:
            Python dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "auto_detect": self.auto_detect,
                "distribution_type": self.distribution_type,
                "transform_type": self.transform_type,
                "add_distribution_embedding": self.add_distribution_embedding,
            },
        )
        return config
