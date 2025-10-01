"""This module implements a GraphFeatureAggregation layer that treats features as nodes in a graph
and uses attention mechanisms to learn relationships between features. This approach is useful
for tabular data where features have inherent relationships.
"""

from typing import Any
from loguru import logger
from keras import layers, ops
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class GraphFeatureAggregation(BaseLayer):
    """Graph-based feature aggregation layer with self-attention for tabular data.

    This layer treats each input feature as a node and projects it into an embedding space.
    It then computes pairwise attention scores between features and aggregates feature
    information based on these scores. Finally, it projects the aggregated features back
    to the original feature space and adds a residual connection.

    The process involves:
      1. Projecting each scalar feature to an embedding (shape: [batch, num_features, embed_dim]).
      2. Computing pairwise concatenated embeddings and scoring them via a learnable attention vector.
      3. Normalizing the scores with softmax to yield a dynamic adjacency (attention) matrix.
      4. Aggregating neighboring features via weighted sum.
      5. Projecting back to a vector of original dimension, then adding a residual connection.

    Args:
        embed_dim: Dimensionality of the projected feature embeddings. Default is 8.
        dropout_rate: Dropout rate to apply on attention weights. Default is 0.0.
        leaky_relu_alpha: Alpha parameter for the LeakyReLU activation. Default is 0.2.
        name: Optional name for the layer.

    Input shape:
        2D tensor with shape: `(batch_size, num_features)`

    Output shape:
        2D tensor with shape: `(batch_size, num_features)` (same as input)

    Example:
        ```python
        import keras
        from kmr.layers import GraphFeatureAggregation

        # Tabular data with 10 features
        x = keras.random.normal((32, 10))

        # Create the layer with an embedding dimension of 8 and dropout rate of 0.1
        graph_layer = GraphFeatureAggregation(embed_dim=8, dropout_rate=0.1)
        y = graph_layer(x, training=True)
        print("Output shape:", y.shape)  # Expected: (32, 10)
        ```
    """

    def __init__(
        self,
        embed_dim: int = 8,
        dropout_rate: float = 0.0,
        leaky_relu_alpha: float = 0.2,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Set public attributes
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.leaky_relu_alpha = leaky_relu_alpha

        # Initialize instance variables
        self.num_features = None
        self.projection = None
        self.attention_a = None
        self.attention_bias = None
        self.leaky_relu = None
        self.dropout_layer = None
        self.out_proj = None

        # Validate parameters during initialization
        self._validate_params()
        # Call parent's __init__
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {self.embed_dim}")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError(
                f"dropout_rate must be between 0 and 1, got {self.dropout_rate}",
            )
        if self.leaky_relu_alpha <= 0:
            raise ValueError(
                f"leaky_relu_alpha must be positive, got {self.leaky_relu_alpha}",
            )

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Builds the layer with the given input shape.

        Args:
            input_shape: Tuple of integers defining the input shape.
        """
        # Validate parameters again during build
        self._validate_params()

        # Number of features is inferred from the last dimension of input
        self.num_features = input_shape[-1]

        # Project each scalar feature to an embedding vector
        self.projection = layers.Dense(
            self.embed_dim,
            activation=None,
            use_bias=True,  # Use bias to ensure non-zero outputs with zero inputs
            name="projection",
        )

        # Learnable attention vector (for concatenated embeddings of two nodes)
        # Shape: (2 * embed_dim, 1)
        self.attention_a = self.add_weight(
            name="attention_a",
            shape=(2 * self.embed_dim, 1),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Add a bias term to the attention mechanism
        self.attention_bias = self.add_weight(
            name="attention_bias",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )

        # Activation and regularization layers
        self.leaky_relu = layers.LeakyReLU(alpha=self.leaky_relu_alpha)
        self.dropout_layer = (
            layers.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None
        )

        # Final projection to compress aggregated embeddings back to the original number of features
        self.out_proj = layers.Dense(
            1,  # Project each embedding to a scalar
            activation=None,
            use_bias=True,  # Use bias to ensure non-zero outputs with zero inputs
            name="out_proj",
        )

        logger.debug(
            f"GraphFeatureAggregation built with num_features={self.num_features}, "
            f"embed_dim={self.embed_dim}, dropout_rate={self.dropout_rate}",
        )
        super().build(input_shape)

    def call(self, inputs: KerasTensor, training: bool | None = None) -> KerasTensor:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor with shape (batch_size, num_features).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Only relevant when using dropout.

        Returns:
            Output tensor with the same shape as input.
        """
        # Get batch size
        batch_size = ops.shape(inputs)[0]

        # Project each feature scalar to an embedding vector
        # First expand dims to create a "channel" dimension for the Dense layer
        inputs_expanded = ops.expand_dims(inputs, axis=-1)  # (batch, num_features, 1)
        h = self.projection(inputs_expanded)  # (batch, num_features, embed_dim)

        # Compute attention scores for all pairs of features
        # We'll use a different approach to avoid shape incompatibility

        # Create all possible pairs of feature embeddings
        # First, prepare the embeddings for broadcasting
        h_i = ops.reshape(h, (batch_size, self.num_features, 1, self.embed_dim))
        h_j = ops.reshape(h, (batch_size, 1, self.num_features, self.embed_dim))

        # Create feature pairs by broadcasting and concatenating
        # This creates a tensor of shape (batch, num_features, num_features, 2*embed_dim)
        # where each position (i,j) contains the concatenation of embeddings for features i and j
        h_i_tiled = ops.repeat(
            h_i,
            self.num_features,
            axis=2,
        )  # (batch, num_features, num_features, embed_dim)
        h_j_tiled = ops.repeat(
            h_j,
            self.num_features,
            axis=1,
        )  # (batch, num_features, num_features, embed_dim)

        # Now we can safely concatenate along the last dimension
        h_pairs = ops.concatenate(
            [h_i_tiled, h_j_tiled],
            axis=-1,
        )  # (batch, num_features, num_features, 2*embed_dim)

        # Reshape for matrix multiplication with attention vector
        h_pairs_flat = ops.reshape(
            h_pairs,
            (-1, 2 * self.embed_dim),
        )  # (batch*num_features*num_features, 2*embed_dim)

        # Apply attention vector to get raw scores
        raw_scores_flat = ops.matmul(
            h_pairs_flat,
            self.attention_a,
        )  # (batch*num_features*num_features, 1)

        # Add bias term to ensure non-zero attention scores even with zero inputs
        raw_scores_flat = raw_scores_flat + self.attention_bias

        # Reshape back to (batch, num_features, num_features)
        raw_scores = ops.reshape(
            raw_scores_flat,
            (batch_size, self.num_features, self.num_features),
        )

        # Apply LeakyReLU activation
        scores = self.leaky_relu(raw_scores)

        # Normalize scores using softmax over the neighbor dimension
        attention = ops.softmax(scores, axis=-1)  # (batch, num_features, num_features)

        # Apply dropout if needed
        if self.dropout_layer is not None and training:
            attention = self.dropout_layer(attention, training=training)

        # Aggregate embeddings: for each feature i, compute weighted sum over features j
        aggregated = ops.matmul(attention, h)  # (batch, num_features, embed_dim)

        # Project back to original feature space
        # Apply the output projection to each feature embedding separately
        projected = self.out_proj(aggregated)  # (batch, num_features, 1)

        # Remove the last dimension to get shape (batch, num_features)
        projected = ops.squeeze(projected, axis=-1)  # (batch, num_features)

        # Residual connection: add original inputs
        output = projected + inputs

        return output

    def get_config(self) -> dict[str, Any]:
        """Returns the config of the layer.

        Returns:
            Python dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "dropout_rate": self.dropout_rate,
                "leaky_relu_alpha": self.leaky_relu_alpha,
            },
        )
        return config
