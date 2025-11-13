from loguru import logger
from keras import layers, initializers, ops
from keras import KerasTensor
from kerasfactory.layers._base_layer import BaseLayer
from keras.saving import register_keras_serializable


@register_keras_serializable(package="kerasfactory.layers")
class AdvancedGraphFeatureLayer(BaseLayer):
    """Advanced graph-based feature layer for tabular data.

    This layer projects scalar features into an embedding space and then applies
    multi-head self-attention to compute data-dependent dynamic adjacencies between
    features. It learns edge attributes by considering both the raw embeddings and
    their differences. Optionally, a hierarchical aggregation is applied, where
    features are grouped via a learned soft-assignment and then re-expanded back to
    the original feature space. A residual connection and layer normalization are
    applied before the final projection back to the original feature space.

    The layer is highly configurable, allowing for control over the embedding dimension,
    number of attention heads, dropout rate, and hierarchical aggregation.

    Notes:
        **When to Use This Layer:**
        - When working with tabular data where feature interactions are important
        - For complex feature engineering tasks where manual feature crosses are insufficient
        - When dealing with heterogeneous features that require dynamic, learned relationships
        - In scenarios where feature importance varies across different samples
        - When hierarchical feature relationships exist in your data

        **Best Practices:**
        - Start with a small embed_dim (e.g., 16 or 32) and increase if needed
        - Use num_heads=4 or 8 for most applications
        - Enable hierarchical=True when you have many features (>20) or known grouping structure
        - Set dropout_rate=0.1 or 0.2 for regularization during training
        - Use layer normalization (enabled by default) to stabilize training

        **Performance Considerations:**
        - Memory usage scales quadratically with the number of features
        - Consider using hierarchical mode for large feature sets to reduce complexity
        - The layer works best with normalized input features
        - For very large feature sets (>100), consider feature pre-selection

    Args:
        embed_dim (int): Dimensionality of the projected feature embeddings. Determines the size
            of the learned feature representations.
        num_heads (int): Number of attention heads. Must divide embed_dim evenly. Each head
            learns different aspects of feature relationships.
        dropout_rate (float, optional): Dropout rate applied to attention weights during training.
            Helps prevent overfitting. Defaults to 0.0.
        hierarchical (bool, optional): Whether to apply hierarchical aggregation. If True, features
            are grouped into clusters, and aggregation is performed at the cluster level.
            Defaults to False.
        num_groups (int, optional): Number of groups to cluster features into when hierarchical is True.
            Must be provided if hierarchical is True. Controls the granularity of hierarchical
            aggregation.

    Raises:
        ValueError: If embed_dim is not divisible by num_heads. Ensures that the embedding dimension
            can be evenly split across attention heads.
        ValueError: If hierarchical is True but num_groups is not provided. The number of groups
            must be specified when hierarchical aggregation is enabled.

    Examples:
        **Basic Usage:**

        ```python
        import keras
        from kerasfactory.layers import AdvancedGraphFeatureLayer

        # Dummy tabular data with 10 features for 32 samples.
        x = keras.random.normal((32, 10))
        # Create the advanced graph layer with an embedding dimension of 16 and 4 heads.
        layer = AdvancedGraphFeatureLayer(embed_dim=16, num_heads=4)
        y = layer(x, training=True)
        print("Output shape:", y.shape)  # Expected: (32, 10)
        ```

        **With Hierarchical Aggregation:**

        ```python
        import keras
        from kerasfactory.layers import AdvancedGraphFeatureLayer

        # Dummy tabular data with 10 features for 32 samples.
        x = keras.random.normal((32, 10))
        # Create the advanced graph layer with hierarchical aggregation into 4 groups.
        layer = AdvancedGraphFeatureLayer(embed_dim=16, num_heads=4, hierarchical=True, num_groups=4)
        y = layer(x, training=True)
        print("Output shape:", y.shape)  # Expected: (32, 10)
        ```

        **Without Training:**

        ```python
        import keras
        from kerasfactory.layers import AdvancedGraphFeatureLayer

        # Dummy tabular data with 10 features for 32 samples.
        x = keras.random.normal((32, 10))
        # Create the advanced graph layer with an embedding dimension of 16 and 4 heads.
        layer = AdvancedGraphFeatureLayer(embed_dim=16, num_heads=4)
        y = layer(x, training=False)
        print("Output shape:", y.shape)  # Expected: (32, 10)
        ```
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        hierarchical: bool = False,
        num_groups: int | None = None,
        **kwargs,
    ) -> None:
        """Initialize the AdvancedGraphFeature layer.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            dropout_rate: Dropout rate.
            hierarchical: Whether to use hierarchical attention.
            num_groups: Number of groups for hierarchical attention.
            **kwargs: Additional keyword arguments.
        """
        # Validate parameters before setting attributes
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if hierarchical and num_groups is None:
            raise ValueError("num_groups must be specified when hierarchical is True")

        # Set attributes before calling super().__init__
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.hierarchical = hierarchical
        self.num_groups = num_groups
        self.depth = embed_dim // num_heads

        super().__init__(**kwargs)

    def build(self, input_shape) -> None:
        """Build the layer based on input shape.

        Args:
            input_shape: Shape of the input tensor.
        """
        # Number of features is the last dimension of the input.
        self.num_features = int(input_shape[-1])
        logger.debug(
            "Building AdvancedGraphFeatureLayer with {} features.",
            self.num_features,
        )

        # Projection layer: project each scalar feature to an embedding vector.
        self.projection = layers.Dense(
            self.embed_dim,
            activation=None,
            name="projection",
            kernel_initializer=initializers.GlorotUniform(seed=42),
        )

        # Edge attention dense layer for edge attribute learning.
        # It takes as input the concatenation of: h_i, h_j, and |h_i - h_j|,
        # where each has dimension `depth`. Total input dimension: 3 * depth.
        self.edge_attention_dense = layers.Dense(
            1,
            activation=None,
            use_bias=False,
            name="edge_attention_dense",
            kernel_initializer=initializers.GlorotUniform(seed=42),
        )
        self.leaky_relu = layers.LeakyReLU(alpha=0.2)
        self.dropout = (
            layers.Dropout(self.dropout_rate)
            if self.dropout_rate > 0
            else lambda x, training=None: x
        )

        # Final projection: maps the flattened aggregated embedding back to the original feature space.
        self.out_proj = layers.Dense(
            self.num_features,
            activation=None,
            name="out_proj",
            kernel_initializer=initializers.GlorotUniform(seed=42),
        )

        # Hierarchical aggregation components.
        if self.hierarchical:
            # Learned grouping matrix: shape (num_features, num_groups)
            self.grouping_matrix = self.add_weight(
                name="grouping_matrix",
                shape=(self.num_features, self.num_groups),
                initializer=initializers.GlorotUniform(seed=42),
                trainable=True,
            )
            # Group expander: maps from (num_groups * embed_dim) back to (num_features * embed_dim)
            self.group_expander = layers.Dense(
                self.num_features * self.embed_dim,
                activation=None,
                name="group_expander",
                kernel_initializer=initializers.GlorotUniform(seed=42),
            )

        # Final layer normalization.
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6, name="layer_norm")
        super().build(input_shape)

    def call(self, inputs: KerasTensor, training: bool | None = None) -> KerasTensor:
        """Args:
            inputs (KerasTensor): Input tensor of shape (batch, num_features).
            training (bool, optional): Whether the call is in training mode.

        Returns:
            KerasTensor: Output tensor of shape (batch, num_features).
        """
        batch_size = ops.shape(inputs)[0]
        # Step 1: Project scalar inputs to embeddings.
        # New shape: (batch, num_features, embed_dim)
        h = self.projection(ops.expand_dims(inputs, -1))

        # Step 2: Reshape for multi-head attention.
        # Reshape to (batch, num_features, num_heads, depth) then transpose to (batch, num_heads, num_features, depth).
        h = ops.reshape(h, (batch_size, self.num_features, self.num_heads, self.depth))
        h = ops.transpose(h, axes=[0, 2, 1, 3])

        # Step 3: Compute pairwise edge features for attention.
        # h_i: (batch, num_heads, num_features, num_features, depth)
        h_i = ops.repeat(ops.expand_dims(h, axis=3), repeats=self.num_features, axis=3)
        # h_j: (batch, num_heads, num_features, num_features, depth)
        h_j = ops.repeat(ops.expand_dims(h, axis=2), repeats=self.num_features, axis=2)
        # Compute edge features: concatenate h_i, h_j and the absolute difference |h_i - h_j|.
        edge_features = ops.concatenate([h_i, h_j, ops.abs(h_i - h_j)], axis=-1)
        # Shape: (batch, num_heads, num_features, num_features, 3 * depth)

        # Step 4: Compute raw attention scores using the edge_attention_dense layer.
        raw_scores = self.edge_attention_dense(
            edge_features,
        )  # (batch, num_heads, num_features, num_features, 1)
        raw_scores = ops.squeeze(
            raw_scores,
            axis=-1,
        )  # (batch, num_heads, num_features, num_features)
        raw_scores = self.leaky_relu(raw_scores)
        # Normalize attention scores with softmax over the last axis (neighbor dimension).
        attention = ops.nn.softmax(raw_scores, axis=-1)
        attention = self.dropout(attention, training=training)

        # Store attention weights for testing
        self._last_attention = attention

        # Step 5: Aggregate feature embeddings using computed attention.
        # Multiply attention (batch, num_heads, num_features, num_features) with h
        # (batch, num_heads, num_features, depth).
        aggregated = ops.matmul(attention, h)  # (batch, num_heads, num_features, depth)
        # Reassemble multi-head outputs: transpose and reshape back to (batch, num_features, embed_dim).
        aggregated = ops.transpose(aggregated, axes=[0, 2, 1, 3])
        aggregated = ops.reshape(
            aggregated,
            (batch_size, self.num_features, self.embed_dim),
        )

        # Step 6 (Optional): Hierarchical aggregation.
        if self.hierarchical:
            # Normalize grouping matrix over groups.
            grouping_weights = ops.nn.softmax(
                self.grouping_matrix,
                axis=-1,
            )  # (num_features, num_groups)
            # Compute group representations as a weighted sum over features.
            # Transpose aggregated to (batch, embed_dim, num_features) and multiply by grouping_weights.
            group_repr = ops.matmul(
                ops.transpose(aggregated, axes=[0, 2, 1]),
                grouping_weights,
            )
            # Transpose back to (batch, num_groups, embed_dim).
            group_repr = ops.transpose(group_repr, axes=[0, 2, 1])
            # Flatten and expand back to feature space.
            group_flat = ops.reshape(
                group_repr,
                (batch_size, -1),
            )  # (batch, num_groups * embed_dim)
            hierarchical_features = self.group_expander(group_flat)
            hierarchical_features = ops.reshape(
                hierarchical_features,
                (batch_size, self.num_features, self.embed_dim),
            )
            # Fuse hierarchical aggregated features with the multi-head aggregated features.
            aggregated = aggregated + hierarchical_features

        # Step 7: Residual connection.
        # Re-project inputs to the same embedding space (reuse the projection layer).
        input_proj = self.projection(ops.expand_dims(inputs, -1))
        aggregated = aggregated + input_proj

        # Step 8: Layer normalization.
        normalized = self.layer_norm(aggregated)

        # Step 9: Final projection.
        # Flatten the normalized tensor and project back to original feature space.
        normalized_flat = ops.reshape(normalized, (batch_size, -1))
        output = self.out_proj(normalized_flat)
        return output

    def get_config(self) -> dict:
        """Returns the configuration of the layer.

        This method is used to serialize the layer and restore it later.

        Returns:
            dict: A dictionary containing the configuration of the layer.
        """
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
                "hierarchical": self.hierarchical,
                "num_groups": self.num_groups if self.hierarchical else None,
            },
        )
        return config

    def compute_output_shape(self, input_shape) -> tuple[int, ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple (batch_size, num_features)

        Returns:
            Output shape tuple (batch_size, num_features)
        """
        return input_shape
