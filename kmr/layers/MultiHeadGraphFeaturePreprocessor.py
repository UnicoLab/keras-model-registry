"""
This module implements a MultiHeadGraphFeaturePreprocessor layer that treats features as nodes
in a graph and learns multiple "views" (heads) of the feature interactions via self-attention.
This approach is useful for tabular data where complex feature relationships need to be captured.
"""

from typing import Any

from keras import layers, ops
from keras import KerasTensor
from keras.saving import register_keras_serializable
from loguru import logger

from kmr.layers._base_layer import BaseLayer

@register_keras_serializable(package="kmr.layers")
class MultiHeadGraphFeaturePreprocessor(BaseLayer):
    """Multi-head graph-based feature preprocessor for tabular data.

    This layer treats each feature as a node and applies multi-head self-attention
    to capture and aggregate complex interactions among features. The process is:
    
    1. Project each scalar input into an embedding of dimension `embed_dim`.
    2. Split the embedding into `num_heads` heads.
    3. For each head, compute queries, keys, and values and calculate scaled dot-product
       attention across the feature dimension.
    4. Concatenate the head outputs, project back to the original feature dimension,
       and add a residual connection.
    
    This mechanism allows the network to learn multiple relational views among features,
    which can significantly boost performance on tabular data.
    
    Args:
        embed_dim: Dimension of the feature embeddings. Default is 16.
        num_heads: Number of attention heads. Default is 4.
        dropout_rate: Dropout rate applied to attention weights. Default is 0.0.
        name: Optional name for the layer.
    
    Input shape:
        2D tensor with shape: `(batch_size, num_features)`
        
    Output shape:
        2D tensor with shape: `(batch_size, num_features)` (same as input)
    
    Example:
        ```python
        import keras
        from kmr.layers import MultiHeadGraphFeaturePreprocessor
        
        # Tabular data with 10 features
        x = keras.random.normal((32, 10))
        
        # Create the layer with 16-dim embeddings and 4 attention heads
        graph_preproc = MultiHeadGraphFeaturePreprocessor(embed_dim=16, num_heads=4)
        y = graph_preproc(x, training=True)
        print("Output shape:", y.shape)  # Expected: (32, 10)
        ```
    """
    
    def __init__(
        self,
        embed_dim: int = 16,
        num_heads: int = 4,
        dropout_rate: float = 0.0,
        name: str | None = None,
        **kwargs: Any
    ) -> None:
        # Set public attributes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        # Initialize instance variables
        self.projection = None
        self.q_dense = None
        self.k_dense = None
        self.v_dense = None
        self.out_proj = None
        self.final_dense = None
        self.dropout_layer = None
        self.num_features = None
        self.depth = None
        
        # Validate parameters
        self._validate_params()
        
        # Call parent's __init__
        super().__init__(name=name, **kwargs)
    
    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {self.embed_dim}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})")
        if self.dropout_rate < 0 or self.dropout_rate >= 1:
            raise ValueError(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")
    
    def build(self, input_shape: tuple) -> None:
        """Build the layer.

        Args:
            input_shape: Shape of the input tensor.
        """
        self.num_features = input_shape[1]
        logger.debug(f"MultiHeadGraphFeaturePreprocessor built with num_features={self.num_features}, "
                    f"embed_dim={self.embed_dim}, num_heads={self.num_heads}")

        # Projection layer to convert scalar features to embeddings
        self.projection = layers.Dense(self.embed_dim)

        # Multi-head attention components
        self.depth = self.embed_dim // self.num_heads
        self.q_dense = layers.Dense(self.embed_dim)
        self.k_dense = layers.Dense(self.embed_dim)
        self.v_dense = layers.Dense(self.embed_dim)
        self.out_proj = layers.Dense(1)

        # Final dense layer for feature transformation
        self.final_dense = layers.Dense(self.num_features)

        # Dropout layer if needed
        if self.dropout_rate > 0:
            self.dropout_layer = layers.Dropout(self.dropout_rate)
        else:
            self.dropout_layer = None

        super().build(input_shape)
    
    def split_heads(self, x: KerasTensor, batch_size: KerasTensor) -> KerasTensor:
        """Split the last dimension into (num_heads, depth) and transpose.
        
        Args:
            x: Input tensor with shape (batch_size, num_features, embed_dim).
            batch_size: Batch size tensor.
            
        Returns:
            Tensor with shape (batch_size, num_heads, num_features, depth).
        """
        # Get the actual number of features from the input tensor
        actual_num_features = ops.shape(x)[1]
        
        x = ops.reshape(x, (batch_size, actual_num_features, self.num_heads, self.depth))
        return ops.transpose(x, (0, 2, 1, 3))
    
    def call(self, inputs: KerasTensor, training: bool | None = None) -> KerasTensor:
        """Forward pass of the layer.
        
        Args:
            inputs: Input tensor with shape (batch_size, num_features).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.
                
        Returns:
            Output tensor with the same shape as input.
        """
        # Get batch size and actual number of features
        batch_size = ops.shape(inputs)[0]
        actual_num_features = ops.shape(inputs)[1]
        
        # If the number of features has changed, we'll handle it differently
        if actual_num_features != self.num_features:
            use_final_dense = False
        else:
            use_final_dense = True

        # Project input features to embeddings: shape (batch, num_features, embed_dim)
        embeddings = self.projection(ops.expand_dims(inputs, -1))
        
        # Compute Q, K, V: shape (batch, num_features, embed_dim)
        q = self.q_dense(embeddings)
        k = self.k_dense(embeddings)
        v = self.v_dense(embeddings)
        
        # Split heads: shape (batch, num_heads, num_features, depth)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled dot-product attention
        # Compute scores: (batch, num_heads, num_features, num_features)
        dk = ops.cast(self.depth, "float32")
        # Transpose k to align dimensions for attention calculation
        k_transposed = ops.transpose(k, (0, 1, 3, 2))
        scores = ops.matmul(q, k_transposed) / ops.sqrt(dk)
        
        # Softmax over the features dimension
        attention_weights = ops.softmax(scores, axis=-1)
        
        # Apply dropout if needed
        if self.dropout_rate > 0 and self.dropout_layer is not None:
            attention_weights = self.dropout_layer(attention_weights, training=training)
        
        # Weighted sum of values: (batch, num_heads, num_features, depth)
        attention_output = ops.matmul(attention_weights, v)
        
        # Concatenate heads: (batch, num_features, embed_dim)
        attention_output = ops.transpose(attention_output, (0, 2, 1, 3))
        concat_attention = ops.reshape(attention_output, (batch_size, actual_num_features, self.embed_dim))
        
        # Project the concatenated output back to scalar per feature
        projected = self.out_proj(concat_attention)  # (batch, num_features, 1)
        projected = ops.squeeze(projected, axis=-1)  # (batch, num_features)
        
        # Final dense layer with residual connection
        if use_final_dense:
            output = self.final_dense(projected) + inputs
        else:
            # Just add the projected output to the input directly
            output = projected + inputs
            
        return output
    
    def get_config(self) -> dict[str, Any]:
        """Returns the config of the layer.
        
        Returns:
            Python dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
        })
        return config 