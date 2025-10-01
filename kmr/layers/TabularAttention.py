"""
This module implements a TabularAttention layer that applies inter-feature and inter-sample
attention mechanisms for tabular data. It's particularly useful for capturing complex
relationships between features and samples in tabular datasets.
"""

from typing import Any, Dict
from loguru import logger
from keras import layers, ops
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kmr.layers._base_layer import BaseLayer

@register_keras_serializable(package="kmr.layers")
class TabularAttention(BaseLayer):
    """Custom layer to apply inter-feature and inter-sample attention for tabular data.
    
    This layer implements a dual attention mechanism:
    1. Inter-feature attention: Captures dependencies between features for each sample
    2. Inter-sample attention: Captures dependencies between samples for each feature
    
    The layer uses MultiHeadAttention for both attention mechanisms and includes
    layer normalization, dropout, and a feed-forward network.
    
    Args:
        num_heads (int): Number of attention heads
        d_model (int): Dimensionality of the attention model
        dropout_rate (float): Dropout rate for regularization
        name (str, optional): Name for the layer
        
    Input shape:
        Tensor with shape: `(batch_size, num_samples, num_features)`
        
    Output shape:
        Tensor with shape: `(batch_size, num_samples, d_model)`
        
    Example:
        ```python
        import keras
        from kmr.layers import TabularAttention
        
        # Create sample input data
        x = keras.random.normal((32, 100, 20))  # 32 batches, 100 samples, 20 features
        
        # Apply tabular attention
        attention = TabularAttention(num_heads=4, d_model=32, dropout_rate=0.1)
        y = attention(x)
        print("Output shape:", y.shape)  # (32, 100, 32)
        ```
    """
    
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        dropout_rate: float = 0.1,
        name: str | None = None,
        **kwargs: Any
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
        self.input_projection = None
        self.feature_attention = None
        self.feature_layernorm = None
        self.feature_dropout = None
        self.feature_layernorm2 = None
        self.feature_dropout2 = None
        self.sample_attention = None
        self.sample_layernorm = None
        self.sample_dropout = None
        self.sample_layernorm2 = None
        self.sample_dropout2 = None
        self.ffn_dense1 = None
        self.ffn_dense2 = None
        self.output_projection = None
        
        # Call parent's __init__ after setting public attributes
        super().__init__(name=name, **kwargs)
    
    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if not isinstance(self._num_heads, int) or self._num_heads <= 0:
            raise ValueError(f"num_heads must be a positive integer, got {self._num_heads}")
        
        if not isinstance(self._d_model, int) or self._d_model <= 0:
            raise ValueError(f"d_model must be a positive integer, got {self._d_model}")
        
        if not isinstance(self._dropout_rate, float) or not 0 <= self._dropout_rate < 1:
            raise ValueError(f"dropout_rate must be a float between 0 and 1, got {self._dropout_rate}")
        
        # Check that d_model is divisible by num_heads
        if self._d_model % self._num_heads != 0:
            raise ValueError(f"d_model ({self._d_model}) must be divisible by num_heads ({self._num_heads})")
    
    def build(self, input_shape: tuple[int, ...]) -> None:
        """Builds the layer with the given input shape.
        
        Args:
            input_shape: Tuple of integers defining the input shape.
        """
        if len(input_shape) != 3:
            raise ValueError(
                f"Input tensor must be 3-dimensional (batch_size, num_samples, num_features), "
                f"got shape {input_shape}"
            )
        
        self.input_dim = input_shape[-1]
        
        # Attention layers
        self.input_projection = layers.Dense(self.d_model)
        self.feature_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.dropout_rate
        )
        self.sample_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.dropout_rate
        )
        
        # Feed-forward network
        self.ffn_dense1 = layers.Dense(self.d_model, activation="relu")
        self.ffn_dense2 = layers.Dense(self.d_model)
        
        # Normalization and dropout
        self.feature_layernorm = layers.LayerNormalization()
        self.feature_dropout = layers.Dropout(self.dropout_rate)
        self.feature_layernorm2 = layers.LayerNormalization()
        self.feature_dropout2 = layers.Dropout(self.dropout_rate)
        
        self.sample_layernorm = layers.LayerNormalization()
        self.sample_dropout = layers.Dropout(self.dropout_rate)
        self.sample_layernorm2 = layers.LayerNormalization()
        self.sample_dropout2 = layers.Dropout(self.dropout_rate)
        
        self.output_projection = layers.Dense(self.d_model)
        
        logger.debug(f"TabularAttention built with d_model={self.d_model}, num_heads={self.num_heads}")
        super().build(input_shape)
    
    def call(self, inputs: KerasTensor, training: bool = False) -> KerasTensor:
        """Forward pass of the layer.
        
        Args:
            inputs: Input tensor of shape (batch_size, num_samples, num_features)
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.
                
        Returns:
            Output tensor of shape (batch_size, num_samples, d_model)
        """
        if len(inputs.shape) != 3:
            raise ValueError(
                "Input tensor must be 3-dimensional (batch_size, num_samples, num_features)"
            )
        
        # Project inputs to d_model dimension
        projected = self.input_projection(inputs)
        
        # Inter-feature attention: across columns (features)
        features = self.feature_attention(
            projected, projected, projected, training=training
        )
        features = self.feature_layernorm(
            projected + self.feature_dropout(features, training=training)
        )
        
        # Apply feed-forward network to features
        features_ffn = self.ffn_dense2(self.ffn_dense1(features))
        features = self.feature_layernorm2(
            features + self.feature_dropout2(features_ffn, training=training)
        )
        
        # Inter-sample attention: across rows (samples)
        # Transpose for sample attention
        samples = ops.transpose(features, [0, 2, 1])
        samples = self.sample_attention(
            samples, samples, samples, training=training
        )
        samples = ops.transpose(samples, [0, 2, 1])
        samples = self.sample_layernorm(
            features + self.sample_dropout(samples, training=training)
        )
        
        # Apply feed-forward network to samples
        samples_ffn = self.ffn_dense2(self.ffn_dense1(samples))
        outputs = self.sample_layernorm2(
            samples + self.sample_dropout2(samples_ffn, training=training)
        )
        
        return outputs
    
    def compute_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Compute the output shape of the layer.
        
        Args:
            input_shape: Shape of the input tensor.
            
        Returns:
            Shape of the output tensor.
        """
        return (input_shape[0], input_shape[1], self.d_model)
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration.
        
        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "d_model": self.d_model,
            "dropout_rate": self.dropout_rate,
        })
        return config 