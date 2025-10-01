"""
This module implements a HyperZZWOperator layer that computes context-dependent weights
by multiplying inputs with hyper-kernels. This is a specialized layer for the Terminator model.
"""

from typing import Any, List
from loguru import logger
from keras import layers, ops
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kmr.layers._base_layer import BaseLayer

@register_keras_serializable(package="kmr.layers")
class HyperZZWOperator(BaseLayer):
    """A layer that computes context-dependent weights by multiplying inputs with hyper-kernels.
    
    This layer takes two inputs: the original input tensor and a context tensor.
    It generates hyper-kernels from the context and performs a context-dependent transformation
    of the input.
    
    Args:
        input_dim: Dimension of the input features.
        context_dim: Optional dimension of the context features. If not provided, it will be inferred.
        name: Optional name for the layer.
        
    Input:
        A list of two tensors:
        - inputs[0]: Input tensor with shape (batch_size, input_dim).
        - inputs[1]: Context tensor with shape (batch_size, context_dim).
        
    Output shape:
        2D tensor with shape: `(batch_size, input_dim)` (same as input)
        
    Example:
        ```python
        import keras
        from kmr.layers import HyperZZWOperator
        
        # Create sample input data
        inputs = keras.random.normal((32, 16))  # 32 samples, 16 features
        context = keras.random.normal((32, 8))  # 32 samples, 8 context features
        
        # Create the layer
        zzw_op = HyperZZWOperator(input_dim=16, context_dim=8)
        context_weights = zzw_op([inputs, context])
        print("Output shape:", context_weights.shape)  # (32, 16)
        ```
    """
    
    def __init__(
        self,
        input_dim: int,
        context_dim: int | None = None,
        name: str | None = None,
        **kwargs: Any
    ) -> None:
        # Set public attributes
        self.input_dim = input_dim
        self.context_dim = context_dim
        
        # Validate parameters
        self._validate_params()
        
        # Call parent's __init__
        super().__init__(name=name, **kwargs)
    
    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.context_dim is not None and self.context_dim <= 0:
            raise ValueError(f"context_dim must be positive, got {self.context_dim}")
    
    def build(self, input_shape: List[tuple[int, ...]]) -> None:
        """Builds the layer with the given input shape.
        
        Args:
            input_shape: List of tuples of integers defining the input shapes.
                input_shape[0] is the shape of the input tensor.
                input_shape[1] is the shape of the context tensor.
        """
        # Validate parameters again during build
        self._validate_params()
        
        # Extract shapes
        input_tensor_shape = input_shape[0]
        context_tensor_shape = input_shape[1]
        
        # Get context dimension from the shape
        context_dim = self.context_dim or context_tensor_shape[-1]
        
        # Create hyper-kernel weights - this is a tensor that will be used to generate
        # context-dependent kernels
        self.hyper_kernel = self.add_weight(
            name="hyper_kernel",
            shape=(context_dim, self.input_dim),  # Simplified shape for easier matrix multiplication
            initializer="glorot_uniform",
            trainable=True
        )
        
        logger.debug(f"HyperZZWOperator built with input_dim={self.input_dim}, context_dim={context_dim}")
        super().build(input_shape)
    
    def call(self, inputs: List[KerasTensor]) -> KerasTensor:
        """Forward pass of the layer.
        
        Args:
            inputs: A list of two tensors:
                inputs[0]: Input tensor with shape (batch_size, input_dim).
                inputs[1]: Context tensor with shape (batch_size, context_dim).
                
        Returns:
            Context-dependent weights tensor with the same shape as input.
        """
        # Unpack inputs
        input_tensor = inputs[0]
        context_tensor = inputs[1]
        
        logger.debug(f"HyperZZWOperator input_tensor shape: {input_tensor.shape}")
        logger.debug(f"HyperZZWOperator context_tensor shape: {context_tensor.shape}")
        
        # Validate input dimensions
        if input_tensor.shape[-1] != self.input_dim:
            raise ValueError(
                f"Input dimension must be {self.input_dim}, got {input_tensor.shape[-1]}."
            )
        
        # Generate context-dependent weights
        # context_tensor: (batch_size, context_dim)
        # hyper_kernel: (context_dim, input_dim)
        # context_weights: (batch_size, input_dim)
        context_weights = ops.matmul(context_tensor, self.hyper_kernel)
        
        # Apply context-dependent transformation (element-wise multiplication)
        # input_tensor: (batch_size, input_dim)
        # context_weights: (batch_size, input_dim)
        # output: (batch_size, input_dim)
        output = input_tensor * context_weights
        
        logger.debug(f"HyperZZWOperator output shape: {output.shape}")
        return output
    
    def get_config(self) -> dict[str, Any]:
        """Returns the config of the layer.
        
        Returns:
            Python dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "context_dim": self.context_dim,
        })
        return config 