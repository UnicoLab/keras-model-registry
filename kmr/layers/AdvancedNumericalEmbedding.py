"""
This module implements an AdvancedNumericalEmbedding layer that embeds continuous numerical features
into a higher-dimensional space using a combination of continuous and discrete branches.
"""

from typing import Any, Dict, List, Union
from keras import layers, ops
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kmr.layers._base_layer import BaseLayer

@register_keras_serializable(package="kmr.layers")
class AdvancedNumericalEmbedding(BaseLayer):
    """Advanced numerical embedding layer for continuous features.

    This layer embeds each continuous numerical feature into a higher-dimensional space by
    combining two branches:

      1. Continuous Branch: Each feature is processed via a small MLP.
      2. Discrete Branch: Each feature is discretized into bins using learnable min/max boundaries
         and then an embedding is looked up for its bin.

    A learnable gate combines the two branch outputs per feature and per embedding dimension.
    Additionally, the continuous branch uses a residual connection and optional batch normalization
    to improve training stability.

    Args:
        embedding_dim (int): Output embedding dimension per feature.
        mlp_hidden_units (int): Hidden units for the continuous branch MLP.
        num_bins (int): Number of bins for discretization.
        init_min (float or list): Initial minimum values for discretization boundaries. If a scalar is
            provided, it is applied to all features.
        init_max (float or list): Initial maximum values for discretization boundaries.
        dropout_rate (float): Dropout rate applied to the continuous branch.
        use_batch_norm (bool): Whether to apply batch normalization to the continuous branch.
        name (str, optional): Name for the layer.

    Input shape:
        Tensor with shape: `(batch_size, num_features)`

    Output shape:
        Tensor with shape: `(batch_size, num_features, embedding_dim)` or
        `(batch_size, embedding_dim)` if num_features=1

    Example:
        ```python
        import keras
        from kmr.layers import AdvancedNumericalEmbedding
        
        # Create sample input data
        x = keras.random.normal((32, 5))  # 32 samples, 5 features
        
        # Create the layer
        embedding = AdvancedNumericalEmbedding(
            embedding_dim=8,
            mlp_hidden_units=16,
            num_bins=10
        )
        y = embedding(x)
        print("Output shape:", y.shape)  # (32, 5, 8)
        ```
    """

    def __init__(
        self,
        embedding_dim: int = 8,
        mlp_hidden_units: int = 16,
        num_bins: int = 10,
        init_min: Union[float, List[float]] = -3.0,
        init_max: Union[float, List[float]] = 3.0,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        name: str | None = None,
        **kwargs: Any
    ) -> None:
        # Set private attributes first
        self._embedding_dim = embedding_dim
        self._mlp_hidden_units = mlp_hidden_units
        self._num_bins = num_bins
        self._init_min = init_min
        self._init_max = init_max
        self._dropout_rate = dropout_rate
        self._use_batch_norm = use_batch_norm
        
        # Validate parameters
        self._validate_params()
        
        # Set public attributes BEFORE calling parent's __init__
        self.embedding_dim = self._embedding_dim
        self.mlp_hidden_units = self._mlp_hidden_units
        self.num_bins = self._num_bins
        self.init_min = self._init_min
        self.init_max = self._init_max
        self.dropout_rate = self._dropout_rate
        self.use_batch_norm = self._use_batch_norm
        
        # Initialize instance variables
        self.num_features = None
        self.hidden_layer = None
        self.output_layer = None
        self.dropout_layer = None
        self.batch_norm = None
        self.residual_proj = None
        self.bin_embeddings = []
        self.learned_min = None
        self.learned_max = None
        self.gate = None
        
        # Call parent's __init__ after setting public attributes
        super().__init__(name=name, **kwargs)
    
    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if not isinstance(self._embedding_dim, int) or self._embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be a positive integer, got {self._embedding_dim}")
        if not isinstance(self._mlp_hidden_units, int) or self._mlp_hidden_units <= 0:
            raise ValueError(f"mlp_hidden_units must be a positive integer, got {self._mlp_hidden_units}")
        if not isinstance(self._num_bins, int) or self._num_bins <= 0:
            raise ValueError(f"num_bins must be a positive integer, got {self._num_bins}")
        if not isinstance(self._dropout_rate, (int, float)) or not 0 <= self._dropout_rate < 1:
            raise ValueError(f"dropout_rate must be between 0 and 1, got {self._dropout_rate}")
        if not isinstance(self._use_batch_norm, bool):
            raise ValueError(f"use_batch_norm must be a boolean, got {self._use_batch_norm}")
    
    def build(self, input_shape: tuple[int, ...]) -> None:
        """Builds the layer with the given input shape.
        
        Args:
            input_shape: Tuple of integers defining the input shape.
        """
        # Get number of features
        self.num_features = input_shape[-1]
        
        # Build continuous branch
        self._build_continuous_branch()
        
        # Build dropout layer
        if self.dropout_rate > 0:
            self.dropout_layer = layers.Dropout(self.dropout_rate)
        
        # Build batch normalization layer
        if self.use_batch_norm:
            self.batch_norm = layers.BatchNormalization()
        
        # Build residual projection
        self.residual_proj = layers.Dense(self.embedding_dim)
        
        # Build discrete branch
        self._build_discrete_branch()
        
        # Build gate
        self.gate = self.add_weight(
            name="gate",
            shape=(self.num_features, self.embedding_dim),
            initializer="zeros",
            trainable=True
        )
        
        super().build(input_shape)
    
    def _build_continuous_branch(self) -> None:
        """Build the continuous branch MLP."""
        self.hidden_layer = layers.Dense(self.mlp_hidden_units, activation="relu")
        self.output_layer = layers.Dense(self.embedding_dim)
    
    def _build_discrete_branch(self) -> None:
        """Build the discrete branch with bin embeddings and learned boundaries."""
        # Create embeddings for each feature
        self.bin_embeddings = []
        for i in range(self.num_features):
            embed_layer = layers.Embedding(
                input_dim=self.num_bins,
                output_dim=self.embedding_dim,
                name=f"bin_embed_{i}"
            )
            self.bin_embeddings.append(embed_layer)
        
        # Create learned boundaries
        # Convert init_min and init_max to lists if they are scalars
        init_min_list = [self.init_min] * self.num_features if isinstance(self.init_min, (int, float)) else self.init_min
        init_max_list = [self.init_max] * self.num_features if isinstance(self.init_max, (int, float)) else self.init_max
        
        # Ensure lists have the correct length
        if len(init_min_list) != self.num_features:
            raise ValueError(f"init_min list length ({len(init_min_list)}) must match num_features ({self.num_features})")
        if len(init_max_list) != self.num_features:
            raise ValueError(f"init_max list length ({len(init_max_list)}) must match num_features ({self.num_features})")
        
        # Create learned min and max weights
        self.learned_min = self.add_weight(
            name="learned_min",
            shape=(self.num_features,),
            initializer=lambda shape, dtype: ops.convert_to_tensor(init_min_list, dtype="float32"),
            trainable=True
        )
        
        self.learned_max = self.add_weight(
            name="learned_max",
            shape=(self.num_features,),
            initializer=lambda shape, dtype: ops.convert_to_tensor(init_max_list, dtype="float32"),
            trainable=True
        )
    
    def call(self, inputs: KerasTensor, training: bool = False) -> KerasTensor:
        """Forward pass of the layer.
        
        Args:
            inputs: Input tensor with shape (batch_size, num_features).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.
                
        Returns:
            Output tensor with shape (batch_size, num_features, embedding_dim) or
            (batch_size, embedding_dim) if num_features=1.
        """
        # Cast inputs to float32
        inputs = ops.cast(inputs, "float32")
        
        # Process continuous branch
        # Reshape inputs for feature-wise processing
        batch_size = ops.shape(inputs)[0]
        inputs_reshaped = ops.reshape(inputs, (batch_size * self.num_features, 1))
        
        # Apply MLP
        cont = self.hidden_layer(inputs_reshaped)
        cont = self.output_layer(cont)
        
        # Apply dropout if needed
        if self.dropout_layer is not None:
            cont = self.dropout_layer(cont, training=training)
        
        # Apply batch normalization if needed
        if self.batch_norm is not None:
            cont = self.batch_norm(cont, training=training)
        
        # Apply residual connection
        cont_res = self.residual_proj(inputs_reshaped)
        cont = cont + cont_res
        
        # Reshape back to (batch_size, num_features, embedding_dim)
        cont = ops.reshape(cont, (batch_size, self.num_features, self.embedding_dim))
        
        # Process discrete branch
        # Scale inputs to [0, 1] using learned boundaries
        inputs_expanded = ops.expand_dims(inputs, axis=-1)  # (batch_size, num_features, 1)
        min_expanded = ops.expand_dims(self.learned_min, axis=0)  # (1, num_features)
        max_expanded = ops.expand_dims(self.learned_max, axis=0)  # (1, num_features)
        
        scaled = (inputs - min_expanded) / (max_expanded - min_expanded + 1e-6)
        
        # Compute bin indices
        bin_indices = ops.floor(scaled * self.num_bins)
        bin_indices = ops.cast(bin_indices, "int32")
        bin_indices = ops.clip(bin_indices, 0, self.num_bins - 1)
        
        # Get embeddings for each feature
        disc_embeddings = []
        for i in range(self.num_features):
            # Extract bin indices for feature i
            feat_bins = bin_indices[:, i]
            # Get embeddings
            feat_embed = self.bin_embeddings[i](feat_bins)
            disc_embeddings.append(feat_embed)
        
        # Stack embeddings along feature dimension
        disc = ops.stack(disc_embeddings, axis=1)  # (batch_size, num_features, embedding_dim)
        
        # Combine branches via gate
        gate_sigmoid = ops.sigmoid(self.gate)  # (num_features, embedding_dim)
        gate_expanded = ops.expand_dims(gate_sigmoid, axis=0)  # (1, num_features, embedding_dim)
        
        output = gate_expanded * cont + (1 - gate_expanded) * disc
        
        # If only one feature, squeeze the feature dimension
        if self.num_features == 1:
            output = ops.squeeze(output, axis=1)
        
        return output
    
    def compute_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Compute the output shape of the layer.
        
        Args:
            input_shape: Shape of the input tensor.
            
        Returns:
            Shape of the output tensor.
        """
        if self.num_features == 1:
            return input_shape[:-1] + (self.embedding_dim,)
        else:
            return input_shape[:-1] + (self.num_features, self.embedding_dim)
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration.
        
        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "embedding_dim": self.embedding_dim,
            "mlp_hidden_units": self.mlp_hidden_units,
            "num_bins": self.num_bins,
            "init_min": self.init_min,
            "init_max": self.init_max,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
        })
        return config 