from typing import Any

import tensorflow as tf
from keras import layers, models


class GatedFeatureSelection(layers.Layer):
    """Gated feature selection with residual connection."""

    def __init__(self, input_dim: int, reduction_ratio: int = 4) -> None:
        """Initialize the gated feature selection layer.

        Args:
            input_dim: Input dimension
            reduction_ratio: Reduction ratio for the gate network
        """
        super().__init__()
        self.input_dim = input_dim
        self.reduction_ratio = reduction_ratio
        
        # More powerful gate network with skip connection
        hidden_dim = max(input_dim // reduction_ratio, 1)
        self.gate_net = models.Sequential([
            layers.Dense(hidden_dim, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(hidden_dim, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(input_dim, activation="sigmoid"),
        ])

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply gated feature selection with residual connection.

        Args:
            inputs: Input tensor of shape [batch_size, input_dim]

        Returns:
            Gated feature tensor of shape [batch_size, input_dim]
        """
        # Compute feature gates
        gates = self.gate_net(inputs)
        
        # Residual connection with gating
        return inputs * gates + 0.1 * inputs  # Small residual to maintain gradient flow

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Layer configuration dictionary
        """
        return {"input_dim": self.input_dim, "reduction_ratio": self.reduction_ratio}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GatedFeatureSelection":
        """Create layer from configuration.

        Args:
            config: Layer configuration dictionary

        Returns:
            GatedFeatureSelection instance
        """
        return cls(**config)


class SparseAttentionWeighting(layers.Layer):
    """Sparse attention mechanism with temperature scaling."""

    def __init__(self, num_modules: int, temperature: float = 1.0) -> None:
        """Initialize sparse attention weighting.

        Args:
            num_modules: Number of modules to weight
            temperature: Temperature for softmax scaling
        """
        super().__init__()
        self.num_modules = num_modules
        self.temperature = temperature
        
        # Learnable attention weights
        self.attention_weights = self.add_weight(
            shape=(num_modules,),
            initializer="ones",
            trainable=True,
            name="attention_weights",
        )

    def call(self, module_outputs: list[tf.Tensor]) -> tf.Tensor:
        """Apply sparse attention weighting.

        Args:
            module_outputs: List of module output tensors

        Returns:
            Weighted sum of module outputs
        """
        # Temperature-scaled softmax for sharper attention
        attention_probs = tf.nn.softmax(self.attention_weights / self.temperature)
        
        # Stack and weight module outputs
        stacked_outputs = tf.stack(module_outputs, axis=1)
        attention_weights = tf.expand_dims(tf.expand_dims(attention_probs, 0), -1)
        
        # Weighted combination with residual
        weighted_sum = tf.reduce_sum(stacked_outputs * attention_weights, axis=1)
        return weighted_sum

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Layer configuration dictionary
        """
        return {"num_modules": self.num_modules, "temperature": self.temperature}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SparseAttentionWeighting":
        """Create layer from configuration.

        Args:
            config: Layer configuration dictionary

        Returns:
            SparseAttentionWeighting instance
        """
        return cls(**config)


class ColumnAttention(layers.Layer):
    """Column attention mechanism to weight features dynamically."""

    def __init__(self, input_dim: int) -> None:
        """Initialize column attention.

        Args:
            input_dim: Input dimension
        """
        super().__init__()
        self.input_dim = input_dim
        
        # Two-layer attention mechanism for better feature interaction
        self.attention_net = models.Sequential([
            layers.Dense(max(input_dim // 2, 1), activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(input_dim, activation="softmax"),
        ])

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply column attention.

        Args:
            inputs: Input tensor of shape [batch_size, input_dim]

        Returns:
            Attention weighted tensor of shape [batch_size, input_dim]
        """
        # Compute attention weights with shape [batch_size, input_dim]
        attention_weights = self.attention_net(inputs)
        
        # Apply attention
        return inputs * attention_weights

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Layer configuration dictionary
        """
        return {"input_dim": self.input_dim}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ColumnAttention":
        """Create layer from configuration.

        Args:
            config: Layer configuration dictionary

        Returns:
            ColumnAttention instance
        """
        return cls(**config)


class RowAttention(layers.Layer):
    """Row attention mechanism for sample importance weighting."""

    def __init__(self) -> None:
        """Initialize row attention."""
        super().__init__()
        self.attention = models.Sequential([
            layers.Dense(32, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(1, activation="sigmoid"),  # Changed to sigmoid for stable training
        ])

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply row attention.

        Args:
            inputs: Input tensor of shape [batch_size, input_dim]

        Returns:
            Attention weighted tensor of shape [batch_size, input_dim]
        """
        # Compute attention scores [batch_size, 1]
        attention_scores = self.attention(inputs)
        
        # Apply attention with residual
        return inputs * attention_scores + 0.1 * inputs

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Layer configuration dictionary
        """
        return {}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RowAttention":
        """Create layer from configuration.

        Args:
            config: Layer configuration dictionary

        Returns:
            RowAttention instance
        """
        return cls(**config)


class FeatureCutout(layers.Layer):
    """Feature Cutout to randomly mask features during training."""

    def __init__(self, cutout_prob: float = 0.2) -> None:
        """Initialize feature cutout.

        Args:
            cutout_prob: Probability of masking features
        """
        super().__init__()
        self.cutout_prob = cutout_prob

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Apply feature cutout.

        Args:
            inputs: Input tensor
            training: Whether in training mode

        Returns:
            Masked tensor
        """
        if training:
            mask = tf.random.uniform(shape=tf.shape(inputs)) > self.cutout_prob
            return inputs * tf.cast(mask, inputs.dtype)
        return inputs

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Layer configuration dictionary
        """
        return {"cutout_prob": self.cutout_prob}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FeatureCutout":
        """Create layer from configuration.

        Args:
            config: Layer configuration dictionary

        Returns:
            FeatureCutout instance
        """
        return cls(**config)


class StochasticDepth(layers.Layer):
    """Stochastic depth to randomly drop residual modules."""

    def __init__(self, survival_prob: float = 0.8) -> None:
        """Initialize stochastic depth.

        Args:
            survival_prob: Probability of keeping the residual path
        """
        super().__init__()
        self.survival_prob = survival_prob

    def call(self, inputs: tf.Tensor, residual: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Apply stochastic depth.

        Args:
            inputs: Input tensor
            residual: Residual tensor
            training: Whether in training mode

        Returns:
            Output tensor
        """

        def drop_path() -> tf.Tensor:
            # Generate random number between 0 and 1
            random_tensor = tf.random.uniform([], dtype=tf.float32)
            keep_prob = tf.cast(random_tensor < self.survival_prob, dtype=tf.float32)
            return inputs + (residual * keep_prob)

        def regular_path() -> tf.Tensor:
            return inputs + residual

        return tf.cond(tf.cast(training, tf.bool), drop_path, regular_path)

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Layer configuration dictionary
        """
        return {
            "survival_prob": self.survival_prob,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "StochasticDepth":
        """Create layer from configuration.

        Args:
            config: Layer configuration dictionary

        Returns:
            StochasticDepth instance
        """
        return cls(**config)


class DifferentialBoostingLoss(tf.keras.losses.Loss):
    """Custom loss function for differential boosting.
    
    This loss function combines:
    1. MSE for accurate predictions
    2. L1 regularization on module weights for sparsity
    3. Diversity penalty to encourage different modules to learn different patterns
    """

    def __init__(
        self,
        reg_lambda: float = 0.01,
        diversity_weight: float = 0.05,
        smoothness_weight: float = 0.005
    ) -> None:
        """Initialize loss function.

        Args:
            reg_lambda: L1 regularization strength
            diversity_weight: Weight for diversity penalty
            smoothness_weight: Weight for prediction smoothness
        """
        super().__init__(reduction=tf.keras.losses.Reduction.NONE)  # Use NONE reduction for more control
        self.reg_lambda = reg_lambda
        self.diversity_weight = diversity_weight
        self.smoothness_weight = smoothness_weight

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None) -> tf.Tensor:
        """Compute loss value.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            sample_weight: Optional sample weights

        Returns:
            Loss value per sample
        """
        # Basic MSE loss per sample
        mse_loss = tf.square(y_true - y_pred)
        
        # Final loss is just MSE for now
        # The model's train_step will add regularization
        total_loss = mse_loss
        
        # Apply sample weights if provided
        if sample_weight is not None:
            total_loss = total_loss * sample_weight
        
        return total_loss

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration."""
        return {
            "reg_lambda": self.reg_lambda,
            "diversity_weight": self.diversity_weight,
            "smoothness_weight": self.smoothness_weight,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DifferentialBoostingLoss":
        """Create layer from configuration.

        Args:
            config: Layer configuration dictionary

        Returns:
            Layer instance
        """
        return cls(**config)


class DifferentialBoostingNN(models.Model):
    """Differential Boosting Neural Network with advanced mechanisms."""

    def __init__(
        self,
        input_dim: int,
        num_modules: int = 5,
        hidden_dim: int = 64,
        reg_lambda: float = 0.01,
        survival_prob: float = 0.9,
        learning_rate: float = 5e-4,
        gradient_clip: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_modules = num_modules
        self.hidden_dim = hidden_dim
        self.reg_lambda = reg_lambda
        self.survival_prob = survival_prob
        self.learning_rate = learning_rate
        self.gradient_clip = gradient_clip
        self.last_inputs = None
        
        # Input normalization
        self.input_norm = layers.BatchNormalization(name="input_normalization")
        
        # Initialize stochastic depth
        self.stochastic_depth = StochasticDepth(survival_prob)
        
        # Create modules with increasing capacity
        self.modules = []
        self.module_weights = self.add_weight(
            shape=(self.num_modules,),
            initializer=tf.keras.initializers.Constant(1.0 / num_modules),
            trainable=True,
            name="module_weights",
        )

        for i in range(num_modules):
            # Increase capacity for later modules
            current_hidden_dim = hidden_dim * (i + 1)
            
            module = models.Sequential([
                layers.BatchNormalization(name=f"module_{i}_input_norm"),
                ColumnAttention(input_dim),
                GatedFeatureSelection(input_dim),
                layers.Dense(current_hidden_dim, activation=None),
                layers.BatchNormalization(name=f"module_{i}_hidden_norm_1"),
                layers.Activation("relu"),
                layers.Dropout(0.1),
                layers.Dense(current_hidden_dim // 2, activation=None),
                layers.BatchNormalization(name=f"module_{i}_hidden_norm_2"),
                layers.Activation("relu"),
                layers.Dense(1, activation=None),
            ], name=f"module_{i}")
            self.modules.append(module)

        # Attention with lower temperature for sharper focus
        self.attention = SparseAttentionWeighting(num_modules, temperature=0.1)
        
        # Optimizer with gradient clipping and weight decay
        self.optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.learning_rate,
            weight_decay=1e-4,
            clipnorm=self.gradient_clip,
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass implementing gradient boosting."""
        self.last_inputs = inputs
        batch_size = tf.shape(inputs)[0]
        current_prediction = tf.zeros([batch_size, 1])
        module_predictions = []
        
        # Normalize inputs using pre-created layer
        inputs = self.input_norm(inputs, training=training)
        
        for i, module in enumerate(self.modules):
            # Each module predicts residual
            residual_pred = module(inputs, training=training)
            
            if training:
                residual_pred = self.stochastic_depth(
                    tf.zeros_like(residual_pred),
                    residual_pred,
                    training=training,
                )
            
            # Scale prediction with softmax-normalized weights
            scaled_pred = residual_pred * tf.nn.softmax(self.module_weights)[i]
            module_predictions.append(scaled_pred)
            
            # Update current prediction
            current_prediction += scaled_pred
        
        # Combine predictions with attention
        final_prediction = self.attention(module_predictions)
        
        return final_prediction

    def train_step(self, data: tuple[tf.Tensor, tf.Tensor]) -> dict[str, float]:
        """Custom training step with regularization and gradient clipping.

        Args:
            data: Tuple of (inputs, targets)

        Returns:
            Dictionary of metric results
        """
        x, y = data
        
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(x, training=True)
            
            # Get the basic loss (MSE)
            loss = self.compiled_loss(y, y_pred)
            
            # Add L1 regularization on module weights
            l1_loss = self.reg_lambda * tf.reduce_sum(tf.abs(self.module_weights))
            l1_loss = l1_loss / tf.cast(tf.shape(y)[0], tf.float32)
            
            # Get module predictions for diversity loss
            module_preds = []
            current_pred = tf.zeros_like(y)
            
            for i, module in enumerate(self.modules):
                residual = module(x, training=True)
                scaled_residual = residual * tf.nn.softmax(self.module_weights)[i]
                current_pred += scaled_residual
                module_preds.append(current_pred)
            
            # Stack predictions [batch_size, num_modules, 1]
            stacked_preds = tf.stack(module_preds, axis=1)
            
            # Diversity loss between consecutive modules
            diffs = stacked_preds[:, 1:] - stacked_preds[:, :-1]
            diversity_loss = -0.05 * tf.reduce_mean(tf.square(diffs))
            
            # Smoothness loss
            smoothness_loss = 0.005 * tf.reduce_mean(tf.abs(diffs))
            
            # Total loss
            total_loss = tf.reduce_mean(loss) + l1_loss + diversity_loss + smoothness_loss
        
        # Compute gradients
        gradients = tape.gradient(total_loss, self.trainable_variables)
        
        # Clip gradients
        gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clip)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)
        
        # Return metrics
        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = total_loss
        return results

    def get_config(self) -> dict[str, Any]:
        """Get model configuration."""
        return {
            "input_dim": self.input_dim,
            "num_modules": self.num_modules,
            "hidden_dim": self.hidden_dim,
            "reg_lambda": self.reg_lambda,
            "survival_prob": self.survival_prob,
            "learning_rate": self.learning_rate,
            "gradient_clip": self.gradient_clip,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DifferentialBoostingNN":
        """Create model from configuration.

        Args:
            config: Model configuration dictionary

        Returns:
            DifferentialBoostingNN instance
        """
        return cls(**config)
