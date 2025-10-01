import tensorflow as tf
from tensorflow.keras import layers
from loguru import logger

class NeuralAdditiveModel(layers.Layer):
    """Neural Additive Model (NAM) layer for tabular data.

    This layer processes each feature individually using a small MLP and outputs
    a scalar contribution per feature. The final prediction is the sum of these contributions,
    optionally plus a global bias.

    This additive structure allows for interpretability while capturing nonlinear relationships.

    Example:
        ```python
        import tensorflow as tf

        # Dummy data: 32 samples, 8 features.
        x = tf.random.normal((32, 8))
        nam_layer = NeuralAdditiveModel(hidden_units=16)
        y = nam_layer(x)
        print("NAM output shape:", y.shape)  # Expected: (32, 1) for regression.
        ```
    """
    def __init__(self, hidden_units: int = 16, **kwargs):
        """
        Args:
            hidden_units (int): Number of hidden units in each feature's MLP.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.hidden_units = hidden_units

    def build(self, input_shape):
        self.num_features = input_shape[-1]
        # Create one small MLP per feature.
        self.feature_mlps = []
        for i in range(self.num_features):
            mlp = tf.keras.Sequential([
                layers.Dense(self.hidden_units, activation="relu", name=f"mlp_{i}_dense1"),
                layers.Dense(1, activation=None, name=f"mlp_{i}_dense2")
            ], name=f"mlp_{i}")
            self.feature_mlps.append(mlp)
        # Global bias
        self.bias = self.add_weight(
            name="global_bias", shape=(1,), initializer="zeros", trainable=True
        )
        logger.debug("NeuralAdditiveModel built with {} features and {} hidden units per feature.",
                     self.num_features, self.hidden_units)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Process each feature separately.
        feature_outputs = []
        for i in range(self.num_features):
            # Extract i-th feature (shape: [batch, 1])
            feature = tf.expand_dims(inputs[:, i], axis=-1)
            # Apply the feature-specific MLP.
            out = self.feature_mlps[i](feature)
            feature_outputs.append(out)
        # Sum contributions and add global bias.
        output = tf.add_n(feature_outputs) + self.bias  # shape: (batch, 1)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_units": self.hidden_units})
        return config
