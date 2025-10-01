import tensorflow as tf
from tensorflow.keras import layers
from loguru import logger

class FeatureCrossLayer(layers.Layer):
    """Layer for generating feature crosses from tabular data.

    This layer computes pairwise multiplicative interactions between input features
    and concatenates these with the original features. An optional Dense projection
    can then compress the combined feature set to a desired output dimension.

    Example:
        ```python
        import tensorflow as tf
        from feature_cross_layer import FeatureCrossLayer

        # Suppose we have 8 features.
        x = tf.random.normal((32, 8))  # Batch of 32 samples.
        # Create a FeatureCrossLayer to produce an output of 16 features.
        cross_layer = FeatureCrossLayer(output_dim=16)
        y = cross_layer(x)
        print("Crossed feature output shape:", y.shape)  # Expected: (32, 16)
        ```
    """

    def __init__(self, output_dim: int, **kwargs):
        """
        Args:
            output_dim: Desired output dimension after combining original and cross features.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.num_features = input_shape[-1]
        # Number of pairwise interactions: nC2
        self.num_interactions = self.num_features * (self.num_features - 1) // 2
        # Define a Dense projection layer to compress the concatenated features.
        self.projection = layers.Dense(self.output_dim, activation=None, name="projection")
        logger.debug("FeatureCrossLayer built: {} original features, {} interactions.",
                     self.num_features, self.num_interactions)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Compute pairwise interactions.
        batch_size = tf.shape(inputs)[0]
        # Get indices for upper triangle (i < j)
        indices = []
        for i in range(self.num_features):
            for j in range(i + 1, self.num_features):
                indices.append((i, j))
        indices = tf.constant(indices, dtype=tf.int32)  # shape (num_interactions, 2)
        # Gather the pairs: shape (batch, num_interactions)
        cross_features = tf.gather(inputs, indices[:, 0], axis=1) * tf.gather(inputs, indices[:, 1], axis=1)
        # Concatenate original features and cross features.
        concatenated = tf.concat([inputs, cross_features], axis=1)
        # Project to desired output dimension.
        output = self.projection(concatenated)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"output_dim": self.output_dim})
        return config
