import tensorflow as tf
from tensorflow.keras import layers

class SqueezeExcite2D(layers.Layer):
    """Squeeze-and-Excitation block for 2D inputs.

    This block recalibrates channel-wise feature responses by modeling channel interdependencies.
    
    Example:
        >>> import tensorflow as tf
        >>> from squeeze_excite import SqueezeExcite2D
        >>> input_tensor = tf.random.normal((32, 64, 64, 128))  # Batch of 32, 64x64 images with 128 channels
        >>> se_block = SqueezeExcite2D(ratio=16)
        >>> output_tensor = se_block(input_tensor)
        >>> print(output_tensor.shape)  # Expected shape: (32, 64, 64, 128)
    """
    def __init__(self, ratio: int = 16, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape: tf.TensorShape) -> None:
        channel_dim = input_shape[-1]
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(channel_dim // self.ratio, activation="relu")
        self.dense2 = layers.Dense(channel_dim, activation="sigmoid")
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.global_avg_pool(inputs)
        x = tf.reshape(x, [-1, 1, 1, x.shape[-1]])
        x = self.dense1(x)
        x = self.dense2(x)
        return inputs * x

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config
