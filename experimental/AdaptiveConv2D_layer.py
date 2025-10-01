import tensorflow as tf
from tensorflow.keras import layers

class AdaptiveConv2D(layers.Layer):
    """Adaptive convolution layer with a learnable Gaussian envelope.

    This layer applies a standard convolution operation where the kernel is modulated by a
    Gaussian envelope. The envelope's standard deviation (sigma) is trainable, allowing the layer
    to adapt the effective receptive field size.

    Example:
        >>> import tensorflow as tf
        >>> from adaptive_layers import AdaptiveConv2D
        >>> input_tensor = tf.random.normal((32, 64, 64, 3))  # Batch of 32, 64x64 images with 3 channels
        >>> adaptive_conv = AdaptiveConv2D(filters=16, kernel_size=3)
        >>> output_tensor = adaptive_conv(input_tensor)
        >>> print(output_tensor.shape)  # Expected shape: (32, 62, 62, 16) with VALID padding.
    """
    def __init__(self, filters: int, kernel_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape: tf.TensorShape) -> None:
        kernel_shape = (self.kernel_size, self.kernel_size, input_shape[-1], self.filters)
        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer="glorot_uniform",
            trainable=True,
        )
        # Learnable sigma for the Gaussian envelope (scalar)
        self.sigma = self.add_weight(
            name="sigma",
            shape=(1,),
            initializer=tf.constant_initializer(1.0),
            trainable=True,
        )
        # Create a coordinate grid for the kernel
        k = self.kernel_size
        coords = tf.stack(tf.meshgrid(tf.range(k), tf.range(k), indexing="ij"), axis=-1)
        self.coords = tf.cast(coords, tf.float32) - (k - 1) / 2.0  # Centered grid
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Compute Gaussian envelope: exp(-||coords||^2 / (2 * sigma^2))
        envelope = tf.exp(-tf.reduce_sum(tf.square(self.coords), axis=-1) / (2 * tf.square(self.sigma)))
        envelope = tf.expand_dims(envelope, axis=-1)  # Shape: (kernel_size, kernel_size, 1)
        # Modulate kernel weights with the envelope
        adaptive_kernel = self.kernel * envelope
        # Apply convolution using the adaptive kernel
        return tf.nn.conv2d(inputs, adaptive_kernel, strides=[1, 1, 1, 1], padding="VALID")

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"filters": self.filters, "kernel_size": self.kernel_size})
        return config
