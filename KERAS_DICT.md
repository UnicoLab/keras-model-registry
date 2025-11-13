| TensorFlow | Keras 3.0 |
|---|---| 
| tf.nn.sigmoid_cross_entropy_with_logits | keras.ops.binary_crossentropy (mind the from_logits argument) |
| tf.nn.sparse_softmax_cross_entropy_with_logits | keras.ops.sparse_categorical_crossentropy (mind the from_logits argument) |
| tf.nn.sparse_softmax_cross_entropy_with_logits | keras.ops.categorical_crossentropy(target, output, from_logits=False, axis=-1) |
| tf.nn.conv1d, tf.nn.conv2d, tf.nn.conv3d, tf.nn.convolution | keras.ops.conv |
| tf.nn.conv_transpose, tf.nn.conv1d_transpose, tf.nn.conv2d_transpose, tf.nn.conv3d_transpose | keras.ops.conv_transpose |
| tf.nn.depthwise_conv2d | keras.ops.depthwise_conv |
| tf.nn.separable_conv2d | keras.ops.separable_conv |
| tf.nn.batch_normalization | keras.layers.BatchNormalization |
| tf.nn.dropout | keras.random.dropout |
| tf.nn.embedding_lookup | keras.ops.take |
| tf.nn.l2_normalize | keras.utils.normalize (not an op) |
| x.numpy | keras.ops.convert_to_numpy |
| tf.scatter_nd_update | keras.ops.scatter_update |
| tf.tensor_scatter_nd_update | keras.ops.slice_update |
| tf.signal.fft2d | keras.ops.fft2 |
| tf.signal.inverse_stft | keras.ops.istft |
| tf.image.crop_to_bounding_box | keras.ops.image.crop_images |
| tf.image.pad_to_bounding_box | keras.ops.image.pad_images |
| tf.abs | keras.ops.absolute |
| tf.reduce_all | keras.ops.all |
| tf.reduce_max | keras.ops.amax |
| tf.reduce_min | keras.ops.amin |
| tf.reduce_any | keras.ops.any |
| tf.concat | keras.ops.concatenate |
| tf.range | keras.ops.arange |
| tf.acos | keras.ops.arccos |
| tf.asin | keras.ops.arcsin |
| tf.asinh | keras.ops.arcsinh |
| tf.atan | keras.ops.arctan |
| tf.atan2 | keras.ops.arctan2 |
| tf.atanh | keras.ops.arctanh |
| tf.convert_to_tensor | keras.ops.convert_to_tensor |
| tf.reduce_mean | keras.ops.mean |
| tf.clip_by_value | keras.ops.clip |
| tf.math.conj | keras.ops.conjugate |
| tf.linalg.diag_part | keras.ops.diagonal |
| tf.reverse | keras.ops.flip |
| tf.gather | keras.ops.take |
| tf.math.is_finite | keras.ops.isfinite |
| tf.math.is_inf | keras.ops.isinf |
| tf.math.is_nan | keras.ops.isnan |
| tf.reduce_max | keras.ops.max |
| tf.reduce_mean | keras.ops.mean |
| tf.reduce_min | keras.ops.min |
| tf.rank | keras.ops.ndim |
| tf.math.pow | keras.ops.power |
| tf.reduce_prod | keras.ops.prod |
| tf.math.reduce_std | keras.ops.std |
| tf.reduce_sum | keras.ops.sum |
| tf.gather | keras.ops.take |
| tf.gather_nd | keras.ops.take_along_axis |
| tf.math.reduce_variance | keras.ops.var |


Additional hacks

```python
from keras import layers, models
from keras import KerasTensor
from typing import Any
from keras.saving import register_keras_serializable

@register_keras_serializable(package="kerasfactory.layers")
```