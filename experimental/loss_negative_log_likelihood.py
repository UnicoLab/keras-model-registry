
from keras import ops
from keras import KerasTensor
from keras.saving import register_keras_serializable


@register_keras_serializable(package="kmr.losses")
def nll_loss(y_true: KerasTensor, y_pred: KerasTensor) -> KerasTensor:
    """Gaussian Negative Log-Likelihood loss.

    Assumes y_pred[..., 0] = μ and y_pred[..., 1] = log(σ²).
    """
    mu = y_pred[..., 0:1]
    log_var = y_pred[..., 1:2]
    precision = ops.exp(-log_var)
    loss = 0.5 * precision * ops.square(y_true - mu) + 0.5 * log_var
    return ops.mean(loss)