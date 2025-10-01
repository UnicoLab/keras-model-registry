# standard library

# pypi/conda library
import tensorflow as tf

# project plugin
from config import ModelConfig

mc = ModelConfig()


class FeedBackModel(tf.keras.Model):
    """
    Auto-regressive model for time-series forecasting based on Recurrent Neural Networks.

    Args:
        tf (_type_): _description_
    """

    def __init__(
        self,
        units: int = mc.RECURRENT_UNITS_NR,
        out_steps: int = mc.OUTPUT_STEPS,
        nr_features_out: int = mc.NR_OUTPUT_FEATURES,
    ):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.nr_features_out = nr_features_out
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(self.nr_features_out)

    def warmup(self, inputs: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """This method returns a single time-step prediction and the internal state of the LSTM:"""
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
        """Aggregating multiple single steps into a single time-step prediction"""
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the LSTM state.
        prediction, state = self.warmup(inputs)

        # Insert the first prediction.
        predictions.append(prediction)

        # Run the rest of the prediction steps.
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state, training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions
