class VariableSelectionNetwork(layers.Layer):
    """Variable Selection Network (VSN).

    Applies an individual GRN to each input variable then learns soft weights
    over them to produce a combined representation.

    Args:
        input_sizes (dict): Mapping from variable name to input dimension.
        hidden_size (int): Hidden dimension for each GRN.
        dropout_rate (float): Dropout rate.
        context_size (int, optional): Optional context dimension.
    """
    def __init__(self, input_sizes: dict, hidden_size, dropout_rate=0.1, context_size=None, **kwargs):
        super().__init__(**kwargs)
        self.variable_names = list(input_sizes.keys())
        self.hidden_size = hidden_size
        self.context_size = context_size
        # Create a GRN for each variable
        self.grns = {var: GatedResidualNetwork(input_size=input_sizes[var],
                                                hidden_size=hidden_size,
                                                output_size=hidden_size,
                                                dropout_rate=dropout_rate,
                                                context_size=context_size)
                     for var in self.variable_names}
        # Weight network: will compute weights for each variable based on aggregated inputs.
        total_input_dim = sum(input_sizes[var] for var in self.variable_names)
        self.weight_dense = layers.Dense(len(self.variable_names))
        self.softmax = layers.Softmax(axis=-1)
    def call(self, inputs: dict, context=None, training=False):
        # inputs is a dict mapping variable name to tensor of shape (batch, time, input_dim)
        variable_outputs = []
        weight_inputs = []
        for var in self.variable_names:
            x = inputs[var]  # (batch, time, input_dim)
            # Apply variable-specific GRN
            x_trans = self.grns[var](x, context=context, training=training)  # (batch, time, hidden_size)
            variable_outputs.append(x_trans)
            # For weight calculation, we use the last time step (or mean over time)
            weight_inputs.append(x[:, -1, :])  # (batch, input_dim)
        # Concatenate along last axis: (batch, sum(input_dim))
        weight_concat = tf.concat(weight_inputs, axis=-1)
        raw_weights = self.weight_dense(weight_concat)  # (batch, num_vars)
        var_weights = self.softmax(raw_weights)  # (batch, num_vars)
        # Expand and stack variable outputs: shape (batch, time, num_vars, hidden_size)
        var_weights_exp = tf.expand_dims(var_weights, axis=1)  # (batch, 1, num_vars)
        stacked = tf.stack(variable_outputs, axis=2)  # (batch, time, num_vars, hidden_size)
        weighted_sum = tf.reduce_sum(stacked * tf.expand_dims(var_weights_exp, axis=-1), axis=2)
        return weighted_sum