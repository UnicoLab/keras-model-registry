import tensorflow as tf
from keras import layers, models
from keras import KerasTensor
from keras.saving import register_keras_serializable


class TemporalFusionTransformerKerasFull(models.Model):
    """Full Temporal Fusion Transformer (TFT) in Keras with probabilistic estimation.

    This model integrates:
      - Static covariate processing: embedding static categorical variables and
        linear transformation of static numeric variables are combined via a VSN.
      - Variable selection networks for time-varying encoder and decoder inputs.
      - Static context encoders that produce context vectors used to initialize the LSTM encoder
        and enrich decoder outputs.
      - LSTM encoder and decoder for local temporal processing.
      - Multi-head attention to capture long-range dependencies.
      - Several GateAddNorm blocks and feed-forward GRN blocks.
      - Final output layer that predicts, for each forecast time step, [μ, log(σ²)].

    Args:
        encoder_length (int): Number of historical time steps.
        decoder_length (int): Forecast horizon.
        time_varying_encoder_variables (list[str]): Names of time-varying encoder variables.
        time_varying_decoder_variables (list[str]): Names of time-varying decoder variables.
        static_numeric_variables (list[str]): Names of static numeric variables.
        static_categorical_variables (list[str]): Names of static categorical variables.
        static_embedding_sizes (dict): Mapping for static categorical variables: var -> (num_categories, emb_dim).
        hidden_size (int): Hidden state dimension.
        lstm_layers (int): Number of LSTM layers.
        num_attention_heads (int): Number of attention heads.
        dropout_rate (float): Dropout rate.
        n_targets (int): Number of target series (default 1).
        loss_size (int): Number of output parameters per target (use 2 for [μ, log(σ²)]).
    """
    def __init__(
        self,
        encoder_length: int,
        decoder_length: int,
        time_varying_encoder_variables: list,
        time_varying_decoder_variables: list,
        static_numeric_variables: list,
        static_categorical_variables: list,
        static_embedding_sizes: dict,
        hidden_size: int = 64,
        lstm_layers: int = 1,
        num_attention_heads: int = 4,
        dropout_rate: float = 0.1,
        n_targets: int = 1,
        loss_size: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.n_targets = n_targets
        self.loss_size = loss_size  # usually 2 for probabilistic output [μ, log(σ²)]
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        # ---------------------------
        # Static covariates processing
        # ---------------------------
        # For static numeric variables, use a Dense layer to project to hidden size.
        self.static_numeric_dense = {
            var: layers.Dense(hidden_size, activation='elu')
            for var in static_numeric_variables
        }
        # For static categorical variables, use MultiEmbedding.
        self.static_embedding = MultiEmbedding(static_embedding_sizes)
        # After embedding, we flatten the embedding outputs.
        self.flatten = layers.Flatten()
        # Combine all static features: concatenate along last axis.
        # Then apply a Variable Selection Network for static covariates.
        # Prepare input sizes: assume each static numeric variable becomes hidden_size, and each embedding as given.
        static_input_sizes = {}
        for var in static_numeric_variables:
            static_input_sizes[var] = hidden_size
        for var in static_categorical_variables:
            # Get embedding dimension from static_embedding_sizes
            static_input_sizes[var] = static_embedding_sizes[var][1]
        self.static_vsn = VariableSelectionNetwork(
            input_sizes=static_input_sizes,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            context_size=None,
        )
        # GRNs to generate static context:
        self.static_context_grn = GatedResidualNetwork(hidden_size, hidden_size, dropout_rate=dropout_rate)
        self.static_context_hidden_encoder_grn = GatedResidualNetwork(hidden_size, hidden_size, dropout_rate=dropout_rate)
        self.static_context_cell_encoder_grn = GatedResidualNetwork(hidden_size, hidden_size, dropout_rate=dropout_rate)
        self.static_context_enrichment = GatedResidualNetwork(hidden_size, hidden_size, dropout_rate=dropout_rate)
        
        # ---------------------------
        # Time-varying (encoder/decoder) variable selection
        # ---------------------------
        # Assume each time-varying variable is scalar (input dim 1). Create dicts for input sizes.
        enc_input_sizes = {var: 1 for var in time_varying_encoder_variables}
        dec_input_sizes = {var: 1 for var in time_varying_decoder_variables}
        self.encoder_vsn = VariableSelectionNetwork(
            input_sizes=enc_input_sizes,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            context_size=hidden_size,  # use static context
        )
        self.decoder_vsn = VariableSelectionNetwork(
            input_sizes=dec_input_sizes,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            context_size=hidden_size,
        )

        # ---------------------------
        # LSTM encoder and decoder (local processing)
        # ---------------------------
        self.encoder_lstm = layers.LSTM(hidden_size, return_sequences=True, return_state=True,
                                        dropout=dropout_rate, recurrent_dropout=dropout_rate)
        self.decoder_lstm = layers.LSTM(hidden_size, return_sequences=True, return_state=True,
                                        dropout=dropout_rate, recurrent_dropout=dropout_rate)

        self.lstm_layers = lstm_layers  # currently only single layer is implemented for simplicity

        self.post_lstm_gan = GateAddNorm(hidden_size, dropout_rate=dropout_rate)

        # ---------------------------
        # Long-range processing: Multi-head attention
        # ---------------------------
        self.multihead_attn = InterpretableMultiHeadAttention(d_model=hidden_size, n_head=num_attention_heads, dropout_rate=dropout_rate)
        self.post_attn_gan = GateAddNorm(hidden_size, dropout_rate=dropout_rate)

        # ---------------------------
        # Feed-forward block (using GRN)
        # ---------------------------
        self.feed_forward_block = GatedResidualNetwork(hidden_size, hidden_size, dropout_rate=dropout_rate)
        self.ff_addnorm = layers.LayerNormalization()

        # ---------------------------
        # Pre-output processing
        # ---------------------------
        self.pre_output_gan = GateAddNorm(hidden_size, dropout_rate=0.0)  # no dropout here

        # Final output layer: outputs n_targets * loss_size (e.g., 1*2 = 2)
        self.output_layer = layers.Dense(n_targets * loss_size, activation=None)

    def call(self, inputs, training=False):
        """
        Args:
            inputs: Tuple containing:
                - encoder_data: dict mapping time_varying_encoder_variables to tensor (batch, encoder_length, 1)
                - decoder_data: dict mapping time_varying_decoder_variables to tensor (batch, decoder_length, 1)
                - static_numeric: dict mapping static numeric variable names to tensor (batch, 1)
                - static_categorical: dict mapping static categorical variable names to tensor (batch, 1) of ints.
            training: Boolean flag.
        Returns:
            Tensor of shape (batch, decoder_length, n_targets * loss_size) where last dim = [μ, log(σ²)].
        """
        encoder_data, decoder_data, static_numeric, static_categorical = inputs
        batch_size = tf.shape(next(iter(encoder_data.values())))[0]

        # Process static covariates:
        static_numeric_transformed = {}
        for var, tensor in static_numeric.items():
            static_numeric_transformed[var] = self.static_numeric_dense[var](tensor)  # (batch, hidden_size)
        static_cat_emb = self.static_embedding(static_categorical)  # dict: var -> (batch, 1, emb_dim)
        static_cat_transformed = {var: self.flatten(static_cat_emb[var]) for var in static_cat_emb}
        # Combine all static features into one dict:
        static_features = {}
        static_features.update(static_numeric_transformed)
        static_features.update(static_cat_transformed)
        # For variable selection, we need to add a time dimension. Expand each to (batch, 1, dim)
        for var in static_features:
            static_features[var] = tf.expand_dims(static_features[var], axis=1)
        # Apply static VSN: outputs (batch, 1, hidden_size)
        static_context = self.static_vsn(static_features, training=training)
        # Remove time dimension: (batch, hidden_size)
        static_context = tf.squeeze(static_context, axis=1)
        # Generate static contexts via GRNs:
        overall_static_context = self.static_context_grn(static_context, training=training)  # for enrichment
        hidden_encoder_context = self.static_context_hidden_encoder_grn(static_context, training=training)
        cell_encoder_context = self.static_context_cell_encoder_grn(static_context, training=training)
        enrichment_context = self.static_context_enrichment(static_context, training=training)
        
        # Expand static context to match time steps for later addition
        enrichment_context_exp = tf.expand_dims(enrichment_context, axis=1)  # (batch, 1, hidden_size)

        # Process time-varying encoder inputs via VSN (with static context as additional input)
        encoder_processed = self.encoder_vsn(encoder_data, context=tf.expand_dims(static_context, axis=1), training=training)
        # Process time-varying decoder inputs similarly
        decoder_processed = self.decoder_vsn(decoder_data, context=tf.expand_dims(static_context, axis=1), training=training)

        # LSTM encoder; note: here we use single-layer LSTM for simplicity
        enc_outputs, enc_h, enc_c = self.encoder_lstm(encoder_processed, training=training)
        # For decoder initial states, combine LSTM states with static context
        dec_init_h = hidden_encoder_context  # (batch, hidden_size)
        dec_init_c = cell_encoder_context      # (batch, hidden_size)
        dec_outputs, _, _ = self.decoder_lstm(decoder_processed, initial_state=[dec_init_h, dec_init_c], training=training)
        # Post LSTM GateAddNorm
        dec_outputs = self.post_lstm_gan(dec_outputs, dec_outputs, training=training)

        # Multi-head attention: query=decoder, key=value=encoder outputs
        attn_output = self.multihead_attn(query=dec_outputs, key=enc_outputs, value=enc_outputs, training=training)
        attn_output = self.post_attn_gan(dec_outputs, attn_output, training=training)

        # Add static enrichment (broadcast along time axis)
        static_enriched = attn_output + tf.tile(enrichment_context_exp, [1, self.decoder_length, 1])
        # Feed-forward block
        ff_output = self.feed_forward_block(static_enriched, training=training)
        ff_output = self.ff_addnorm(static_enriched + ff_output)
        # Pre-output gate-add-norm (no dropout)
        pre_out = self.pre_output_gan(ff_output, ff_output, training=training)
        # Final output layer
        out = self.output_layer(pre_out)
        # Reshape to (batch, decoder_length, n_targets, loss_size); here n_targets=1 so we can squeeze.
        out = tf.reshape(out, (batch_size, self.decoder_length, self.n_targets, self.loss_size))
        # For probabilistic forecasting, we return last dim as [μ, log(σ²)]
        return tf.squeeze(out, axis=2)