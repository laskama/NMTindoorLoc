import tensorflow as tf
import numpy as np


class Encoder(tf.keras.layers.Layer):
    """
    Sensor data encoder. Utilizes LSTM that receives concatenation of processed RSS data (via dense layer) and
    IMU data as input. Absolute information should be extracted from RSS, while IMU is required for relative position
    updates.
    """
    def __init__(self, hidden_size=256, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.rss_encoder = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hidden_size, activation='relu'))
        # self.rss_dropout = tf.keras.layers.Dropout(0.5)

        self.joint_encoder = tf.keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True, name='LSTM_encoder')

    def call(self, inputs):
        rss_input, imu_input = inputs
        rss = self.rss_encoder(rss_input)

        # concat processed rss with imu input
        encoder_inputs = tf.keras.layers.concatenate([imu_input, rss])

        encoder_outputs, final_state_h, final_state_c = self.joint_encoder(encoder_inputs)

        return encoder_outputs, [final_state_h, final_state_c]


class Decoder(tf.keras.layers.Layer):
    """
    Decoder model that utilizes final encoder hidden states for initialization. Outputs fixed sequence of positions.
    The base decoder has no cross attention to the encoder outputs.
    """
    def __init__(self, decoder_hidden=256, num_coords=2, out_seq_length=10, **kwargs):
        super(Decoder, self).__init__(**kwargs)

        self.out_seq_length = out_seq_length

        self.decoder = tf.keras.layers.LSTM(decoder_hidden, return_sequences=True, return_state=True, name='LSTM_decoder')
        self.out_proj = tf.keras.layers.Dense(num_coords, activation='linear', name='Dense_decoder')

    def call(self, inputs):
        _, decoder_inputs, initial_decoder_state = inputs

        decoder_outputs, h, c = self.decoder(decoder_inputs, initial_state=initial_decoder_state)

        pos_seq = self.out_proj(decoder_outputs)

        return pos_seq, h, c

    def predict(self, inputs, batch_size=32):
        encoder_outputs, states_value = inputs

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((batch_size, 1, 2))

        decoded_traj = []
        for idx in range(10):

            output_tokens, h, c = self.call([encoder_outputs] + [target_seq] + [states_value])

            pos = output_tokens
            decoded_traj += [pos]

            # Update the target sequence (of length 1).
            target_seq = output_tokens

            # Update states
            states_value = [h, c]
        return np.concatenate(decoded_traj, axis=1)


class DecoderWithCrossAttention(tf.keras.layers.Layer):
    """
    Decoder model that utilizes final encoder hidden states for initialization. Outputs fixed sequence of positions.
    This variant has cross attention (dot-product) to the encoder outputs.
    """
    def __init__(self, decoder_hidden=256, num_coords=2, out_seq_length=10, **kwargs):
        super(DecoderWithCrossAttention, self).__init__(**kwargs)

        self.out_seq_length = out_seq_length

        self.decoder = tf.keras.layers.LSTM(decoder_hidden, return_sequences=True, return_state=True, name='LSTM_decoder')
        self.attention = tf.keras.layers.Attention()
        self.att_proj = tf.keras.layers.Dense(decoder_hidden, activation='relu')
        self.out_proj = tf.keras.layers.Dense(num_coords, activation='linear', name='Dense_decoder')
        self.train = True

    def call(self, inputs):
        encoder_outputs, decoder_inputs, initial_decoder_state = inputs

        decoder_outputs, h, c = self.decoder(decoder_inputs, initial_state=initial_decoder_state)

        # during prediction, we want to call the decoder for each time step individually
        if self.train:
            num_time_steps = self.out_seq_length
        else:
            num_time_steps = 1

        res = []
        for t in range(num_time_steps):
            decoder_output = decoder_outputs[:, t, :][:, None, :]
            att_vals = self.attention([decoder_output, encoder_outputs])

            # Concatenate attention output with decoder output and apply projection
            joint_state = tf.keras.layers.concatenate([att_vals, decoder_output])
            att_proj = self.att_proj(joint_state)

            pos_pred = self.out_proj(att_proj)
            res.append(pos_pred)

        pos_seq = tf.concat(res, axis=1)

        return pos_seq, h, c

    def predict(self, inputs, batch_size=32):
        encoder_outpus, states_value = inputs
        self.train = False

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((batch_size, 1, 2))

        decoded_traj = []
        for idx in range(self.out_seq_length):

            output_tokens, h, c = self.call([encoder_outpus] + [target_seq] + [states_value])

            pos = output_tokens
            decoded_traj += [pos]

            # Update the target sequence (of length 1).
            target_seq = output_tokens

            # Update states
            states_value = [h, c]

        self.train = True

        return np.concatenate(decoded_traj, axis=1)


class NMTindoorLoc(tf.keras.Model):

    def __init__(self, hidden_size=256, bidirectional_encoder=False, cross_attention=False, **kwargs):

        super(NMTindoorLoc, self).__init__(**kwargs)

        self.encoder = Encoder(hidden_size=hidden_size)

        if cross_attention:
            self.decoder = DecoderWithCrossAttention(decoder_hidden=hidden_size)
        else:
            self.decoder = Decoder(decoder_hidden=hidden_size)

    def call(self, inputs):
        imu_input, rss_input, dec_input = inputs

        encoder_outputs, initial_dec_state = self.encoder([imu_input, rss_input])

        pos_seq, _, _ = self.decoder([encoder_outputs, dec_input, initial_dec_state])

        return pos_seq

    def predict_seq(self, inputs):
        imu_input, rss_input = inputs

        encoder_outputs, initial_dec_state = self.encoder([imu_input, rss_input])

        pos_seq = self.decoder.predict([encoder_outputs, initial_dec_state], batch_size=len(encoder_outputs))

        return pos_seq


class LSTMCellReturnHiddenStateSequence(tf.keras.layers.LSTMCell):

    def call(self, inputs, states, training=None):
        # real_inputs = inputs[:, :self.units + 6]  # decouple [h, c]
        outputs, [h, c] = super().call(inputs, states, training=training)
        return h, [h, c]


def get_tf_model(num_ap, num_imu, hidden_size, seq_length, num_coords=2):
    rss_input = tf.keras.Input(shape=(seq_length, num_ap))

    rss_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hidden_size, activation='relu'))(rss_input)

    imu_input = tf.keras.Input(shape=(seq_length, num_imu))

    con_input = tf.keras.layers.concatenate([imu_input, rss_dense])

    out = tf.keras.layers.LSTM(hidden_size, return_sequences=False)(con_input)

    pred = tf.keras.layers.Dense(num_coords)(out)

    model = tf.keras.Model(inputs=(imu_input, rss_input), outputs=pred)

    return model


def get_tf_seq2seq_train_model(num_ap, num_imu, hidden_size, seq_length, out_seq_length, num_coords=2, bidirectional_encoder=False):
    rss_input = tf.keras.Input(shape=(None, num_ap))
    imu_input = tf.keras.Input(shape=(None, num_imu))

    rss_encoder = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hidden_size, activation='relu'))(rss_input)
    rss_encoder = tf.keras.layers.Dropout(0.5)(rss_encoder)
    rss_encoder = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hidden_size, activation='relu'))(rss_encoder)
    encoder_inputs = tf.keras.layers.concatenate([imu_input, rss_encoder])

    if bidirectional_encoder:
        encoder_fw = tf.keras.layers.LSTM(hidden_size, return_sequences=False, return_state=True)
        encoder_bw = tf.keras.layers.LSTM(hidden_size, return_sequences=False, return_state=True, go_backwards=True)
        encoder = tf.keras.layers.Bidirectional(encoder_fw, backward_layer=encoder_bw, name='LSTM_encoder')
        encoder_outputs, state_h_fw, state_c_fw, state_h_bw, state_c_bw, = encoder(encoder_inputs)
        state_h = tf.keras.layers.Concatenate()([state_h_fw, state_h_bw])
        state_c = tf.keras.layers.Concatenate()([state_c_fw, state_c_bw])
    else:
        # encoder = tf.keras.layers.RNN(LSTMCellReturnHiddenStateSequence(hidden_size), return_sequences=True, return_state=True)
        encoder = tf.keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True, name='LSTM_encoder')
        encoder_outputs, final_state_h, final_state_c = encoder(encoder_inputs)

    # setup initial state of decoder from final encoder state
    initial_dec_state = [final_state_h, final_state_c]

    decoder_inputs = tf.keras.Input(shape=(None, num_coords))

    decoder_hidden = 2 * hidden_size if bidirectional_encoder else hidden_size
    decoder_lstm = tf.keras.layers.LSTM(decoder_hidden, return_sequences=True, return_state=True, name='LSTM_decoder')

    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=initial_dec_state)

    # cross-attention between decoder outputs and encoder outputs
    attention = tf.keras.layers.Attention()
    decoder_dense = tf.keras.layers.Dense(num_coords, activation='linear', name='Dense_decoder')
    output_proj = tf.keras.layers.Dense(hidden_size, activation='relu')
    concat_layer = tf.keras.layers.Concatenate()

    res = []
    for t in range(10):
        decoder_output = decoder_outputs[:, t, :][:, None, :]
        att_vals = attention([decoder_output, encoder_outputs])

        # Concatenate attention output with decoder output and apply projection
        joint_state = concat_layer([att_vals, decoder_output])
        att_proj = output_proj(joint_state)

        pos_pred = decoder_dense(att_proj)  # decoder_outputs)
        res.append(pos_pred)

    pos_seq = tf.concat(res, axis=1)

    model = tf.keras.Model([imu_input, rss_input, decoder_inputs], pos_seq)

    return model


def get_tf_seq2seq_encoder_model(model, bidirectional_encoder=False):

    encoder_imu_inputs = model.input[0]  # input_1
    encoder_rss_inputs = model.input[1]  # input_2

    if bidirectional_encoder:
        encoder_outputs, state_h_fw, state_c_fw, state_h_bw, state_c_bw, = model.get_layer('LSTM_encoder').output
        state_h_enc = tf.keras.layers.Concatenate()([state_h_fw, state_h_bw])
        state_c_enc = tf.keras.layers.Concatenate()([state_c_fw, state_c_bw])
    else:
        encoder_outputs, state_h_enc, state_c_enc = model.get_layer('LSTM_encoder').output  # model.layers[5].output  # lstm_1

    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = tf.keras.Model((encoder_imu_inputs, encoder_rss_inputs), encoder_states)

    return encoder_model


def get_tf_seq2seq_decoder_model(model, hidden_size=256):

    decoder_inputs = model.input[2]  # input_3
    decoder_state_input_h = tf.keras.Input(shape=(hidden_size,))
    decoder_state_input_c = tf.keras.Input(shape=(hidden_size,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.get_layer('LSTM_decoder')  # model.layers[6]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.get_layer('Dense_decoder')  # model.layers[7]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
    )

    return decoder_model


def decode_sequence_new(input_seq, encoder_model, decoder_model, init_pos=None):
    # Encode the input as state vectors.
    _, states_value = encoder_model(input_seq, False)

    # Generate empty target sequence of length 1.
    if init_pos is None:
        target_seq = np.zeros((len(input_seq[0]), 1, 2))
    else:
        target_seq = init_pos

    # target_seq[0, 0, :] = START_TOKEN
    # Populate the first character of target sequence with the start character.
    # target_seq[0, 0, target_token_index["\t"]] = 1.0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_traj = []
    while not stop_condition:
        output_tokens, h, c = decoder_model([target_seq] + states_value, False)

        pos = output_tokens  # [0]
        decoded_traj += [pos]

        # Exit condition: either hit max length
        # or find stop character.
        if len(decoded_traj) > 9:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = output_tokens

        # Update states
        states_value = [h, c]
    return np.concatenate(decoded_traj, axis=1)


def decode_sequence(input_seq, encoder_model, decoder_model, init_pos=None):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    if init_pos is None:
        target_seq = np.zeros((len(input_seq[0]), 1, 2))
    else:
        target_seq = init_pos

    # target_seq[0, 0, :] = START_TOKEN
    # Populate the first character of target sequence with the start character.
    # target_seq[0, 0, target_token_index["\t"]] = 1.0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_traj = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        pos = output_tokens#[0]
        decoded_traj += [pos]

        # Exit condition: either hit max length
        # or find stop character.
        if len(decoded_traj) > 9:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = output_tokens

        # Update states
        states_value = [h, c]
    return np.concatenate(decoded_traj, axis=1)
