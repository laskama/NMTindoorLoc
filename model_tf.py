import tensorflow as tf
import numpy as np


class NMTindoorLoc(tf.keras.Model):
    """
    Seq2seq model that transforms RSS+IMU sources to position sequences.
    """
    def __init__(self, hidden_size=256, bidirectional_encoder=False, cross_attention=False, **kwargs):
        """
        Allows for specifying the model architecture
        :param hidden_size: Number of hidden units used in LSTM and Dense hidden layers
        :param bidirectional_encoder: Whether to use a bidirectional encoder on the source sequence (RSS+IMU)
        :param cross_attention: Whether to add cross attention between decoder and encoder outputs
        :param kwargs:
        """
        super(NMTindoorLoc, self).__init__(**kwargs)

        self.encoder = Encoder(hidden_size=hidden_size, bidirectional=bidirectional_encoder)

        # since bidirectional decoder resulting states are concatenated
        decoder_hidden = 2 * hidden_size if bidirectional_encoder else hidden_size

        if cross_attention:
            self.decoder = DecoderWithCrossAttention(decoder_hidden=decoder_hidden)
        else:
            self.decoder = Decoder(decoder_hidden=decoder_hidden)

    def call(self, inputs):
        imu_input, rss_input, dec_input = inputs

        encoder_outputs, initial_dec_state = self.encoder([imu_input, rss_input])

        pos_seq, _, _ = self.decoder([encoder_outputs, dec_input, initial_dec_state])

        return pos_seq

    def predict(self, inputs):
        imu_input, rss_input = inputs

        encoder_outputs, initial_dec_state = self.encoder([imu_input, rss_input])

        pos_seq = self.decoder.predict([encoder_outputs, initial_dec_state], batch_size=len(encoder_outputs))

        return pos_seq


class Encoder(tf.keras.layers.Layer):
    """
    Sensor data encoder. Utilizes LSTM that receives concatenation of processed RSS data (via dense layer) and
    IMU data as input. Absolute information should be extracted from RSS, while IMU is required for relative position
    updates.
    """
    def __init__(self, hidden_size=256, bidirectional=False, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.rss_encoder = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hidden_size, activation='relu'))

        self.bidirectional = bidirectional

        if bidirectional:
            enc_fw = tf.keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True)
            enc_bw = tf.keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True, go_backwards=True)
            self.joint_encoder = tf.keras.layers.Bidirectional(enc_fw, backward_layer=enc_bw, name='LSTM_encoder')
        else:
            self.joint_encoder = tf.keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True, name='LSTM_encoder')

    def call(self, inputs):
        imu_input, rss_input = inputs
        rss = self.rss_encoder(rss_input)

        # concat processed rss with imu input
        encoder_inputs = tf.keras.layers.concatenate([imu_input, rss])

        if self.bidirectional:
            encoder_outputs, h_fw, c_fw, h_bw, c_bw, = self.joint_encoder(encoder_inputs)
            final_state_h = tf.keras.layers.concatenate([h_fw, h_bw])
            final_state_c = tf.keras.layers.concatenate([c_fw, c_bw])
        else:
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

            target_seq = output_tokens

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
