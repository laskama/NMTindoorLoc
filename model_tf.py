import tensorflow as tf
import numpy as np

from data_provider import START_TOKEN


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
        encoder = tf.keras.layers.LSTM(hidden_size, return_sequences=False, return_state=True, name='LSTM_encoder')
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    decoder_inputs = tf.keras.Input(shape=(None, num_coords))

    decoder_hidden = 2 * hidden_size if bidirectional_encoder else hidden_size
    decoder_lstm = tf.keras.layers.LSTM(decoder_hidden, return_sequences=True, return_state=True, name='LSTM_decoder')

    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    decoder_dense = tf.keras.layers.Dense(num_coords, activation='linear', name='Dense_decoder')

    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = tf.keras.Model([imu_input, rss_input, decoder_inputs], decoder_outputs)

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
