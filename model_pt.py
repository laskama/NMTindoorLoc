import torch
import torch.nn as nn
import numpy as np


class NMTindoorLocPT(nn.Module):

    def __init__(self, num_imu=6, num_ap=188, hidden_size=256):
        super(NMTindoorLocPT, self).__init__()

        self.encoder = Encoder(num_imu=num_imu, num_ap=num_ap, hidden_size=hidden_size)
        self.decoder = Decoder(decoder_hidden=hidden_size)

    def forward(self, inputs):
        imu_input, rss_input, dec_input = inputs

        enc_out, init_state = self.encoder([imu_input, rss_input])

        pos_seq, _, _ = self.decoder([enc_out, dec_input, init_state])

        return pos_seq

    def predict(self, inputs):

        imu_input, rss_input = inputs

        enc_out, init_state = self.encoder([imu_input, rss_input])

        # create dummy tensor
        dummy_pos = torch.Tensor(np.random.random((len(imu_input), 10, 2)))

        pos_seq, _, _ = self.decoder([enc_out, dummy_pos, init_state])

        return pos_seq


class Encoder(nn.Module):

    def __init__(self, num_imu, num_ap, hidden_size=256, bidirectional=False):
        super(Encoder, self).__init__()

        self.bidirectional = bidirectional

        self.rss_encoder = nn.Linear(in_features=num_ap, out_features=hidden_size)

        self.joint_encoder = nn.LSTM(input_size=num_imu + hidden_size, hidden_size=hidden_size, bidirectional=False, batch_first=True)

    def forward(self, inputs):
        imu_input, rss_input = inputs
        rss = self.rss_encoder(rss_input)

        # concat processed rss with imu input
        encoder_inputs = torch.concat([imu_input, rss], dim=-1)

        encoder_outputs, (final_state_h, final_state_c) = self.joint_encoder(encoder_inputs)

        # reshape from (1, b, h) -> (b, h)
        final_state_h = torch.squeeze(final_state_h, dim=0)
        final_state_c = torch.squeeze(final_state_c, dim=0)

        return encoder_outputs, [final_state_h, final_state_c]


class Decoder(nn.Module):

    def __init__(self, decoder_hidden=256, num_coords=2, out_seq_length=10):
        super(Decoder, self).__init__()

        self.out_seq_length = out_seq_length

        self.decoder = nn.LSTMCell(input_size=num_coords, hidden_size=decoder_hidden)
        self.out_proj = nn.Linear(in_features=decoder_hidden, out_features=num_coords)

    def forward(self, inputs):
        _, decoder_inputs, initial_decoder_state = inputs

        pos_seq, (h, c) = self.decode(decoder_inputs, initial_decoder_state, teacher_forced=self.training)

        return pos_seq, h, c

    def decode(self, target, initial_decoder_state, teacher_forced=True):
        dec_state = initial_decoder_state
        combined_outputs = []

        pos = torch.zeros((len(target), 2))

        for Y_t in torch.split(target, split_size_or_sections=1, dim=1):
            if teacher_forced:
                Y_t = torch.squeeze(Y_t)
            else:
                Y_t = pos

            dec_state = self.decoder(Y_t, dec_state)
            h, c = dec_state

            pos = self.out_proj(h)

            combined_outputs.append(pos)

        combined_outputs = torch.stack(combined_outputs, dim=1)

        return combined_outputs, (h, c)


if __name__ == '__main__':
    rss_input = torch.Tensor(np.random.random((32, 200, 188)))
    imu_input = torch.Tensor(np.random.random((32, 200, 6)))
    pos = torch.Tensor(np.random.random((32, 10, 2)))

    nmt = NMTindoorLocPT(num_imu=6, num_ap=188, hidden_size=256)

    train_pos = nmt([imu_input, rss_input, pos])

    test_pos = nmt.predict([imu_input, rss_input])
