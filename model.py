import torch
import torch.nn as nn
from utils import get_padded_wlan_tensor


class HybridSeq2Point(nn.Module):
    def __init__(self, num_imu, num_ap, hidden_size, coord_dim=2):
        super(HybridSeq2Point, self).__init__()

        self.dense_distributed = nn.Linear(in_features=num_ap, out_features=hidden_size)

        self.imu_lstm = nn.LSTM(input_size=num_imu + hidden_size, hidden_size=hidden_size, bidirectional=False, batch_first=True)

        # output layer
        self.output_projection = nn.Linear(in_features=hidden_size, out_features=coord_dim, bias=True)

        self.apply(_weights_init)

    def forward(self, source_imu: torch.Tensor, source_rss: torch.Tensor) -> torch.Tensor:
        rss_output = torch.relu(self.dense_distributed(source_rss))

        lstm_input = torch.concat((source_imu, rss_output), dim=-1)

        lstm_output, (hn, cn) = self.imu_lstm(lstm_input)

        output = self.output_projection(lstm_output[:, -1, :])

        return output


class StaticBaseline(nn.Module):
    """
    Baseline model that is meant for validating whether absolute position can be extracted from static
    WLAN fingerprints. Uses plain fully-connected feed forward layers with linear output layer to transform
    to coordinate space.
    """
    def __init__(self, num_features, hidden_size, coord_dim=2, dropout_rate=0.2):

        super(StaticBaseline, self).__init__()

        self.h1 = nn.Linear(in_features=num_features, out_features=hidden_size)

        self.h2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)

        self.out = nn.Linear(in_features=hidden_size, out_features=coord_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, source: torch.Tensor, target: torch.Tensor):

        out = torch.relu(self.h1(source))

        out = torch.relu(self.h2(out))

        out = self.out(out)

        loss = nn.MSELoss(reduction='mean')

        score = loss(out, target)

        return score, out


class JointSeq2Point(nn.Module):
    """
    Model that predicts the final position for a joint input sequence (IMU + WLAN tensor).
    Only uses an LSTM encoder model on the joint input tensors and transforms the output via fully-connected
    dense layer and output projection into coordinate space.
    """
    def __init__(self, num_features, hidden_size, coord_dim=2, dropout_rate=0.2, seq_length=200):
        super(JointSeq2Point, self).__init__()

        # encoder for joint input tensor (IMU + WLAN)
        self.encoder = nn.LSTM(input_size=num_features, hidden_size=hidden_size, bidirectional=True, batch_first=True)

        # dense output layer
        self.output_dense = nn.Linear(in_features=2*hidden_size*seq_length, out_features=hidden_size, bias=True)

        # output layer
        self.output_projection = nn.Linear(in_features=hidden_size, out_features=coord_dim, bias=True)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        enc_hiddens, _ = self.encoder(source)

        enc_hiddens = torch.flatten(enc_hiddens, start_dim=1)

        out = self.output_dense(enc_hiddens)

        out = self.dropout(torch.tanh(out))

        out = self.output_projection(out)

        loss = nn.MSELoss(reduction='mean')

        score = loss(out, target)

        return score, out


class JointSeq2Seq(nn.Module):
    """
    Sequence to sequence model the uses single LSTM encoder on joint input sequences (IMU + WLAN tensor) and
    outputs corresponding fixed length output sequence of coordinates via LSTM decoder.
    """
    def __init__(self, num_features, hidden_size, coord_dim=2, dropout_rate=0.2):

        super(JointSeq2Seq, self).__init__()

        self.num_features = num_features
        self.hidden_size = hidden_size

        self.encoder = nn.LSTM(input_size=num_features, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.decoder = nn.LSTMCell(input_size=coord_dim + hidden_size, hidden_size=hidden_size)

        # for obtaining initial states of decoder
        self.h_projection = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size, bias=False)
        self.c_projection = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size, bias=False)

        # dense output layer
        self.output_dense = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)

        # output layer
        self.output_projection = nn.Linear(in_features=hidden_size, out_features=coord_dim, bias=True)

        self.dropout = nn.Dropout(dropout_rate)

    def encode(self, source: torch.Tensor):

        enc_hiddens, (last_hidden, last_cell) = self.encoder(source)

        # reshape from (2, b, h) -> (b, 2h)
        last_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        init_decoder_hidden = self.h_projection(last_hidden)

        # reshape from (2, b, h) -> (b, 2h)
        last_cell = torch.cat((last_cell[0], last_cell[1]), dim=1)
        init_decoder_cell = self.c_projection(last_cell)

        dec_init_state = (init_decoder_hidden, init_decoder_cell)

        return enc_hiddens, dec_init_state

    def decode(self, enc_hiddens, dec_init_state, target):

        # Initialize the decoder state (hidden and cell)
        dec_state = dec_init_state

        # Initialize previous combined output vector o_{t-1} as zero
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size)

        # Initialize a list we will use to collect the combined output o_t on each step
        combined_outputs = []

        for Y_t in torch.split(target, split_size_or_sections=1, dim=1):
            Y_t = torch.squeeze(Y_t)
            Ybar_t = torch.cat((Y_t, o_prev), dim=-1)

            (c_t, h_t), o_t = self.step(Ybar_t, dec_state)
            combined_outputs.append(o_t)
            o_prev = o_t

        combined_outputs = torch.stack(combined_outputs, dim=0)

        return combined_outputs

    def step(self, Ybar_t, dec_state):

        dec_state = self.decoder(Ybar_t, dec_state)
        dec_hidden, dec_cell = dec_state

        v_t = self.output_dense(dec_hidden)
        O_t = self.dropout(torch.tanh(v_t))

        combined_output = O_t

        return dec_state, combined_output

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        enc_hiddens, dec_init_state = self.encode(source)
        combined_outputs = self.decode(enc_hiddens, dec_init_state, target)

        o_t = self.output_projection(combined_outputs)

        o_t = torch.permute(o_t, (1, 0, 2))

        loss = nn.MSELoss(reduction='mean')

        score = loss(o_t[:, :-1, :], target[:, 1:, :])

        return score, o_t

    def predict(self, source, y_target, T_out=10):
        with torch.no_grad():
            enc_hiddens, dec_init_state = self.encoder(source)

            # Initialize previous combined output vector o_{t-1} as zero
            batch_size = enc_hiddens.size(0)
            o_prev = torch.zeros(batch_size, self.hidden_size)

            # take ground truth as initial input of decoder (should be noisy)
            Y_t = y_target[:, 0, :] + torch.randn(y_target.size(0), y_target.size(2))

            dec_state = dec_init_state

            # Initialize a list we will use to collect the combined output o_t on each step
            predictions = []

            for idx in range(T_out):
                Ybar_t = torch.cat((Y_t, o_prev), dim=-1)
                (c_t, h_t), o_t = self.step(Ybar_t, dec_state)
                Y_t = self.output_projection(o_t)

                predictions.append(Y_t)
                o_prev = o_t

            predictions = torch.stack(predictions, dim=0)
            predictions = torch.permute(predictions, (1, 0, 2))

        return predictions


class SingleFPencoder(JointSeq2Seq):
    """
    Multi-source model that uses a LSTM encoder for the IMU input sequence and a dense encoder on a single WLAN
    fingerprint for extracting the absolute location of the sequence. The encoder outputs are then concatenated
    and linearly projected to obtain the initial decoder states.
    """
    def __init__(self, num_imu, num_ap, hidden_size, coord_dim=2, dropout_rate=0.2):
        super(SingleFPencoder, self).__init__(num_features=num_imu + num_ap, hidden_size=hidden_size)

        self.num_imu = num_imu
        self.num_ap = num_ap
        self.hidden_size = hidden_size

        # imu encoder
        self.imu_encoder = nn.LSTM(input_size=num_imu, hidden_size=hidden_size, bidirectional=True, batch_first=True)

        # rss encoder
        self.rss_1 = nn.Linear(in_features=num_ap, out_features=hidden_size, bias=True)
        self.rss_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)

        # for obtaining initial states of decoder (resulting from rss+imu (bidirectional)
        # Linear projection of concatenation of [h_rss; h_cell_forward, h_cell_backward]
        self.h_projection = nn.Linear(in_features=3 * hidden_size, out_features=hidden_size, bias=False)
        # Linear projection of concatenation of [h_rss; c_cell_forward, c_cell_backward]
        self.c_projection = nn.Linear(in_features=3 * hidden_size, out_features=hidden_size, bias=False)

    def encode(self, imu_source: torch.Tensor, rss_source: torch.Tensor):
        # process imu with LSTM
        enc_hiddens, (last_hidden, last_cell) = self.imu_encoder(imu_source)

        # process wlan rss data with dense network
        h_rss = torch.relu(self.rss_1(rss_source))
        h_rss = torch.relu(self.rss_2(h_rss))

        # reshape from (2, b, h) -> (b, 2h)
        last_hidden = torch.cat((h_rss, last_hidden[0], last_hidden[1]), dim=1)
        init_decoder_hidden = self.h_projection(last_hidden)

        # reshape from (2, b, h) -> (b, 2h)
        last_cell = torch.cat((h_rss, last_cell[0], last_cell[1]), dim=1)
        init_decoder_cell = self.c_projection(last_cell)

        dec_init_state = (init_decoder_hidden, init_decoder_cell)

        return enc_hiddens, dec_init_state

    def forward(self, source_imu: torch.Tensor, source_rss: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        enc_hiddens, dec_init_state = self.encode(source_imu, source_rss)
        combined_outputs = self.decode(enc_hiddens, dec_init_state, target)

        o_t = self.output_projection(combined_outputs)

        o_t = torch.permute(o_t, (1, 0, 2))

        loss = nn.MSELoss(reduction='mean')

        score = loss(o_t[:, :-1, :], target[:, 1:, :])

        return score, o_t

    def predict(self, source_imu, source_rss, y_target, T_out=10):
        with torch.no_grad():
            enc_hiddens, dec_init_state = self.encode(source_imu, source_rss)

            # Initialize previous combined output vector o_{t-1} as zero
            batch_size = enc_hiddens.size(0)
            o_prev = torch.zeros(batch_size, self.hidden_size)

            # initialize input of decoder with START_TOKEN (assume no prior position is known)
            Y_t = y_target[:, 0, :] #q+ torch.randn(y_target.size(0), y_target.size(2))

            dec_state = dec_init_state

            # Initialize a list we will use to collect the combined output o_t on each step
            predictions = []

            for idx in range(T_out):
                Ybar_t = torch.cat((Y_t, o_prev), dim=-1)
                (c_t, h_t), o_t = self.step(Ybar_t, dec_state)
                Y_t = self.output_projection(o_t)

                predictions.append(Y_t)
                o_prev = o_t

            predictions = torch.stack(predictions, dim=0)
            predictions = torch.permute(predictions, (1, 0, 2))

        return predictions


class MultiSourceEncoder(SingleFPencoder):
    """
    Model that uses two separate LSTM encoder on IMU and WLAN data. Note that the WLAN sequence length
    (number of scans per IMU sequence) is much lower than the IMU sequence length.
    """
    def __init__(self, num_imu, num_ap, hidden_size, coord_dim=2, dropout_rate=0.2, loss_fnc=None):
        super(MultiSourceEncoder, self).__init__(num_imu, num_ap, hidden_size, coord_dim, dropout_rate)

        self.num_imu = num_imu
        self.num_ap = num_ap
        self.hidden_size = hidden_size

        # imu encoder
        self.imu_encoder = nn.LSTM(input_size=num_imu, hidden_size=hidden_size, bidirectional=True, batch_first=True)

        # rss encoder
        self.rss_encoder = nn.LSTM(input_size=num_ap, hidden_size=hidden_size, bidirectional=True)

        # for obtaining initial states of decoder (resulting from rss+imu (bidirectional)
        # Linear projection of concatenation of [h_rss; h_cell_forward, h_cell_backward]
        self.h_projection = nn.Linear(in_features=4 * hidden_size, out_features=hidden_size, bias=False)
        # Linear projection of concatenation of [h_rss; c_cell_forward, c_cell_backward]
        self.c_projection = nn.Linear(in_features=4 * hidden_size, out_features=hidden_size, bias=False)

        if loss_fnc is None:
            self.loss_fnc = nn.MSELoss(reduction='mean')
        else:
            self.loss_fnc = loss_fnc

    def forward(self, source_imu, source_rss, target) -> torch.Tensor:

        # convert to tensor
        source_imu = torch.Tensor(source_imu)
        target = torch.Tensor(target)

        # add noise to initial target token for decoder (only required if real pos is supplied)
        # target[:, 0, :] += torch.randn(target.size(0), target.size(2))

        source_rss, source_rss_lengths = get_padded_wlan_tensor(source_rss)

        enc_hiddens_imu, _, dec_init_state = self.encode(source_imu, source_rss, source_rss_lengths)
        combined_outputs = self.decode(enc_hiddens_imu, dec_init_state, target)

        o_t = self.output_projection(combined_outputs)

        o_t = torch.permute(o_t, (1, 0, 2))

        score = self.loss_fnc(o_t[:, :-1, :], target[:, 1:, :])

        return score, o_t

    def encode(self, imu_source: torch.Tensor, rss_source: torch.Tensor, rss_source_lengths):
        # process imu with LSTM
        enc_hiddens_imu, (last_hidden_imu, last_cell_imu) = self.imu_encoder(imu_source)

        # process wlan rss data with separate LSTM
        rss_pad = nn.utils.rnn.pack_padded_sequence(rss_source, rss_source_lengths, enforce_sorted=True)
        enc_hiddens_rss, (last_hidden_rss, last_cell_rss) = self.rss_encoder(rss_pad)

        enc_hiddens_rss, _ = nn.utils.rnn.pad_packed_sequence(enc_hiddens_rss)

        enc_hiddens_rss = torch.permute(enc_hiddens_rss, (1, 0, 2))

        # reshape from (2, b, h) -> (b, 2h)
        last_hidden = torch.cat((last_hidden_rss[0], last_hidden_rss[1], last_hidden_imu[0], last_hidden_imu[1]), dim=1)
        init_decoder_hidden = self.h_projection(last_hidden)

        # reshape from (2, b, h) -> (b, 2h)
        last_cell = torch.cat((last_cell_rss[0], last_cell_rss[1], last_cell_imu[0], last_cell_imu[1]), dim=1)
        init_decoder_cell = self.c_projection(last_cell)

        dec_init_state = (init_decoder_hidden, init_decoder_cell)

        return enc_hiddens_rss, enc_hiddens_imu, dec_init_state

    def predict(self, source_imu, source_rss, y_target, T_out=10):
        with torch.no_grad():
            # convert to tensor
            source_imu = torch.Tensor(source_imu)
            y_target = torch.Tensor(y_target)
            source_rss, source_rss_lengths = get_padded_wlan_tensor(source_rss)

            enc_hiddens, _, dec_init_state = self.encode(source_imu, source_rss, source_rss_lengths)

            # Initialize previous combined output vector o_{t-1} as zero
            batch_size = enc_hiddens.size(0)
            o_prev = torch.zeros(batch_size, self.hidden_size)

            # initialize input of decoder with START_TOKEN (assume no prior position is known)
            Y_t = y_target[:, 0, :] #q+ torch.randn(y_target.size(0), y_target.size(2))

            dec_state = dec_init_state

            # Initialize a list we will use to collect the combined output o_t on each step
            predictions = []

            for idx in range(T_out):
                Ybar_t = torch.cat((Y_t, o_prev), dim=-1)
                (c_t, h_t), o_t = self.step(Ybar_t, dec_state)
                Y_t = self.output_projection(o_t)

                predictions.append(Y_t)
                o_prev = o_t

            predictions = torch.stack(predictions, dim=0)
            predictions = torch.permute(predictions, (1, 0, 2))

        return predictions


class MultiSourceCrossAttention(MultiSourceEncoder):
    """
    Model uses two separate LSTM encoder for IMU and WLAN data. LSTM decoder additionally has cross attention
    to IMU decoder hidden states.
    """
    def __init__(self, num_imu, num_ap, hidden_size, coord_dim=2, dropout_rate=0.2, loss_fnc=None):
        super(MultiSourceCrossAttention, self).__init__(num_imu, num_ap, hidden_size, coord_dim, dropout_rate, loss_fnc)

        # cross attention to IMU decoder
        self.att_projection = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size, bias=False)

        # combine output from attention over encoder hidden states and current decoder hidden state
        self.combined_output_projection = nn.Linear(in_features=3 * hidden_size, out_features=hidden_size, bias=False)

    def decode(self, enc_hiddens_imu, dec_init_state, target):

        # Initialize the decoder state (hidden and cell)
        dec_state = dec_init_state

        # Initialize previous combined output vector o_{t-1} as zero
        batch_size = enc_hiddens_imu.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size)

        # Initialize a list we will use to collect the combined output o_t on each step
        combined_outputs = []

        enc_hiddens_imu_proj = self.att_projection(enc_hiddens_imu)

        for Y_t in torch.split(target, split_size_or_sections=1, dim=1):
            Y_t = torch.squeeze(Y_t)
            Ybar_t = torch.cat((Y_t, o_prev), dim=-1)

            (c_t, h_t), o_t = self.step(Ybar_t, dec_state, enc_hiddens_imu, enc_hiddens_imu_proj)
            combined_outputs.append(o_t)
            o_prev = o_t

        combined_outputs = torch.stack(combined_outputs, dim=0)

        return combined_outputs

    def step(self, Ybar_t, dec_state, enc_hiddens_imu, enc_hiddens_imu_proj):

        dec_state = self.decoder(Ybar_t, dec_state)
        dec_hidden, dec_cell = dec_state

        e_t = torch.squeeze(torch.bmm(torch.unsqueeze(dec_hidden, dim=1),
                                      torch.permute(enc_hiddens_imu_proj, [0, 2, 1])), dim=1)

        alpha_t = nn.functional.softmax(e_t)

        a_t = torch.squeeze(torch.bmm(torch.unsqueeze(alpha_t, dim=1), enc_hiddens_imu), dim=1)

        u_t = torch.cat((a_t, dec_hidden), dim=1)

        v_t = self.combined_output_projection(u_t)

        O_t = self.dropout(torch.tanh(v_t))

        combined_output = O_t

        return dec_state, combined_output


def _weights_init(m):
    if isinstance(m, nn.LSTM):
        nn.init.xavier_normal_(m.weight_ih_l0)
        nn.init.xavier_normal_(m.weight_hh_l0)
        #nn.init.xavier_normal_(m.weight_ih_l0_reverse)
        #nn.init.xavier_normal_(m.weight_hh_l0_reverse)