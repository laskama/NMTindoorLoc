from data_provider import get_joint_source_data, get_scan_data_of_devices, get_multi_source_data
from loss import custom_loss
from model import JointSeq2Seq, JointSeq2Point, StaticBaseline, SingleFPencoder, MultiSourceEncoder, MultiSourceCrossAttention
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


from utils import batch_iter, batch_iter_no_tensor
from visualization import visualize_predictions


def train_hybrid_lstm(save_path, fit_model=True):
    imu, rss, pos = get_multi_source_data(devices=['S20', 'OnePlus', 'Galaxy'])

    # ignore relative position of wlan scan within window for now
    rss = [r[0] for r in rss]

    # imu and pos data dimensions
    N, T, M = np.shape(imu)
    _, T_pos, M_pos = np.shape(pos)

    # scale imu data
    imu = np.reshape(imu, [-1, M])
    data_scaler = StandardScaler()
    imu = data_scaler.fit_transform(imu)
    imu = np.reshape(imu, [-1, T, M])

    # scale rss data
    max_rss = -40.0
    min_rss = -110.0
    rss = [(r - min_rss) / (max_rss - min_rss) for r in rss]

    train_data = list(zip(imu, rss, pos))

    model = MultiSourceEncoder(num_imu=9, num_ap=np.shape(rss[0])[1], hidden_size=256, loss_fnc=custom_loss)

    model.train()

    device = "cpu"

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_iter = epoch = 0

    if fit_model:
        while epoch < 10:
            epoch += 1

            for src_imu, src_rss, tgt in batch_iter_no_tensor(train_data, batch_size=32, shuffle=True):
                train_iter += 1

                optimizer.zero_grad()

                loss, _ = model(src_imu, src_rss, tgt)

                loss.backward()

                optimizer.step()

                if train_iter % 10 == 0:
                    print(loss)

            print("epoch {} finished".format(epoch))

        torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path))
    model.eval()
    with torch.no_grad():
        for src_imu, src_rss, tgt in batch_iter_no_tensor(train_data, batch_size=32, shuffle=True):

            tgt = torch.Tensor(tgt)

            # uses predictions as input for decoder
            pred = model.predict(src_imu, src_rss, tgt)
            # _, pred = model(src_imu, src_rss, tgt)

            visualize_predictions(tgt[:, 1:, :], pred[:, :-1, :], draw_individually=True)


def train_complex_seq(save_path, fit_model=True, load_previous_weights=False):
    num_imu_channels = 9

    data, pos, data_rss, time_vec = get_joint_source_data(devices=['S20', 'OnePlus', 'Galaxy'],
                                                          return_unique_fp_for_sequence=True)

    # only use imu of data tensor
    data_imu = data[:, :, :num_imu_channels]

    # imu and pos data dimensions
    N, T, M = np.shape(data_imu)
    _, T_pos, M_pos = np.shape(pos)

    # scale imu data
    data_imu = np.reshape(data_imu, [-1, M])
    data_scaler = StandardScaler()
    data_imu = data_scaler.fit_transform(data_imu)
    data_imu = np.reshape(data_imu, [-1, T, M])

    # scale rss data
    max_rss = -40.0
    min_rss = -110.0
    data_rss = (data_rss - min_rss) / (max_rss - min_rss)

    # concat rss data with time vec
    # data_rss = np.concatenate((data_rss, time_vec), axis=1)

    # convert to tensors
    data_imu = torch.Tensor(data_imu)
    data_rss = torch.Tensor(data_rss)

    pos = torch.Tensor(pos)

    train_data = list(zip(data_imu, data_rss, pos))

    model = SingleFPencoder(num_imu=num_imu_channels, num_ap=data_rss.size(1), hidden_size=256)

    if load_previous_weights:
        model.load_state_dict(torch.load(save_path))

    model.train()

    device = "cpu"

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_iter = epoch = 0

    if fit_model:
        while epoch < 10:
            epoch += 1

            for src_imu, src_rss, tgt in batch_iter(train_data, batch_size=32, shuffle=True, static_rss=True):
                train_iter += 1

                optimizer.zero_grad()

                loss, _ = model(src_imu, src_rss, tgt)

                loss.backward()

                optimizer.step()

                if train_iter % 10 == 0:
                    print(loss)

                if train_iter % 100 == 0:
                    break

            print("epoch {} finished".format(epoch))

        torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path))
    model.eval()
    with torch.no_grad():
        for src_imu, src_rss, tgt in batch_iter(train_data, batch_size=32, shuffle=True, static_rss=True):

            # _, pred = model(src_imu, src_rss, tgt)
            pred = model.predict(src_imu, src_rss, tgt)

            visualize_predictions(tgt[:, 1:, :], pred[:, :-1, :], draw_individually=True)


def train_seq(save_path, fit_model=True):
    data, pos = get_joint_source_data(devices=['S20', 'OnePlus', 'Galaxy'])

    # data dimensions
    N, T, M = np.shape(data)
    _, T_pos, M_pos = np.shape(pos)

    # scale data
    data = np.reshape(data, [-1, M])
    data_scaler = StandardScaler()
    data_scaled = data_scaler.fit_transform(data)
    data = np.reshape(data_scaled, [-1, T, M])

    # convert to tensors
    data = torch.Tensor(data)
    pos = torch.Tensor(pos)

    train_data = list(zip(data, pos))

    model = JointSeq2Seq(num_features=np.shape(data)[2], hidden_size=256)

    model.train()

    device = "cpu"

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_iter = epoch = 0

    if fit_model:
        while epoch < 10:
            epoch += 1

            for src, tgt in batch_iter(train_data, batch_size=32, shuffle=True):
                train_iter += 1

                optimizer.zero_grad()

                loss, _ = model(src, tgt)

                loss.backward()

                optimizer.step()

                if train_iter % 10 == 0:
                    print(loss)

                if train_iter % 100 == 0:
                    break

            print("epoch {} finished".format(epoch))

        torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path))
    model.eval()
    with torch.no_grad():
        for src, tgt in batch_iter(train_data, batch_size=32, shuffle=True):

            # _, pred = model(src, tgt)
            pred = model.predict(src, tgt)

            visualize_predictions(tgt[:, 1:, :], pred[:, :-1, :], draw_individually=True)


def train_baseline():
    data, pos = get_scan_data_of_devices(['S20'])

    max_rss = -40.0
    min_rss = -110.0

    data = (data - min_rss) / (max_rss - min_rss)

    # convert to tensors
    data = torch.Tensor(data)
    pos = torch.Tensor(pos)

    train_data = list(zip(data, pos))

    model = StaticBaseline(num_features=np.shape(data)[1], hidden_size=256)

    model.train()

    device = "cpu"

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_iter = epoch = 0

    while epoch < 100:
        epoch += 1

        for src, tgt in batch_iter(train_data, batch_size=32, shuffle=True):
            train_iter += 1

            optimizer.zero_grad()

            loss, _ = model(src, tgt)

            loss.backward()

            optimizer.step()

            if train_iter % 10 == 0:
                print(loss)

        print("epoch {} finished".format(epoch))

    with torch.no_grad():
        for src, tgt in batch_iter(train_data, batch_size=32, shuffle=True):
            train_iter += 1

            optimizer.zero_grad()

            _, pred = model(src, tgt)

            visualize_predictions(tgt, pred, seq_to_point=True)


def train_point():
    data, pos = get_joint_source_data(devices=['S20'], seq_to_point=True)

    # data dimensions
    N, T, M = np.shape(data)
    # _, M_pos = np.shape(pos)

    max_rss = -40.0
    min_rss = -110.0

    data = (data - min_rss) / (max_rss - min_rss)

    # scale data
    data = np.reshape(data, [-1, M])
    data_scaler = StandardScaler()
    data_scaled = data_scaler.fit_transform(data)
    data = np.reshape(data_scaled, [-1, T, M])

    # convert to tensors
    data = torch.Tensor(data)
    pos = torch.Tensor(pos)

    train_data = list(zip(data, pos))

    model = JointSeq2Point(num_features=np.shape(data)[2], hidden_size=256)

    model.train()

    device = "cpu"

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_iter = epoch = 0

    while epoch < 10:
        epoch += 1

        for src, tgt in batch_iter(train_data, batch_size=32, shuffle=True):
            train_iter += 1

            optimizer.zero_grad()

            loss, _ = model(src, tgt)

            loss.backward()

            optimizer.step()

            if train_iter % 10 == 0:
                print(loss)

        print("epoch {} finished".format(epoch))
        print(loss)

    with torch.no_grad():
        for src, tgt in batch_iter(train_data, batch_size=32, shuffle=True):

            _, pred = model(src, tgt)

            visualize_predictions(tgt, pred, seq_to_point=True)


if __name__ == '__main__':
    train_hybrid_lstm(save_path="crossAttention_hybrid_cl.pt", fit_model=True)
    # train_point()
    # train_baseline()
    # train_complex_seq(save_path="complexFus_v4.pt", fit_model=True, load_previous_weights=False)
    # train_seq(save_path="seqFus_start_v2.pt.pt", fit_model=True)
