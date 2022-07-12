from data_provider import get_joint_source_data, get_scan_data_of_devices, get_multi_source_data
from dataset import Seq2PointDataset
from loss import custom_loss
from model import JointSeq2Seq, JointSeq2Point, StaticBaseline, SingleFPencoder, MultiSourceEncoder, MultiSourceCrossAttention, HybridSeq2Point
from model_tf import get_tf_model, get_tf_seq2seq_train_model, decode_sequence, get_tf_seq2seq_encoder_model, \
    get_tf_seq2seq_decoder_model
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.model_selection import train_test_split
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


from utils import batch_iter, batch_iter_no_tensor
from visualization import visualize_predictions


def train_hybrid_seq2seq_tf(fit_model=False, include_mag=False, bidirectional_encoder=False, hidden_size=256):
    data, pos = get_joint_source_data(devices=['S20', 'Galaxy', 'OnePlus'],
                                      return_unique_fp_for_sequence=False,
                                      forward_fill_scans=True,
                                      seq_to_point=False,
                                      add_start_token=True,
                                      include_mag=include_mag)

    num_imu_channels = 9 if include_mag else 6

    imu = data[:, :, :num_imu_channels]
    rss = data[:, :, num_imu_channels:]

    # imu and pos data dimensions
    N, T, M = np.shape(imu)
    _, T_pos, _ = np.shape(pos)

    # scale imu data
    imu = np.reshape(imu, [-1, M])
    data_scaler = StandardScaler()
    imu = data_scaler.fit_transform(imu)
    imu = np.reshape(imu, [-1, T, M])

    # scale rss data
    max_rss = -40.0
    min_rss = -110.0
    rss = (rss - min_rss) / (max_rss - min_rss)

    imu_train, imu_test, rss_train, rss_test, pos_train, pos_test = train_test_split(imu, rss, pos, test_size=0.2, random_state=1, shuffle=True)

    dec_init = pos_train[:, :-1, :]
    dec_target = pos_train[:, 1:, :]

    model = get_tf_seq2seq_train_model(num_ap=np.shape(rss)[2], num_imu=num_imu_channels, hidden_size=hidden_size, seq_length=T, out_seq_length=T_pos-1, bidirectional_encoder=bidirectional_encoder)

    # define loss function
    mse = tf.keras.losses.MeanSquaredError()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=mse
    )

    if fit_model:
        history = model.fit(
            (imu_train, rss_train, dec_init),
            dec_target,
            batch_size=32,
            epochs=20,
        )

        model.save("s2s")

    model = tf.keras.models.load_model("s2s")

    encoder_model = get_tf_seq2seq_encoder_model(model, bidirectional_encoder=bidirectional_encoder)
    decoder_model = get_tf_seq2seq_decoder_model(model, hidden_size=2 * hidden_size if bidirectional_encoder else hidden_size)

    # sample test indices
    s_idx = np.random.choice(np.arange(len(imu_test)), len(imu_test), replace=False)

    # compute predictions in teacher-forced setting (not-valid, only for )
    traj_a = model.predict((imu_test[s_idx], rss_test[s_idx], pos_test[s_idx, :-1, :]))
    true = pos_test[s_idx, 1:, :]
    print("Teacher-forced mean error: {}".format(np.mean(np.linalg.norm(true - traj_a, axis=-1))))

    traj_b = decode_sequence((imu_test[s_idx], rss_test[s_idx]), encoder_model, decoder_model, init_pos=None) #pos_test[s_idx, 0, :][:, None, :])
    print("Regular mean error: {}".format(np.mean(np.linalg.norm(true - traj_b, axis=-1))))

    visualize_predictions(pos_test[s_idx, 1:, :], traj_b, seq_to_point=False)


def train_hybrid_seq2point_dl():

    training_set = Seq2PointDataset(train=True, transform=torch.Tensor)
    testing_set = Seq2PointDataset(train=False, transform=torch.Tensor)

    training_loader = torch.utils.data.DataLoader(training_set, batch_size=32, shuffle=True, num_workers=2)
    testing_loader = torch.utils.data.DataLoader(testing_set, batch_size=32, shuffle=False, num_workers=2)

    # Report split sizes
    print('Training set has {} instances'.format(len(training_set)))
    print('Validation set has {} instances'.format(len(testing_set)))

    model = HybridSeq2Point(num_imu=6, num_ap=np.shape(training_set.rss)[2], hidden_size=256, coord_dim=2)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    def train_one_epoch(epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            (imu, rss), labels = data

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(imu, rss)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 10 == 0:
                last_loss = running_loss / 10 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 5

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        for i, vdata in enumerate(testing_loader):
            (vinputs_imu, vinputs_rss), vlabels = vdata
            voutputs = model(vinputs_imu, vinputs_rss)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

def train_hybrid_seq2point_tf():
    data, pos = get_joint_source_data(devices=['S20', 'Galaxy', 'OnePlus'], return_unique_fp_for_sequence=False, forward_fill_scans=True,
                                      seq_to_point=True)

    imu = data[:, :, :6]
    rss = data[:, :, 6:]

    # imu and pos data dimensions
    N, T, M = np.shape(imu)

    # scale imu data
    imu = np.reshape(imu, [-1, M])
    data_scaler = StandardScaler()
    imu = data_scaler.fit_transform(imu)
    imu = np.reshape(imu, [-1, T, M])

    # scale rss data
    max_rss = -40.0
    min_rss = -110.0
    rss = (rss - min_rss) / (max_rss - min_rss)

    imu_train, imu_test, rss_train, rss_test, pos_train, pos_test = train_test_split(imu, rss, pos, test_size=0.2, random_state=1)

    model = get_tf_model(num_ap=np.shape(rss)[2], num_imu=6, hidden_size=256, seq_length=T)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError()
    )

    history = model.fit(
        (imu_train, rss_train),
        pos_train,
        batch_size=32,
        epochs=10,
    )

    results = model.evaluate((imu_test, rss_test), pos_test, batch_size=32)
    pred = model.predict((imu_test, rss_test), batch_size=32)
    print("test loss, test acc:", results)

    visualize_predictions(pos_test, pred, seq_to_point=True)


def train_hybrid_seq2point(save_path, fit_model=True):
    data, pos = get_joint_source_data(devices=['S20'], return_unique_fp_for_sequence=False, forward_fill_scans=True, seq_to_point=True)

    imu = data[:, :, :6]
    rss = data[:, :, 6:]

    # imu and pos data dimensions
    N, T, M = np.shape(imu)

    # scale imu data
    imu = np.reshape(imu, [-1, M])
    data_scaler = StandardScaler()
    imu = data_scaler.fit_transform(imu)
    imu = np.reshape(imu, [-1, T, M])

    # scale rss data
    max_rss = -40.0
    min_rss = -110.0
    rss = (rss - min_rss) / (max_rss - min_rss)

    train_data = list(zip(torch.Tensor(imu), torch.Tensor(rss), torch.Tensor(pos)))

    model = HybridSeq2Point(num_imu=6, num_ap=np.shape(rss)[2], hidden_size=256, coord_dim=2)

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

            print("epoch {} finished".format(epoch))

        torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path))
    model.eval()
    with torch.no_grad():
        for src_imu, src_rss, tgt in batch_iter(train_data, batch_size=32, shuffle=True, static_rss=True):

            # uses predictions as input for decoder
            # pred = model.predict(src_imu, src_rss, tgt)
            _, pred = model(src_imu, src_rss, tgt)

            visualize_predictions(tgt[:, 1:, :], pred[:, :-1, :], draw_individually=True)


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

    model = MultiSourceEncoder(num_imu=9, num_ap=np.shape(rss[0])[1], hidden_size=256, loss_fnc=nn.MSELoss(reduction='mean'))

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
    train_hybrid_seq2seq_tf(fit_model=True, bidirectional_encoder=True)
    # train_hybrid_seq2point_dl()
    # train_hybrid_seq2point_tf()
    # train_hybrid_seq2point(save_path="hybridseq2point.pt", fit_model=True)
    # train_hybrid_lstm(save_path="crossAttention_hybrid_cl.pt", fit_model=True)
    # train_point()
    # train_baseline()
    # train_complex_seq(save_path="complexFus_v4.pt", fit_model=True, load_previous_weights=False)
    # train_seq(save_path="seqFus_start_v2.pt.pt", fit_model=True)
