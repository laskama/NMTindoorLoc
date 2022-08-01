from data_provider import get_joint_source_data, get_scan_data_of_devices, get_multi_source_data, scale_imu_data
from dataset import Seq2PointDataset, Seq2SeqDataset
from loss import custom_loss
from model import JointSeq2Seq, JointSeq2Point, StaticBaseline, SingleFPencoder, MultiSourceEncoder, MultiSourceCrossAttention, HybridSeq2Point
from model_pt import NMTindoorLocPT
from model_tf import NMTindoorLoc
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


def train_pt_model(hidden_size=256, num_epochs=1):
    num_imu_channels = 6

    training_loader = torch.utils.data.DataLoader(Seq2SeqDataset(devices=['OnePlus'], train=True, num_imu_channels=num_imu_channels, transform=torch.Tensor), batch_size=32, shuffle=True)

    # construct PyTorch model
    model = NMTindoorLocPT(hidden_size=hidden_size, num_imu=num_imu_channels, num_ap=np.shape(training_loader.dataset.rss)[-1])

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def train_one_epoch(epoch_index, tb_writer):

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

        return loss

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = num_epochs

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)

        # We don't need gradients on to do reporting
        model.train(False)

        print('LOSS train {}'.format(avg_loss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss},
                           epoch_number + 1)
        writer.flush()

        epoch_number += 1

    # sample random sequence of test indices
    test_ds = Seq2SeqDataset(devices=['OnePlus'], train=True, num_imu_channels=num_imu_channels)
    s_idx = np.random.choice(np.arange(len(test_ds.imu)), len(test_ds.imu), replace=False)

    # predict sequences with learned model (uses prior pos prediction as next input of decoder)
    pred_seq = model.predict([torch.Tensor(test_ds.imu[s_idx[:]]), torch.Tensor(test_ds.rss[s_idx[:]])])
    true_seq = test_ds.pos_target[s_idx, :, :]
    pred_seq = pred_seq.detach().numpy()

    print("Regular mean error: {}".format(np.mean(np.linalg.norm(true_seq - pred_seq, axis=-1))))

    visualize_predictions(true_seq, pred_seq)


def train_tf_model(fit_model=False, include_mag=False, bidirectional_encoder=False, hidden_size=256, num_epochs=1, model_name="s2s"):
    data, pos = get_joint_source_data(devices=['S20', 'Galaxy', 'OnePlus'],
                                      return_unique_fp_for_sequence=False,
                                      forward_fill_scans=True,
                                      seq_to_point=False,
                                      add_start_token=True,
                                      include_mag=include_mag)

    num_imu_channels = 9 if include_mag else 6

    # decouple input data
    imu = data[:, :, :num_imu_channels]
    rss = data[:, :, num_imu_channels:]

    # imu and pos data dimensions
    _, T_pos, _ = np.shape(pos)

    # scale imu data
    imu = scale_imu_data(imu)

    # scale rss data
    max_rss = -40.0
    min_rss = -110.0
    rss = (rss - min_rss) / (max_rss - min_rss)

    # split into train/test data
    imu_train, imu_test, rss_train, rss_test, pos_train, pos_test = train_test_split(imu, rss, pos, test_size=0.2, random_state=1, shuffle=True)

    # Prepare position data for teacher forced sequence prediction (target will be offset by 1)
    dec_init = pos_train[:, :-1, :]
    dec_target = pos_train[:, 1:, :]

    # construct tensorflow model and compile
    model = NMTindoorLoc(hidden_size=hidden_size, cross_attention=False, bidirectional_encoder=bidirectional_encoder)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError()
    )

    if fit_model:
        history = model.fit(
            (imu_train, rss_train, dec_init),
            dec_target,
            batch_size=32,
            epochs=num_epochs,
        )

        model.save_weights(model_name + ".hdf5")

    model.load_weights(model_name + ".hdf5")  # = tf.keras.models.load_model(model_name)

    # sample random sequence of test indices
    s_idx = np.random.choice(np.arange(len(imu_test)), len(imu_test), replace=False)

    # predict sequences with learned model (uses prior pos prediction as next input of decoder)
    pred_seq = model.predict([imu_test[s_idx[:]], rss_test[s_idx[:]]])
    true_seq = pos_test[s_idx, 1:, :]

    print("Regular mean error: {}".format(np.mean(np.linalg.norm(true_seq - pred_seq, axis=-1))))

    visualize_predictions(pos_test[s_idx, 1:, :], pred_seq)


if __name__ == '__main__':
    train_pt_model(num_epochs=10)
    train_tf_model(fit_model=True, bidirectional_encoder=True, model_name="att_v2", num_epochs=10)
