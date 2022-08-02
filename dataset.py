from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from data_provider import get_joint_source_data, scale_imu_data
import numpy as np


class Seq2SeqDataset(Dataset):
    """
    Wrapper class for PyTorch around seq2seq dataset.
    Currently, performs batch dataset preprocessing and loading within init phase.
    Could be optimized by on-demand preprocessing within __getitem__. However,
    we want to reuse get_joint_source_data to achieve comparability to tf models.
    """
    def __init__(self, devices=None, num_imu_channels=6, train=True, transform=None):
        self.transform = transform

        data, pos = get_joint_source_data(devices=devices,
                                          return_unique_fp_for_sequence=False,
                                          forward_fill_scans=True,
                                          seq_to_point=False,
                                          add_start_token=True,
                                          include_mag=False)

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
        imu_train, imu_test, rss_train, rss_test, pos_train, pos_test = train_test_split(imu, rss, pos, test_size=0.2,
                                                                                         random_state=1, shuffle=True)
        if train:
            self.imu, self.rss, self.pos_init, self.pos_target = imu_train, rss_train, pos_train[:, :-1, :], pos_train[:, 1:, :]
        else:
            self.imu, self.rss, self.pos_init, self.pos_target = imu_test, rss_test, pos_test[:, :-1, :], pos_test[:, 1:, :]

    def __len__(self):
        return len(self.imu)

    def __getitem__(self, idx):
        if self.transform:
            return (self.transform(self.imu[idx]), self.transform(self.rss[idx]), self.transform(self.pos_init[idx])), self.transform(self.pos_target[idx])
        else:
            return (self.imu[idx], self.rss[idx], self.pos_init[idx]), self.pos_target[idx]
