from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from data_provider import get_joint_source_data, scale_imu_data
import numpy as np


class Seq2SeqDataset(Dataset):

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


class Seq2PointDataset(Dataset):

    def __init__(self, devices=None, num_imu_channels=6, train=True, transform=None):
        data, pos = get_joint_source_data(devices, return_unique_fp_for_sequence=False, seq_to_point=True)
        imu = data[:, :, :num_imu_channels]
        rss = data[:, :, num_imu_channels:]
        self.transform = transform

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

        imu_train, imu_test, rss_train, rss_test, pos_train, pos_test = train_test_split(imu, rss, pos, test_size=0.2,
                                                                                         random_state=1)

        if train:
            self.imu, self.rss, self.pos = imu_train, rss_train, pos_train
        else:
            self.imu, self.rss, self.pos = imu_test, rss_test, pos_test

    def __len__(self):
        return len(self.imu)

    def __getitem__(self, idx):
        if self.transform:
            return (self.transform(self.imu[idx]), self.transform(self.rss[idx])), self.transform(self.pos[idx])
        else:
            return (self.imu[idx], self.rss[idx]), self.pos[idx]
