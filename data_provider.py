from typing import List

import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import os

from sklearn.preprocessing import StandardScaler

NANO_PER_MILLI = 1000000
START_TOKEN = np.array([-10000, 10000])


def scale_imu_data(imu):
    N, T, M = np.shape(imu)

    imu = np.reshape(imu, [-1, M])
    data_scaler = StandardScaler()
    imu = data_scaler.fit_transform(imu)
    imu = np.reshape(imu, [-1, T, M])

    return imu


def get_unique_ap_vec(data_folders: List[str]) -> np.array:
    """
    Computes mac addresses of all Access Points (APs) that have been observed in the dataset
    :param data_folders: The data folder which should be scanned for collected trajectories
    :return: Sorted array of all mac addresses
    """
    macs = np.unique(np.concatenate(
        [pd.read_csv("giaIndoorLoc/{}/wifi_annotated.csv".format(df))['mac'].unique() for df in data_folders]))
    macs.sort()

    return macs


def _interpolate_imu_tensor(int_points: np.array, data_df: pd.DataFrame) -> np.array:
    """
    Interpolates given 3-dimensional component (ACC, GYRO, or MAG) of inertial measurement unit (IMU).
    Measurement frequency is not the same for each component, however, the network requires synced data.
    :param int_points: The timestamps at which the interpolated data should be obtained
    :param data_df: The dataframe of IMU measurements that holds the 3-dimensional measurements with their
                    corresponding timestamps
    :return: 3-dimensional numpy array of shape (len(int_points), 3)
    """
    tensor = np.zeros((len(int_points), 3))

    for c_idx in range(3):
        tensor[:, c_idx] = np.interp(int_points, data_df['time'].to_numpy(), data_df.iloc[:, c_idx + 2])
        tensor[:, c_idx] = np.interp(int_points, data_df['time'].to_numpy(), data_df.iloc[:, c_idx + 2])
        tensor[:, c_idx] = np.interp(int_points, data_df['time'].to_numpy(), data_df.iloc[:, c_idx + 2])

    return tensor


def get_scan_wlan_data_of_folder(folder="floor_0/S20/2021-12-20T13:46:39", unique_mac_addr: np.array=None, return_time=False):
    """
    Obtains scan-based WLAN fingerprints. Each scanned AP-RSS value is assigned to a unique scan id (a scan might take
    several seconds to complete). Use the scan id for constructing the fingerprints and assign unique position by
    averaging across all position of a scan.
    :param folder: Path to data folder where annotated WiFi data can be found
    :param unique_mac_addr: Array of unique mac addresses of entire dataset for constructing unique assignment of
                            mac address to position within fingerprint vector
    :param return_time: Whether to additionally return the timestamps of each fingerprint
    :return: tuple of (fingerprints of shape (num_scans, num_ap), positions (num_scans, 2) and [time (num_scans,)])
    """
    df = pd.read_csv("giaIndoorLoc/{}/wifi_annotated.csv".format(folder)).sort_values(by='time')

    # exclude scans without position
    null_pos_scans = df[["x_coord", "y_coord"]].isnull().any(axis=1)
    df = df[~null_pos_scans]

    scan_ids = df["id"].unique()

    data = np.full((len(scan_ids), len(unique_mac_addr)), -110.0)
    time = np.zeros(len(scan_ids))
    labels = np.zeros((len(scan_ids), 2))

    for idx, id in enumerate(scan_ids):
        sub = df[df["id"] == id]
        positions = sub[["x_coord", "y_coord"]].to_numpy()
        t = np.mean(sub["time"].to_numpy())
        pos_avg = np.mean(positions, axis=0)
        labels[idx, :] = pos_avg
        time[idx] = t

        for _, s in sub.iterrows():
            mac_idx = np.where(unique_mac_addr == s["mac"])[0]
            data[idx, mac_idx] = s["rss"]

    if return_time:
        return data, labels, time
    else:
        return data, labels


def get_scan_data_of_devices(devices: List[str]=None):
    """
    Obtains scan-based fingerprints for all selected devices by inspecting all folders in the base data directory.
    :param devices: List of devices that should be incorporated into the dataset
    :return: tuple of (fingerprints of shape (num_fingerprints, num_aps) and positions (num_fingerprints, 2)
    """
    data_dir = 'giaIndoorLoc'
    path = "floor_1/{}/"

    if devices is None:
        devices = ['Galaxy', 'OnePlus', 'S20']

    data_folders = [path.format(dev) + '/' + d for dev in devices for d in os.listdir(data_dir + '/' + path.format(dev))
                    if not d.startswith('.')]

    unique_mac = get_unique_ap_vec(data_folders)

    # obtain data for each trajectory
    res = [get_scan_wlan_data_of_folder(df, unique_mac) for df in data_folders]

    # join data from trajectoreis
    data = np.concatenate([r[0] for r in res])
    pos = np.concatenate([r[1] for r in res])

    return data, pos


def get_IMU_and_WLAN_of_folder(folder="floor_1/S20/2021-12-20T13:19:42"):
    """
    Read the IMU and WLAN data from a given folder and returns them as Pandas dataframes.
    :param folder: Path to data
    :return: tuple of (acc, gyro, mag, wlan, start_time) where start_time is the first entry in the trajectory,
                        for which a ground truth position is known)
    """
    # read data
    imu = pd.read_csv("giaIndoorLoc/{}/sensors_annotated.csv".format(folder)).sort_values(by='time')
    wlan = pd.read_csv("giaIndoorLoc/{}/wifi_annotated.csv".format(folder)).sort_values(by='time')

    # only retain certain IMU readings
    imu = imu[imu['type'].isin([" ACC", " GYRO", " MAG"])]

    # find timestamp where first position is obtained via VSLAM2tag and cut data
    start_time, end_time = imu.iloc[np.where(~imu['x_coord'].isna().to_numpy())[0][[0, -1]], 0]

    imu = imu[imu['time'].between(start_time, end_time)]
    wlan = wlan[wlan['time'].between(start_time, end_time)]

    # subtract start time from time
    imu['time'] -= start_time
    wlan['time'] -= start_time

    # scale to milli sec
    imu['time'] /= NANO_PER_MILLI
    wlan['time'] /= NANO_PER_MILLI

    # get different channels
    acc = imu[imu['type'] == " ACC"]
    gyro = imu[imu['type'] == " GYRO"]
    mag = imu[imu['type'] == " MAG"]

    return acc, gyro, mag, wlan, start_time


def get_synced_imu_tensors(acc, gyro, mag):
    """
    Obtain tensors for all channels of IMU (acc, gyro, mag) by syncing in time domain and
    interpolating if channels are not perfectly synced.
    :param acc: ACC channel of IMU
    :param gyro: GYRO channel of IMU
    :param mag: MAG channel of IMU
    :return: tuple of (acc_tensor, gyro_tensor, mag_tensor, of shape (num_max_sensor_measurements, 3) and position
                        tensor of shape (num_max_sensor_measurements, 3))
    """
    if not (len(acc) == len(gyro) == len(mag)):

        # perform interpolation (use most frequent IMU sensor as baseline for interpolation points [x])
        max_sensor_idx = np.argmax(np.array([len(acc), len(gyro), len(mag)]))
        sensors = [acc, gyro, mag]
        pos_tensor = sensors[max_sensor_idx][["x_coord", "y_coord"]].to_numpy()

        x = sensors[max_sensor_idx]['time'].to_numpy()

        acc_tensor = _interpolate_imu_tensor(x, acc)
        gyro_tensor = _interpolate_imu_tensor(x, gyro)
        mag_tensor = _interpolate_imu_tensor(x, mag)

    else:
        acc_tensor = acc.iloc[:, 2:5].to_numpy()
        gyro_tensor = gyro.iloc[:, 2:5].to_numpy()
        mag_tensor = mag.iloc[:, 2:5].to_numpy()
        pos_tensor = acc[["x_coord", "y_coord"]].to_numpy()

    return acc_tensor, gyro_tensor, mag_tensor, pos_tensor


def construct_wlan_tensor(wlan_df, len_IMU_tensor, missing_ap_val=-110.0, imu_readings_per_sec=20, unique_mac_addr=None):
    """
    Construct WLAN tensor that is aligned in time-domain with IMU tensor by utilizing IMU sampling frequency and WLAN
    scan timestamp for alignment.
    :param wlan_df: WLAN scan dataframe that holds
    :param len_IMU_tensor: The length of the IMU tensor
    :param missing_ap_val: RSS placeholder value for APs that have not been detected
    :param imu_readings_per_sec: determined by sampling frequency of IMU
    :param unique_mac_addr: Unique mac addresses of entire dataset (required for constructing fingerprints)
    :return: tuple of (wlan_tensor with shape (len_IMU_tensor, len_unique_mac_addr), unique_mac_addr)
    """
    if unique_mac_addr is None:
        unique_mac_addr = wlan_df['mac'].unique()
        unique_mac_addr.sort()

    num_mac = len(unique_mac_addr)

    wlan_tensor = np.full((len_IMU_tensor, num_mac), missing_ap_val)

    for mac_id, mac in enumerate(unique_mac_addr):
        sub = wlan_df[wlan_df['mac'] == mac]
        sub['time'] /= imu_readings_per_sec
        sub['time'] = sub['time'].round(decimals=0).astype(int)
        wlan_tensor[sub['time'], mac_id] = sub['rss']

    return wlan_tensor, unique_mac_addr


def partition_data_via_sliding_window(data_tensor, pos_tensor, imu_seq_length, step_size, output_step_size):
    """
    partition entire trajectory into sequences of seq_length via sliding window (only maintaining the
    step-size-th window). Sequence length is +1 here, since each first point of window serves as
    initialization (start token), which is cutoff later
    :param data_tensor: Source data tensor
    :param pos_tensor: Target position tensor
    :param imu_seq_length: Sequence length of IMU data (T)
    :param step_size: Step-size for shifting sliding window
    :param output_step_size: Every output_step_size-th (of T) value is taken to form position tensor
    :return: tuple of (data with shape (num_seq, T, num_data_channels), pos (num_seq, O+1, 2))
    """
    data_tensor_windows = np.transpose(sliding_window_view(
        data_tensor, imu_seq_length + 1, axis=0)[::step_size], axes=[0, 2, 1])
    pos_tensor_windows = np.transpose(sliding_window_view(
        pos_tensor, imu_seq_length + 1, axis=0)[::step_size], axes=[0, 2, 1])[:, ::output_step_size, :]

    return data_tensor_windows, pos_tensor_windows


def _get_joint_source_data(folder="floor_1/S20/2021-12-20T13:19:42", unique_mac_addr=None, seq_to_point=False,
                           add_start_token=False, return_unique_fp_for_sequence=False,
                           step_size=10, seq_length=200, output_step_size=20):
    """
    Constructs a joint-source data tensor of IMU and WLAN data. IMU data determines the sequence length (T) of the
    data tensor. WLAN data is aligned by expanding it in the time-domain.
    :param folder: The path to the data folder
    :param unique_mac_addr: Unique mac addresses of entire dataset (required for constructing fingerprints)
    :param seq_to_point: Whether to reduce target position tensor to only include the last position of a sequence
    :param add_start_token: Whether to replace the initial position of a sequence with the fixed start token
    :param return_unique_fp_for_sequence: Whether to additionally return a unique fingerprint for each sequence
    :param step_size: Number of steps that the window is forwarded to obtain the next sequence
    :param seq_length: IMU sequence length (T)
    :param output_step_size: Every output_step_size-th (of T) value is taken to form position tensor
    :return: Tuple of (data_tensor of shape (num_sequences, sequence_len (T), 9+num_AP),
                        position tensor (num_sequences, output_seq_len (O), 2),
                        [rss_tensor (num_sequences, num_AP), time_tensor (num_sequences, )])
    """
    imu_readings_per_sec = 20
    missing_ap_val = -110.0

    acc, gyro, mag, wlan, _ = get_IMU_and_WLAN_of_folder(folder)

    acc_tensor, gyro_tensor, mag_tensor, pos_tensor = get_synced_imu_tensors(acc, gyro, mag)

    # visualize_data_time_distribution(acc, gyro, mag, wlan)

    wlan_tensor, unique_mac_addr = construct_wlan_tensor(wlan, len(acc_tensor), missing_ap_val, imu_readings_per_sec, unique_mac_addr)

    # merge into large data tensor
    data_tensor = np.concatenate((
        acc_tensor,
        gyro_tensor,
        # mag_tensor,
        wlan_tensor),
        axis=1)

    # check for nan entries in pos tensor
    not_nan_pos_mask = np.where(~np.any(np.isnan(pos_tensor), axis=1))[0]
    data_tensor = data_tensor[not_nan_pos_mask]
    pos_tensor = pos_tensor[not_nan_pos_mask]

    data_tensor_windows, pos_tensor_windows = partition_data_via_sliding_window(
        data_tensor, pos_tensor, seq_length, step_size, output_step_size)

    # add start token to pos_tensor
    if add_start_token:
        pos_tensor_windows = replace_init_pos_by_start_token(pos_tensor_windows)

    # cutoff initial data from sequence (since only its position is used as initial entry (start token)
    data_tensor_windows = data_tensor_windows[:, 1:, :]

    if seq_to_point:
        pos_tensor_windows = pos_tensor_windows[:, -1, :]

    num_mac = len(unique_mac_addr)
    rss_tensor = np.full([len(data_tensor_windows), num_mac], -110.0)
    time_tensor = np.zeros([len(data_tensor_windows), num_mac])

    for seq_idx, seq in enumerate(data_tensor_windows):
        rss = np.full(num_mac, -110.0)
        time = np.zeros(num_mac)
        wlan_seq = seq[:, 6:]

        # generate time tensor that hold relative times with same shape as wlan seq tensor
        time_t = np.concatenate([np.arange(0, 1, 0.005).reshape(-1, 1)] * num_mac, axis=1)

        # Identify those parts in wlan seq tensor with RSS readings
        # detected_aps[1] will have duplicates (however axis 0 is sorted => later values will overwrite old entries)
        detected_aps = np.where(wlan_seq != -110.0)

        rss[detected_aps[1]] = wlan_seq[detected_aps]
        time[detected_aps[1]] = time_t[detected_aps]

        rss_tensor[seq_idx, :] = rss
        time_tensor[seq_idx, :] = time

    # visualize_window_trajectories(pos_tensor_windows)

    if return_unique_fp_for_sequence:
        return data_tensor_windows, pos_tensor_windows, rss_tensor, time_tensor
    else:
        return data_tensor_windows, pos_tensor_windows


def _get_multi_source_data(folder="floor_0/S20/2021-12-20T13:46:39", unique_mac_addr=None, add_pos_start_token=True,
                           step_size=10, seq_length=200, output_step_size=20):
    """
    Constructs multi-source data (non-synced time-domain of IMU and WLAN measurements).
    :param folder: Path to data folder
    :param unique_mac_addr: Unique mac addresses of entire dataset (required for constructing fingerprints)
    :param add_pos_start_token: Whether to replace the initial position of a sequence with the fixed start token
    :param step_size: Number of steps that the window is forwarded to obtain the next sequence
    :param seq_length: IMU sequence length (T)
    :param output_step_size: Every output_step_size-th (of T) value is taken to form position tensor
    :return: Tuple of (imu_tensor of shape (num_sequences, sequence_len (T), 9), WLAN scans (list that contains tuple
             of scans and their timestamps for each sequence), position tensor (num_sequences, output_seq_len (O), 2)
    """
    ms_per_imu_scan = 20
    missing_ap_val = -110.0

    acc, gyro, mag, wlan, start_time = get_IMU_and_WLAN_of_folder(folder)

    acc_tensor, gyro_tensor, mag_tensor, pos_tensor = get_synced_imu_tensors(acc, gyro, mag)

    # visualize_data_time_distribution(acc, gyro, mag, wlan)

    if unique_mac_addr is None:
        unique_mac_addr = wlan['mac'].unique()
        unique_mac_addr.sort()

    num_mac = len(unique_mac_addr)

    # obtain scan based WLAN fingerprints (and their time for aligning with IMU tensor)
    rss, pos, time = get_scan_wlan_data_of_folder(folder, unique_mac_addr=unique_mac_addr, return_time=True)

    # transform time according to IMU tensor
    time -= start_time
    time /= NANO_PER_MILLI

    # compute discrete index within sequence based on timestamp and IMU sampling frequency
    time_idx = time / ms_per_imu_scan
    time_idx = np.round(time_idx, decimals=0).astype(int)

    # setup WLAN tensor with same dimension as IMU tensor
    wlan_tensor = np.full((len(acc_tensor), num_mac), missing_ap_val)
    wlan_tensor[time_idx, :] = rss

    # merge data tensor for easily obtaining shifted windows by step size
    data_tensor = np.concatenate((
        acc_tensor,
        gyro_tensor,
        mag_tensor,
        wlan_tensor),
        axis=1)

    # check for nan entries in pos tensor
    not_nan_pos_mask = np.where(~np.any(np.isnan(pos_tensor), axis=1))[0]
    data_tensor = data_tensor[not_nan_pos_mask]
    pos_tensor = pos_tensor[not_nan_pos_mask]

    data_tensor_windows, pos_tensor_windows = partition_data_via_sliding_window(
        data_tensor, pos_tensor, seq_length, step_size, output_step_size)

    # use the WLAN data of the new tensor for constructing WLAN scans per sequences
    wlan_tensor_view = data_tensor_windows[:, :, 9:]
    scan_per_window = np.sum(np.any(wlan_tensor_view != -110.0, axis=2), axis=1)
    scan_pos_per_window = np.where(np.any(wlan_tensor_view != -110.0, axis=2))

    # construct lists that hold wlan scans + relative pos within window for each window
    rss_windows = []
    keep_window_idx = []

    # construct list of wlan scans for each sequence
    for idx in range(len(wlan_tensor_view)):
        ap_idx = np.where(scan_pos_per_window[0] == idx)[0]
        if len(ap_idx) == 0:
            continue
        else:
            ap_scan = wlan_tensor_view[idx, scan_pos_per_window[1][ap_idx], :]
            time_scan = scan_pos_per_window[1][ap_idx] / seq_length
            rss_windows += [(ap_scan, time_scan)]
            keep_window_idx.append(idx)

    # add start token to pos_tensor
    if add_pos_start_token:
        pos_tensor_windows = replace_init_pos_by_start_token(pos_tensor_windows)

    # cutoff initial data from sequence (since only its position is used as initial entry (start token)
    # and filter out those windows where no wlan scan was registered
    imu_tensor_windows = data_tensor_windows[keep_window_idx, 1:, :9]
    pos_tensor_windows = pos_tensor_windows[keep_window_idx]

    return imu_tensor_windows, rss_windows, pos_tensor_windows


def replace_init_pos_by_start_token(pos_tensor_windows):
    """
    The initial position of each output sequence is replaced with a placeholder start token position.
    The first entry in the target tensor will be used as initial start token to the LSTM decoder.
    Similar to <start_token> in language model.
    :param pos_tensor_windows: Tensor of shape (num_seq, target length (O+1), 2)
    :return:
    """
    shp_pos = np.shape(pos_tensor_windows)
    slice = np.zeros((shp_pos[0], 1, shp_pos[2]))
    slice[:, 0, :] = START_TOKEN
    pos_tensor_windows = np.concatenate((slice, pos_tensor_windows[:, 1:, :]), axis=1)

    return pos_tensor_windows


def get_joint_source_data(devices=None, seq_to_point=False, return_unique_fp_for_sequence=False):
    """
    Obtains joint-source data (IMU + WLAN) for all trajectories collected by the specified devices
    and merges them together. See documentation of _get_joint_source_data for details.
    :param devices: Specified devices to include
    :param seq_to_point: Whether to reduce target position tensor to only include the last position of a sequence
    :param return_unique_fp_for_sequence: Whether to additionally return a unique fingerprint for each sequence
    :return: same as _get_joint_source_data
    """
    data_dir = 'giaIndoorLoc'
    path = "floor_1/{}/"

    if devices is None:
        devices = ['Galaxy', 'OnePlus', 'S20']

    data_folders = [path.format(dev) + '/' + d for dev in devices for d in os.listdir(data_dir + '/' + path.format(dev))
                    if not d.startswith('.')]

    unique_mac = get_unique_ap_vec(data_folders)

    # obtain data for each trajectory
    res = [_get_joint_source_data(df, unique_mac, seq_to_point=seq_to_point,
                                  return_unique_fp_for_sequence=return_unique_fp_for_sequence) for df in data_folders]

    # join data from trajectoreis
    data = np.concatenate([r[0] for r in res])
    pos = np.concatenate([r[1] for r in res])

    if return_unique_fp_for_sequence:
        rss_vec = np.concatenate([r[2] for r in res])
        time_vec = np.concatenate([r[3] for r in res])

    # delete those windows without any WLAN RSS reading
    no_wlan_seq_ids = np.all(np.all(data[:, :, 10:] == 0, axis=2), axis=1)

    print("Deleted {} % of data due to missing RSS readings in windows".format(
        np.round(len(np.where(no_wlan_seq_ids)[0]) / len(data) * 100, decimals=1)
    ))

    if return_unique_fp_for_sequence:
        data = data[~no_wlan_seq_ids]
        pos = pos[~no_wlan_seq_ids]
        rss_vec = rss_vec[~no_wlan_seq_ids]
        time_vec = time_vec[~no_wlan_seq_ids]

        return data, pos, rss_vec, time_vec

    else:
        data = data[~no_wlan_seq_ids]
        pos = pos[~no_wlan_seq_ids]

        return data, pos


def get_multi_source_data(devices=None, step_size=10, seq_length=200, output_step_size=20):
    """
    Obtains multi-source data (IMU + WLAN in separate time-domain) for all trajectories collected by the specified
    devices and merges them together. See documentation of _get_multi_source_data for details.
    :param devices: Specified devices to include
    :param step_size: Number of steps that the window is forwarded to obtain the next sequence
    :param seq_length: IMU sequence length (T)
    :param output_step_size: Every output_step_size-th (of T) value is taken to form position tensor
    :return: same as _get_multi_source_data
    """
    data_dir = 'giaIndoorLoc'
    path = "floor_1/{}/"

    if devices is None:
        devices = ['Galaxy', 'OnePlus', 'S20']

    data_folders = [path.format(dev) + '/' + d for dev in devices for d in os.listdir(data_dir + '/' + path.format(dev))
                    if not d.startswith('.')]

    unique_mac = get_unique_ap_vec(data_folders)

    # obtain data for each trajectory
    res = [_get_multi_source_data(df, unique_mac, add_pos_start_token=True, step_size=step_size,
                                  seq_length=seq_length, output_step_size=output_step_size) for df in data_folders]

    # join data from trajectories
    imu = np.concatenate([r[0] for r in res])
    rss = [val for sublist in [r[1] for r in res] for val in sublist]
    pos = np.concatenate([r[2] for r in res])

    return imu, rss, pos


if __name__ == '__main__':
    data, pos, rss_vec, time_vec = get_joint_source_data(return_unique_fp_for_sequence=True)
    imu, wlan, pos = get_multi_source_data()
