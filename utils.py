import math
import numpy as np
import torch


def batch_iter(data, batch_size, shuffle=False, static_rss=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)

        if static_rss:
            imu = torch.stack([e[0] for e in examples], dim=0)
            rss = torch.stack([e[1] for e in examples], dim=0)
            pos = torch.stack([e[2] for e in examples], dim=0)

            yield imu, rss, pos
        else:
            src_sents = torch.stack([e[0] for e in examples], dim=0)
            tgt_sents = torch.stack([e[1] for e in examples], dim=0)

            yield src_sents, tgt_sents


def batch_iter_no_tensor(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[1]), reverse=True)

        imu = [e[0] for e in examples]
        rss = [e[1] for e in examples]
        pos = [e[2] for e in examples]

        yield imu, rss, pos


def get_padded_wlan_tensor(wlan_scans_per_sequences):
    """
    Constructs a padded tensor of length (max scan per sequence) of the given wlan scans
    :param wlan_scans_per_sequences: List of WLAN fingerprints for each sequence
    :return: Padded tensor that can be used by WLAN encoder LSTM
    """
    source_length = [len(rss) for rss in wlan_scans_per_sequences]
    max_len = max(source_length)
    num_ap = np.shape(wlan_scans_per_sequences[0])[1]

    padded = [np.concatenate((rss_w, np.full((max_len - len(rss_w), num_ap), -110.0))) for rss_w in wlan_scans_per_sequences]

    padded = np.stack(padded, axis=1)

    return torch.Tensor(padded), source_length
