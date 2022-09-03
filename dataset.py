import torch
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt
import random


def dataset():

    # source_labels
    file_str = './dataset/labels_manually.bin'
    source_data = np.fromfile(file_str, dtype=np.float32)
    source_data = np.reshape(source_data, [1000, 1, 128, 128], 'F')
    # plt.imshow(source_data[0, 0, :, :])
    # plt.show()

    # target_data
    file_str = './dataset/target_data_kerry_filter.bin'
    target_data = np.fromfile(file_str, dtype=np.float32)
    target_data = np.reshape(target_data, [1000, 1, 128, 128], 'F')
    for i in range(1000):
        temp = target_data[i, 0, :, :]
        temp = (temp - temp.mean()) / temp.std()
        # temp = 2 * (temp - temp.min()) / (temp.max() - temp.min()) - 1
        target_data[i, 0, :, :] = temp
    # plt.imshow(target_data[0, 0, :, :])
    # plt.show()

    source_data = torch.from_numpy(source_data)
    target_data = torch.from_numpy(target_data)

    # data set
    data_set = Data.TensorDataset(source_data, target_data)

    return data_set
