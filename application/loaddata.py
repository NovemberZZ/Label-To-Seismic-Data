import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as Data


def dataset():

    # TRAIN DATA
    total_data = np.fromfile("../dataset/label_2_kerry_no_seed_epoch_48.bin", dtype=np.float32)
    total_data = np.reshape(total_data, [1000, 1, 128, 128], 'F')
    total_data = total_data.astype(np.float32)
    total_data = (total_data - np.mean(total_data)) / np.std(total_data)
    # plt.imshow(total_data[0, 0, :, :])
    # plt.show()

    total_labels = np.fromfile("../dataset/labels_manually.bin", dtype=np.float32)
    total_labels = np.reshape(total_labels, [1000, 128, 128], 'F')
    # plt.imshow(total_labels[0, :, :])
    # plt.show()

    total_data = torch.from_numpy(total_data)
    total_labels = torch.from_numpy(total_labels)

    ind_train = list(range(0, 1000))
    ind_test = list(range(900, 950))
    ind_val = list(range(950, 1000))

    train_data = total_data[ind_train]
    train_labels = total_labels[ind_train]
    test_data = total_data[ind_test]
    test_labels = total_labels[ind_test]
    val_data = total_data[ind_val]
    val_labels = total_labels[ind_val]

    # TRAIN SET
    train_set = Data.TensorDataset(train_data, train_labels)

    return train_set, val_data, val_labels, test_data, test_labels
