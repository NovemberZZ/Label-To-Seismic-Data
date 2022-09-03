import numpy as np
import matplotlib.pyplot as plt


def overlap_kernel(windows_size, overlap):

    kernel = np.ones([windows_size, windows_size])

    decay_list = np.arange(overlap) / (overlap-1)
    decay_array = np.expand_dims(decay_list, 1)
    corner_1 = decay_array * np.transpose(decay_array, [1, 0])
    corner_2 = np.flip(corner_1, 1)
    corner_3 = np.flip(corner_1, 0)
    corner_4 = np.flip(corner_1)

    # plt.figure(1)
    # plt.subplot(2, 2, 1)
    # plt.imshow(corner_1)
    # plt.subplot(2, 2, 2)
    # plt.imshow(corner_2)
    # plt.subplot(2, 2, 3)
    # plt.imshow(corner_3)
    # plt.subplot(2, 2, 4)
    # plt.imshow(corner_4)

    rectangle_1 = decay_array * np.ones([1, (windows_size-2*overlap)])
    rectangle_2 = np.rot90(rectangle_1, 1)
    rectangle_3 = np.rot90(rectangle_1, 2)
    rectangle_4 = np.rot90(rectangle_1, 3)

    # plt.figure(2)
    # plt.subplot(2, 2, 1)
    # plt.imshow(rectangle_1)
    # plt.subplot(2, 2, 2)
    # plt.imshow(rectangle_2)
    # plt.subplot(2, 2, 3)
    # plt.imshow(rectangle_3)
    # plt.subplot(2, 2, 4)
    # plt.imshow(rectangle_4)

    kernel[0: overlap, 0: overlap] = corner_1
    kernel[0: overlap, windows_size-overlap: windows_size] = corner_2
    kernel[windows_size-overlap: windows_size, 0: overlap] = corner_3
    kernel[windows_size-overlap: windows_size, windows_size-overlap: windows_size] = corner_4

    kernel[0: overlap, overlap: windows_size-overlap] = rectangle_1
    kernel[overlap: windows_size - overlap, 0: overlap] = rectangle_2
    kernel[windows_size - overlap: windows_size, overlap: windows_size - overlap] = rectangle_3
    kernel[overlap: windows_size - overlap, windows_size - overlap: windows_size] = rectangle_4

    # plt.figure(3)
    # plt.imshow(kernel)
    # plt.show()

    return kernel
