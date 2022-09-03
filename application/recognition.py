import time
import model
import win_func
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

time_start = time.time()

win_size = 128
overlap = int(win_size/2)
output_max_threshold = 20

print('Loading data ...')
input_data = np.loadtxt('./test_data/test_data_kerry_filter_100.dat')
# input_data = np.flip(input_data, 1)
input_data = np.pad(input_data, pad_width=((overlap, win_size), (overlap, win_size)), mode='reflect')
output_data = np.zeros(np.shape(input_data))

print('Loading model ...')
model = model.UNet(in_channels=1, n_classes=2, depth=5, wf=5, padding=True, batch_norm=True, up_mode='upconv')
model = model.cuda()
model.load_state_dict(torch.load(
    './model/label_2_kerry_no_seed_epoch_48/checkpoint_depth5_wf5_bs100_lr1e_5_epoch200.pkl'))
print('Recognition ...')

kernel = win_func.overlap_kernel(windows_size=win_size, overlap=overlap)

for Time in range(0, np.size(input_data, 0), (win_size-overlap)):
    if (Time + win_size) > np.size(input_data, 0):
        break
    for CDP in range(0, np.size(input_data, 1), (win_size-overlap)):
        if (CDP + win_size) > np.size(input_data, 1):
            break
        win_data = input_data[Time:Time+win_size, CDP:CDP+win_size]
        win_data = (win_data - np.mean(win_data)) / np.std(win_data)
        # win_data = 2 * (win_data - np.min(win_data)) / (np.max(win_data) - np.min(win_data))-1
        win_data = torch.from_numpy(win_data).float().cuda()
        win_data = torch.reshape(win_data, [1, 1, win_size, win_size])
        # model.eval()
        win_output = model(win_data)
        # win_output = F.softmax(win_output, dim=1)
        win_output = win_output.cpu().detach()
        channel_ind = torch.max(win_output, 1)[1]
        channel_ind = torch.squeeze(channel_ind, 0)
        fault_ind = np.where(channel_ind == 1)
        win_output = win_output[0, 1, :, :]
        temp = np.zeros([win_size, win_size])
        temp[fault_ind] = win_output[fault_ind]
        win_output = temp
        output_data[Time:Time+win_size, CDP:CDP+win_size] = \
            win_output * kernel + output_data[Time:Time+win_size, CDP:CDP+win_size]

        ind_threshold = np.where(output_data[Time:Time+win_size, CDP:CDP+win_size] >= output_max_threshold)
        output_data[Time:Time+win_size, CDP:CDP+win_size][ind_threshold] = output_max_threshold

print(output_data.max(), output_data.min())
input_data = input_data[overlap:np.size(input_data, 0)-win_size, overlap:np.size(input_data, 1)-win_size]
output_data = output_data[overlap:np.size(output_data, 0)-win_size, overlap:np.size(output_data, 1)-win_size]

time_end = time.time()
print('Elapsed time: %.2f' % (time_end - time_start))

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(input_data, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(output_data, cmap='gray')
plt.show()

