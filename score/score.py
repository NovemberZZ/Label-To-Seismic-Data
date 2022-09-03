import time
import UNet
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


input_data = np.fromfile("../dataset/label_2_kerry_SSIMLoss.bin", dtype=np.float32)
input_data = np.reshape(input_data, [1000, 128, 128], 'F')
input_data = input_data.astype(np.float32)
input_data = (input_data - np.mean(input_data)) / np.std(input_data)


model = UNet.UNet(in_channels=1, n_classes=2, depth=6, wf=4, padding=True, batch_norm=True, up_mode='upconv')
model = model.cuda()
model.load_state_dict(torch.load('./UNet_Score_Model.pkl'))


predict_labels = np.zeros([1000, 128, 128])

for i in range(1000):

    seis_image = input_data[i, :, :]
    seis_image = (seis_image - np.mean(seis_image)) / np.std(seis_image)
    seis_image = torch.from_numpy(seis_image).float().cuda()
    seis_image = torch.unsqueeze(torch.unsqueeze(seis_image, 0), 0)
    predict = model(seis_image)
    predict = predict.cpu().detach()
    predict = torch.max(predict, 1)[1]
    predict = torch.squeeze(predict)
    predict_labels[i, :, :] = predict

    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(torch.squeeze(torch.squeeze(seis_image.detach().cpu(), 0), 0), cmap='gray')
    # plt.subplot(1, 2, 2)
    # plt.imshow(predict, cmap='gray')
    # plt.show()


real_labels = np.fromfile("../dataset/labels_manually.bin", dtype=np.float32)
real_labels = np.reshape(real_labels, [1000, 128, 128], 'F')
real_labels = real_labels.astype(np.float32)


# *********************************** estimate ***************************** #

# pixel acc
TP_TN = (predict_labels == real_labels).sum()
FP_FN = (predict_labels != real_labels).sum()
pixel_acc = float(TP_TN/(TP_TN + FP_FN))

print('pixel_acc: %f' % pixel_acc)

# MIoU
correct_index = np.where(predict_labels == real_labels)
intersection_0 = (predict_labels[correct_index] == 0).sum()
union_0 = (predict_labels == 0).sum() + (real_labels == 0).sum() - intersection_0
IoU_0 = float(intersection_0/union_0)

print('IoU_class0: %f' % IoU_0)

correct_index = np.where(predict_labels == real_labels)
intersection_1 = (predict_labels[correct_index] == 1).sum()
union_1 = (predict_labels == 1).sum() + (real_labels == 1).sum() - intersection_1
IoU_1 = float(intersection_1/union_1)

print('IoU_class1: %f' % IoU_1)

print('MIou: %f' % ((IoU_0+IoU_1)/2))

# Mean Class acc
correct_index = np.where(predict_labels == real_labels)
correct_0 = (predict_labels[correct_index] == 0).sum()
sum_0 = (real_labels == 0).sum()
class0_acc = float(correct_0/sum_0)

print('class0_acc: %f' % class0_acc)

correct_1 = (predict_labels[correct_index] == 1).sum()
sum_1 = (real_labels == 1).sum()
class1_acc = float(correct_1/sum_1)

print('class1_acc: %f' % class1_acc)

print('class acc: %f' % ((class0_acc+class1_acc)/2))

