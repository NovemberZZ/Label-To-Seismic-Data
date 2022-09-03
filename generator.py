import torch
import dataset
import numpy as np
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt
from model import Generator, CoupledGenerators, BiGenerator


# load data
data_set = dataset.dataset()
train_loader = Data.DataLoader(dataset=data_set, batch_size=1, shuffle=False)

# CycleGAN model
G = Generator(in_channels=1, n_classes=1, depth=4, wf=5, padding=True, batch_norm=True, up_mode='upconv')
G.cuda()
G.load_state_dict(torch.load('./model/label_2_kerry_SSIMLoss/generator_A_param_Epoch100_step100.pkl'))

# BiGAN model
# G = BiGenerator()
# G.cuda()
# G.load_state_dict(torch.load('./model/label_2_kerry_BiGAN/G_Epoch200_step100.pkl'))

output_seis_images = np.zeros([1000, 1, 128, 128])

# training
for step, (input_labels, _,) in enumerate(train_loader):
    print('complete...%d/%d' % (int(step+1), int(len(train_loader))))
    input_labels = input_labels.cuda()
    seis_images = G(input_labels)
    seis_images = seis_images.detach().cpu()
    seis_images = (seis_images - seis_images.mean()) / seis_images.std()
    output_seis_images[step, 0, :, :] = seis_images
print('finish!')

output_seis_images = np.reshape(output_seis_images, [1000 * 128 * 128], 'F')

output_seis_images.astype(np.float32).tofile('./dataset/label_2_kerry_SSIMLoss.bin')


# ******************************** CoGAN model ********************************** #
# G = CoupledGenerators(1, 128, 1024)
# G.cuda()
# G.load_state_dict(torch.load('./model/label_2_kerry_CoGAN/generators_Epoch200_step100.pkl'))
#
# output_seis_images = np.zeros([1000, 1, 128, 128])
# output_seis_labels = np.zeros([1000, 1, 128, 128])
#
# # training
# for step in range(1000):
#     print('complete...%d/%d' % (int(step+1), int(1000)))
#     labels, seis_images = G(Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (1, 1024)))))
#     seis_images = seis_images.detach().cpu()
#     seis_images = (seis_images - seis_images.mean()) / seis_images.std()
#     labels = torch.squeeze((torch.squeeze(labels))).detach().cpu()
#     threshold = 0.3
#     ind1 = np.where(labels <= threshold)
#     ind2 = np.where(labels > threshold)
#     labels[ind1] = 0
#     labels[ind2] = 1
#     output_seis_images[step, 0, :, :] = seis_images
#     output_seis_labels[step, :, :] = labels
# print('finish!')
#
# output_seis_images = np.reshape(output_seis_images, [1000 * 128 * 128], 'F')
# output_seis_images.astype(np.float32).tofile('./dataset/label_2_kerry_CoGAN.bin')
#
# output_seis_labels = np.reshape(output_seis_labels, [1000 * 128 * 128], 'F')
# output_seis_labels.astype(np.float32).tofile('./dataset/label_2_kerry_CoGAN_labels.bin')
