import os
import argparse
import itertools
import torch
import dataset
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torch.autograd import Variable
import model
# from visdom import Visdom  # python -m visdom.server


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=10, help='train batch size')
parser.add_argument('--num_epochs', type=int, default=200, help='number of train epochs')
parser.add_argument('--lrG', type=float, default=1 * 1e-4, help='learning rate for generator')
parser.add_argument('--lrE', type=float, default=1 * 1e-4, help='learning rate for generator')
parser.add_argument('--lrD', type=float, default=5 * 1e-5, help='learning rate for discriminator')
params = parser.parse_args()
print(params)

images_dir = './images/label_2_kerry_BiGAN/'
model_dir = './model/label_2_kerry_BiGAN/'
if not os.path.exists(images_dir):
    os.mkdir(images_dir)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# LOAD DATA
data_set = dataset.dataset()
train_loader = Data.DataLoader(dataset=data_set, batch_size=params.batch_size, shuffle=True)

# Models
G = model.BiGenerator()
E = model.BiGenerator()
D = model.BiDiscriminator()

G.cuda()
E.cuda()
D.cuda()

# Loss function
loss_func = torch.nn.MSELoss()

# optimizers
G_optimizer = torch.optim.Adam(G.parameters(), lr=params.lrG)
E_optimizer = torch.optim.Adam(E.parameters(), lr=params.lrG)
D_optimizer = torch.optim.Adam(D.parameters(), lr=params.lrD)


for epoch in range(params.num_epochs):
    for step, (labels, images) in enumerate(train_loader):

        labels = labels.cuda()
        images = images.cuda()

        images_ = G(labels)
        labels_ = E(images)

        real = D(labels_, images)
        fake = D(labels, images_)
        d_loss_real = loss_func(real, Variable(torch.ones(real.size()).cuda()))
        d_loss_fake = loss_func(fake, Variable(torch.zeros(fake.size()).cuda()))
        D_loss = 0.5 * (d_loss_real + d_loss_fake)

        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        images_ = G(labels)
        fake = D(labels, images_)
        G_loss = loss_func(fake, Variable(torch.ones(real.size()).cuda()))

        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        labels_ = E(images)
        real = D(labels_, images)
        E_loss = loss_func(real, Variable(torch.zeros(fake.size()).cuda()))

        E_optimizer.zero_grad()
        E_loss.backward()
        E_optimizer.step()

        # plot the images
        if (step + 1) % 20 == 0:
            plt.figure()

            plt.subplot(2, 3, 1)
            plt.imshow(labels.cpu()[0, 0, :, :], cmap='gray')
            plt.title('label1')
            plt.yticks([])
            plt.xticks([])
            plt.subplot(2, 3, 2)
            plt.imshow(labels.cpu()[1, 0, :, :], cmap='gray')
            plt.title('label2')
            plt.yticks([])
            plt.xticks([])
            plt.subplot(2, 3, 3)
            plt.imshow(labels.cpu()[2, 0, :, :], cmap='gray')
            plt.title('label3')
            plt.yticks([])
            plt.xticks([])

            plt.subplot(2, 3, 4)
            plt.imshow(images_.detach().cpu()[0, 0, :, :], cmap='gray')
            plt.title('img1')
            plt.yticks([])
            plt.xticks([])
            plt.subplot(2, 3, 5)
            plt.imshow(images_.detach().cpu()[1, 0, :, :], cmap='gray')
            plt.title('img2')
            plt.yticks([])
            plt.xticks([])
            plt.subplot(2, 3, 6)
            plt.imshow(images_.detach().cpu()[2, 0, :, :], cmap='gray')
            plt.title('img3')
            plt.yticks([])
            plt.xticks([])

            plt.savefig(images_dir + '/Epoch_' + str(epoch + 1) + '_step_' + str(step + 1) + '.jpg')
            plt.close()
        # plot the images

        if (step + 1) % 20 == 0:
            print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f, E_loss: %.4f' %
                  (epoch + 1, params.num_epochs, step + 1, len(train_loader), D_loss.item(), G_loss.item(), E_loss.item()))
            torch.save(G.state_dict(),
                       model_dir + 'G_' + 'Epoch' + str(epoch + 1) + '_step' + str(step + 1) + '.pkl')
            torch.save(E.state_dict(),
                       model_dir + 'E_' + 'Epoch' + str(epoch + 1) + '_step' + str(step + 1) + '.pkl')

