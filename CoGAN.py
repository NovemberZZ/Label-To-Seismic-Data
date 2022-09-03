import argparse
import os
import numpy as np
import math
import scipy
import itertools
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import dataset
import model


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=10, help="size of the batches")
parser.add_argument("--lr", type=float, default=1 * 1e-4, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=10, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=1024, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)
img_shape = (opt.channels, opt.img_size, opt.img_size)
cuda = True if torch.cuda.is_available() else False

images_dir = './images/label_2_kerry_CoGAN/'
model_dir = './model/label_2_kerry_CoGAN/'
if not os.path.exists(images_dir):
    os.mkdir(images_dir)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# LOAD DATA
data_set = dataset.dataset()
train_loader = DataLoader(dataset=data_set, batch_size=opt.batch_size, shuffle=True)


# Model
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Loss function
adversarial_loss = torch.nn.MSELoss()

# Initialize models
coupled_generators = model.CoupledGenerators(opt.channels, opt.img_size, opt.latent_dim)
coupled_discriminators = model.CoupledDiscriminators(opt.channels, opt.img_size)

if cuda:
    coupled_generators.cuda()
    coupled_discriminators.cuda()

# Initialize weights
coupled_generators.apply(weights_init_normal)
coupled_discriminators.apply(weights_init_normal)


# Optimizers
optimizer_G = torch.optim.Adam(coupled_generators.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(coupled_discriminators.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs1, imgs2) in enumerate(train_loader):

        batch_size = imgs1.shape[0]

        # Adversarial ground truths
        valid = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        imgs1 = Variable(imgs1.type(Tensor))
        imgs2 = Variable(imgs2.type(Tensor))

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))

        # Generate a batch of images
        gen_imgs1, gen_imgs2 = coupled_generators(z)
        # Determine validity of generated images
        validity1, validity2 = coupled_discriminators(gen_imgs1, gen_imgs2)

        g_loss = (adversarial_loss(validity1, valid) + adversarial_loss(validity2, valid)) / 2

        g_loss.backward()
        optimizer_G.step()

        # ----------------------
        #  Train Discriminators
        # ----------------------

        optimizer_D.zero_grad()

        # Determine validity of real and generated images
        validity1_real, validity2_real = coupled_discriminators(imgs1, imgs2)
        validity1_fake, validity2_fake = coupled_discriminators(gen_imgs1.detach(), gen_imgs2.detach())

        d_loss = (
            adversarial_loss(validity1_real, valid)
            + adversarial_loss(validity1_fake, fake)
            + adversarial_loss(validity2_real, valid)
            + adversarial_loss(validity2_fake, fake)
        ) / 4

        d_loss.backward()
        optimizer_D.step()

        if (i + 1) % 20 == 0:
            print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f' %
                  (epoch + 1, opt.n_epochs, i + 1, len(train_loader), d_loss.item(), g_loss.item()))
            torch.save(coupled_generators.state_dict(),
                       model_dir + 'generators_' + 'Epoch' + str(epoch + 1) + '_step' + str(i + 1) + '.pkl')
            torch.save(coupled_discriminators.state_dict(),
                       model_dir + 'discriminators_' + 'Epoch' + str(epoch + 1) + '_step' + str(i + 1) + '.pkl')

            gen_imgs = torch.cat((gen_imgs1.data, gen_imgs2.data), 0)
            save_image(gen_imgs, images_dir + '/Epoch_' + str(epoch + 1) + '_step_' + str(i + 1) + '.jpg', nrow=10, normalize=True)
