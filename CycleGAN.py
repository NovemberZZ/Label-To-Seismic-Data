import os
import argparse
import itertools
import torch
import dataset
import imagepool
import rand_seed
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torch.autograd import Variable
from model import Generator, Discriminator
import pytorch_ssim
# from visdom import Visdom  # python -m visdom.server

# rand_seed.rand_seed(100)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=10, help='train batch size')
parser.add_argument('--num_epochs', type=int, default=100, help='number of train epochs')
parser.add_argument('--lrG', type=float, default=5 * 1e-5, help='learning rate for generator')
parser.add_argument('--lrD', type=float, default=1 * 1e-4, help='learning rate for discriminator')
parser.add_argument('--weight_G_A', type=float, default=1, help='weight for G_A loss')
parser.add_argument('--weight_G_B', type=float, default=1, help='weight for G_B loss')
parser.add_argument('--weight_cycle_A', type=float, default=2, help='weight for cycle loss')
parser.add_argument('--weight_cycle_B', type=float, default=2, help='weight for cycle loss')
params = parser.parse_args()
print(params)

images_dir = './images/label_2_kerry_SSIMLoss/'
model_dir = './model/label_2_kerry_SSIMLoss/'
if not os.path.exists(images_dir):
    os.mkdir(images_dir)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# LOAD DATA
data_set = dataset.dataset()
train_loader = Data.DataLoader(dataset=data_set, batch_size=params.batch_size, shuffle=True)

# Models
G_A = Generator(in_channels=1, n_classes=1, depth=4, wf=5, padding=True, batch_norm=True, up_mode='upconv')
G_B = Generator(in_channels=1, n_classes=1, depth=4, wf=4, padding=True, batch_norm=True, up_mode='upconv')
D_A = Discriminator(input_dims=1, hidden_dims=32, output_dims=1)
D_B = Discriminator(input_dims=1, hidden_dims=32, output_dims=1)

G_A.cuda()
G_B.cuda()
D_A.cuda()
D_B.cuda()

# Loss function
MSE_loss = torch.nn.MSELoss().cuda()
L1_loss = torch.nn.L1Loss().cuda()

# optimizers
G_optimizer = torch.optim.Adam(itertools.chain(G_A.parameters(), G_B.parameters()), lr=params.lrG)
D_A_optimizer = torch.optim.Adam(D_A.parameters(), lr=params.lrD)
D_B_optimizer = torch.optim.Adam(D_B.parameters(), lr=params.lrD)

# Generated image pool
num_pool = 100
fake_A_pool = imagepool.ImagePool(num_pool)
fake_B_pool = imagepool.ImagePool(num_pool)

# Visdom
# viz = Visdom()
# viz_step = 0
# viz.line([[0., 0.]], [0.], win='G_A_loss_&_G_B_loss', opts=dict(title='G_A_loss_&_G_B_loss',
#                                                                 legend=['G_A_loss', 'G_B_loss']))
# viz.line([[0., 0.]], [0.], win='D_A_loss_&_D_B_loss', opts=dict(title='D_A_loss_&_D_B_loss',
#                                                                 legend=['D_A_loss', 'D_B_loss']))

for epoch in range(params.num_epochs):
    for step, (real_A, real_B) in enumerate(train_loader):

        # ************* Train A -> B ************ #
        real_A = Variable(real_A.cuda())
        fake_B = G_A(real_A)
        D_B_fake_decision = D_B(fake_B)
        G_A_loss = MSE_loss(D_B_fake_decision, Variable(torch.ones(D_B_fake_decision.size()).cuda()))

        recon_A = G_B(fake_B)
        # cycle_A_loss = L1_loss(recon_A, real_A)
        cycle_A_loss = 1 - pytorch_ssim.ssim(recon_A, real_A)

        # ************* Train B -> A ************ #
        real_B = Variable(real_B.cuda())
        fake_A = G_B(real_B)
        D_A_fake_decision = D_A(fake_A)
        G_B_loss = MSE_loss(D_A_fake_decision, Variable(torch.ones(D_A_fake_decision.size()).cuda()))

        recon_B = G_A(fake_A)
        # cycle_B_loss = L1_loss(recon_B, real_B)
        cycle_B_loss = 1 - pytorch_ssim.ssim(recon_B, real_B)

        G_loss = G_A_loss * params.weight_G_A + cycle_A_loss * params.weight_cycle_A +\
                 G_B_loss * params.weight_G_B + cycle_B_loss * params.weight_cycle_B

        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        # plot the images
        # if (step + 1) % 20 == 0:
        #     plt.figure()
        #
        #     plt.subplot(3, 7, 1)
        #     plt.imshow(real_A.cpu()[0, 0, :, :], cmap='gray')
        #     plt.title('real_A')
        #     plt.yticks([])
        #     plt.xticks([])
        #     plt.subplot(3, 7, 2)
        #     plt.imshow(fake_B.detach().cpu()[0, 0, :, :], cmap='gray')
        #     plt.title('fake_B')
        #     plt.yticks([])
        #     plt.xticks([])
        #     plt.subplot(3, 7, 3)
        #     plt.imshow(recon_A.detach().cpu()[0, 0, :, :], cmap='gray')
        #     plt.title('recon_A')
        #     plt.yticks([])
        #     plt.xticks([])
        #     plt.subplot(3, 7, 4)
        #     plt.imshow(real_B.cpu()[0, 0, :, :], cmap='gray')
        #     plt.title('real_B')
        #     plt.yticks([])
        #     plt.xticks([])
        #     plt.subplot(3, 7, 5)
        #     plt.imshow(fake_A.detach().cpu()[0, 0, :, :], cmap='gray')
        #     plt.title('fake_A')
        #     plt.yticks([])
        #     plt.xticks([])
        #     plt.subplot(3, 7, 6)
        #     plt.imshow(recon_B.detach().cpu()[0, 0, :, :], cmap='gray')
        #     plt.title('recon_B')
        #     plt.yticks([])
        #     plt.xticks([])
        #
        #     plt.subplot(3, 7, 8)
        #     plt.imshow(real_A.cpu()[1, 0, :, :], cmap='gray')
        #     plt.title('real_A')
        #     plt.yticks([])
        #     plt.xticks([])
        #     plt.subplot(3, 7, 9)
        #     plt.imshow(fake_B.detach().cpu()[1, 0, :, :], cmap='gray')
        #     plt.title('fake_B')
        #     plt.yticks([])
        #     plt.xticks([])
        #     plt.subplot(3, 7, 10)
        #     plt.imshow(recon_A.detach().cpu()[1, 0, :, :], cmap='gray')
        #     plt.title('recon_A')
        #     plt.yticks([])
        #     plt.xticks([])
        #     plt.subplot(3, 7, 11)
        #     plt.imshow(real_B.cpu()[1, 0, :, :], cmap='gray')
        #     plt.title('real_B')
        #     plt.yticks([])
        #     plt.xticks([])
        #     plt.subplot(3, 7, 12)
        #     plt.imshow(fake_A.detach().cpu()[1, 0, :, :], cmap='gray')
        #     plt.title('fake_A')
        #     plt.yticks([])
        #     plt.xticks([])
        #     plt.subplot(3, 7, 13)
        #     plt.imshow(recon_B.detach().cpu()[1, 0, :, :], cmap='gray')
        #     plt.title('recon_B')
        #     plt.yticks([])
        #     plt.xticks([])
        #
        #     plt.subplot(3, 7, 15)
        #     plt.imshow(real_A.cpu()[2, 0, :, :], cmap='gray')
        #     plt.title('real_A')
        #     plt.yticks([])
        #     plt.xticks([])
        #     plt.subplot(3, 7, 16)
        #     plt.imshow(fake_B.detach().cpu()[2, 0, :, :], cmap='gray')
        #     plt.title('fake_B')
        #     plt.yticks([])
        #     plt.xticks([])
        #     plt.subplot(3, 7, 17)
        #     plt.imshow(recon_A.detach().cpu()[2, 0, :, :], cmap='gray')
        #     plt.title('recon_A')
        #     plt.yticks([])
        #     plt.xticks([])
        #     plt.subplot(3, 7, 18)
        #     plt.imshow(real_B.cpu()[2, 0, :, :], cmap='gray')
        #     plt.title('real_B')
        #     plt.yticks([])
        #     plt.xticks([])
        #     plt.subplot(3, 7, 19)
        #     plt.imshow(fake_A.detach().cpu()[2, 0, :, :], cmap='gray')
        #     plt.title('fake_A')
        #     plt.yticks([])
        #     plt.xticks([])
        #     plt.subplot(3, 7, 20)
        #     plt.imshow(recon_B.detach().cpu()[2, 0, :, :], cmap='gray')
        #     plt.title('recon_B')
        #     plt.yticks([])
        #     plt.xticks([])
        #
        #     plt.savefig(images_dir + '/Epoch_' + str(epoch + 1) + '_step_' + str(step + 1) + '.jpg')
        #     plt.close()
        # plot the images

        # ************* Train discriminator D_A ************ #
        D_A_real_decision = D_A(real_A)
        D_A_real_loss = MSE_loss(D_A_real_decision, Variable(torch.ones(D_A_real_decision.size()).cuda()))
        fake_A = fake_A_pool.query(fake_A)
        D_A_fake_decision = D_A(fake_A)
        D_A_fake_loss = MSE_loss(D_A_fake_decision, Variable(torch.zeros(D_A_fake_decision.size()).cuda()))

        # Back propagation
        D_A_loss = (D_A_real_loss + D_A_fake_loss)
        D_A_optimizer.zero_grad()
        D_A_loss.backward()
        D_A_optimizer.step()

        # ************* Train discriminator D_B ************ #
        D_B_real_decision = D_B(real_B)
        D_B_real_loss = MSE_loss(D_B_real_decision, Variable(torch.ones(D_B_real_decision.size()).cuda()))
        fake_B = fake_B_pool.query(fake_B)
        D_B_fake_decision = D_B(fake_B)
        D_B_fake_loss = MSE_loss(D_B_fake_decision, Variable(torch.zeros(D_B_fake_decision.size()).cuda()))

        # Back propagation
        D_B_loss = (D_B_real_loss + D_B_fake_loss)
        D_B_optimizer.zero_grad()
        D_B_loss.backward()
        D_B_optimizer.step()

        # if step > 0 and step % 10 == 0:
        #     viz_step = viz_step + 1
        #     viz.line([[G_A_loss.item(), G_B_loss.item()]], [viz_step],
        #              win='G_A_loss_&_G_B_loss', update='append')
        #     viz.line([[D_A_loss.item(), D_B_loss.item()]], [viz_step],
        #              win='D_A_loss_&_D_B_loss', update='append')

        if (step + 1) % 20 == 0:
            print('Epoch [%d/%d], Step [%d/%d], '
                  'G_A_loss: %.4f, cycle_A_loss: %.4f, G_B_loss: %.4f, cycle_B_loss: %.4f, '
                  'G_loss: %.4f, D_A_loss: %.4f, D_B_loss: %.4f' %
                  (epoch + 1, params.num_epochs, step + 1, len(train_loader),
                   G_A_loss.item(), cycle_A_loss.item(), G_B_loss.item(), cycle_B_loss.item(),
                   G_loss.item(), D_A_loss.item(), D_B_loss.item()))
            torch.save(G_A.state_dict(),
                       model_dir + 'generator_A_param_' + 'Epoch' + str(epoch + 1) + '_step' + str(step + 1) + '.pkl')

# torch.save(G_B.state_dict(), model_dir + 'generator_B_param.pkl')
# torch.save(D_A.state_dict(), model_dir + 'discriminator_A_param.pkl')
# torch.save(D_B.state_dict(), model_dir + 'discriminator_B_param.pkl')
