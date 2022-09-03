import torch
import loaddata
import model
import numpy as np
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

# LOAD DATA
train_set, val_data, val_labels, test_data, test_labels = loaddata.dataset()
val_data = val_data.cuda()
test_data = test_data.cuda()
train_loader = Data.DataLoader(dataset=train_set, batch_size=100, shuffle=True)

# LOAD NET
model = model.UNet(in_channels=1, n_classes=2, depth=5, wf=5, padding=True, batch_norm=True, up_mode='upconv')
model.cuda()

# SET PARA
EPOCH = 200
LEARN_RATE = 1 * 1e-5

# TRAIN
optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE, weight_decay=1e-4)
loss_func = torch.nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, train_set in enumerate(train_loader):
        train_data, train_labels = train_set

        # ########################################################################### #
        l0 = float(np.equal(train_labels, 0).sum())
        l1 = float(np.equal(train_labels, 1).sum())
        w0 = (1/(l0/(l0 + l1)))/(1/(l0/(l0 + l1))+1/(l1/(l0 + l1)))
        w1 = (1/(l1/(l0 + l1)))/(1/(l0/(l0 + l1))+1/(l1/(l0 + l1)))
        weight = torch.FloatTensor([w0, w1]).cuda()
        # ########################################################################### #

        train_data = train_data.cuda()
        train_labels = train_labels.cuda()

        model.train()
        output = model(train_data)
        loss = F.cross_entropy(output, train_labels.long(), weight=weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step+1) % 10 == 0:
            with torch.no_grad():
                model.eval()
                val_output = model(val_data)

                # ########################################################################### #
                l0 = float(np.equal(val_labels, 0).sum())
                l1 = float(np.equal(val_labels, 1).sum())
                w0 = (1 / (l0 / (l0 + l1))) / (1 / (l0 / (l0 + l1)) + 1 / (l1 / (l0 + l1)))
                w1 = (1 / (l1 / (l0 + l1))) / (1 / (l0 / (l0 + l1)) + 1 / (l1 / (l0 + l1)))
                weight = torch.FloatTensor([w0, w1]).cuda()
                # ########################################################################### #

                val_labels = val_labels.cuda()
                loss_val = F.cross_entropy(val_output, val_labels.long(), weight=weight)
                val_labels = val_labels.cpu()
                val_output = torch.max(val_output, 1)[1].cpu()
                val_output = val_output.float()
                val_labels = torch.squeeze(torch.FloatTensor(val_labels))
                accuracy = float((val_output == val_labels).sum()) / float(
                    val_labels.size(0) * val_labels.size(1) * val_labels.size(2))
                print('Epoch: ', epoch+1, '| train loss: %.4f' % loss, '| validation loss: %.4f' % loss_val,
                      '| validation accuracy: %.4f' % accuracy)

    # SAVE MODEL
    if (epoch+1) >= 20 and (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(),
                   './model/label_2_kerry_no_seed_epoch_48/checkpoint_depth5_wf5_bs100_lr1e_5_epoch' + str(
                       epoch + 1) + '.pkl')

# TEST
test_output = model(test_data)
test_output = torch.max(test_output, 1)[1].cpu()
test_output = test_output.float()
test_labels = torch.squeeze(torch.FloatTensor(test_labels))
accuracy = float((test_output == test_labels).sum()) / \
           float(test_labels.size(0) * test_labels.size(1) * test_labels.size(2))
print('The accuracy in test set is: %.4f' % accuracy)

