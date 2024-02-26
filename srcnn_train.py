"""
Train
"""

from __future__ import print_function

from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from srcnn_data import get_training_set, get_test_set
from srcnn_model import SRCNN,VDSR
import time
# Settings
use_cuda = 1
upscale_factor = 3
batch_size = 10
test_batch_size = 10
learn_rate = 0.0001
epochs = 15000

seed = 3000
if use_cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed(seed)

train_set = get_training_set(upscale_factor)
test_set = get_test_set(upscale_factor)
training_data_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, batch_size=test_batch_size, shuffle=False)

srcnn = SRCNN()
# ck = torch.load('checkpoint\single_channel_model_epoch_520.pth')
# srcnn.load_state_dict(ck.state_dict())
criterion = nn.MSELoss()

if use_cuda :
    srcnn.cuda()
    criterion = criterion.cuda()

optimizer = optim.Adam(srcnn.parameters(), lr=learn_rate)


def train(epoch):
    epoch_loss = 0
    srcnn.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1])
        if use_cuda:
            input = input.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        input = torch.div(input, 255.0)
        model_out = srcnn(input)
        loss = criterion(model_out * 255.0, target)
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.data))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def test():
    avg_psnr = 0
    srcnn.eval()
    average_loss = 0
    for batch in testing_data_loader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if use_cuda:
            input = input.cuda()
            target = target.cuda()
        input = torch.div(input, 255.0)
        prediction = srcnn(input)
        prediction_255 = prediction * 255
        mse = criterion(prediction_255, target)
        psnr = 10 * log10(1 / mse.data)
        avg_psnr += psnr
        average_loss += mse.data
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    average_loss = average_loss / len(testing_data_loader)
    test_loss.append(average_loss)


def checkpoint(epoch):
    model_out_path = "./checkpoint/single_channel_Vmodel_epoch_{}.pth".format(epoch)
    torch.save(srcnn, model_out_path, )
    print("Checkpoint saved to {}".format(model_out_path))

early_stop_epochs = 150
early_stop = 0
best_epoch = 0
test_loss = []
best_loss = 100000
for epoch in range(1, epochs + 1):
    start = time.time()
    train(epoch)
    test()
    if epoch % 10 == 0:
        checkpoint(epoch)
    if test_loss[-1] < best_loss :
        best_loss = test_loss[-1]
        early_stop = 0
        best_epoch = epoch
    else:
        early_stop += 1
        if early_stop > early_stop_epochs:
            break
    end = time.time()
    print("early stop = ",early_stop)
    print("time of per epoch:{}s".format(end-start))