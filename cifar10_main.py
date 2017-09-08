'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os, sys
import argparse

# from models import *
import progressbar
from torch.autograd import Variable
from datasets.cifar10 import lenet
from py_utils import dir_utils
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default='OWA', help='Type of model')
parser.add_argument("--gpu_id", default=None, type=int)

args = parser.parse_args()
model = {'OWA': lenet.LeNet_RRSVM, 'BASE': lenet.LeNet_Base}
use_cuda = torch.cuda.is_available() and args.gpu_id is not None
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_batch_size = 128
test_batch_size = 100
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)
n_traindata = len(trainset)
print('# training: {:d}\t batch size: {:d}, # batch: {:d}'.format(n_traindata, train_batch_size, n_traindata//train_batch_size))

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)
n_testdata = len(testset)
print('# testing: {:d}\t batch size: {:d}, # batch: {:d}'.format(n_testdata, test_batch_size, n_testdata//test_batch_size))

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model

save_dir = dir_utils.get_dir('./snapshots/cifar10_letnet_{:s}'.format(args.model))



if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(os.path.join(save_dir, 'ckpt.t7')), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(save_dir, 'ckpt.t7'))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    net_struct = model[args.model.upper()]
    net = net_struct()
    if os.path.isfile(os.path.join(save_dir, 'log.txt')):
        os.remove(os.path.join(save_dir, 'log.txt'))


print ("Model:{:s}".format(args.model))

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    total_n = 0
    pbar = progressbar.ProgressBar(max_value=n_traindata//train_batch_size)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        pbar.update(batch_idx)
        total_n = batch_idx + 1
    # Save checkpoint.
    t_acc = 100. * correct / total
    t_loss = train_loss / (total_n)

    w_line = '\nTrain:\t{:d}\t{:.04f}\t{:.04f}\n'.format(epoch, t_loss, t_acc)
    print (w_line)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    total_n = 0
    pbar = progressbar.ProgressBar(max_value=n_testdata//test_batch_size)
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        pbar.update(batch_idx)
        total_n = batch_idx + 1
    # Save checkpoint.
    t_acc = 100.*correct/total
    t_loss = test_loss/(total_n)

    w_line = '\nVal:\t{:d}\t{:.04f}\t{:.04f}\n'.format(epoch, t_loss, t_acc)
    print (w_line)
    save_dir = dir_utils.get_dir('./snapshots/cifar10_letnet_{:s}'.format(args.model))
    log_file = os.path.join(save_dir, 'log.txt')
    if not os.path.isfile(log_file):
        with open(log_file, 'w') as f:
            f.write(w_line)
    else:
        with open(log_file, 'a') as f:
            f.write(w_line)
    sys.stdout.flush()

    if t_acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': t_acc,
            'loss': t_loss,
            'epoch': epoch,
        }

        torch.save(state,  os.path.join(save_dir,'ckpt.t7'))
        best_acc = t_acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)

