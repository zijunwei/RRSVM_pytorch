"""
Experiment code for RRSVM modules

Adapted from Zijun's Mnist_main.py
By: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
Created: 29-Dec-2017
Last modified: 29-Dec-2017
"""

from __future__ import print_function
import os, sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/RRSVM_pytorch')
sys.path.append(project_root)

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datasets.Cifar10
from torchvision import transforms
from torch.autograd import Variable
from RRSVM.SoftMax_RRSVM import RRSVM_Module as SoftRRSVM_Module  # SoftMax RRSVM
from RRSVM.RRSVM import RRSVM_Module as RRSVM_Module # RRSVM
from pt_utils import cuda_model
from py_utils import m_progressbar
from torch.utils.data.sampler import SubsetRandomSampler

import time
import nets.Cifar_ResNet as ResNet

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Cifar10 Example')
parser.add_argument('--train-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')

parser.add_argument("--gpu_id", default=1, type=str)
parser.add_argument('--multiGpu', '-m', action='store_true', help='positivity constraint')

parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--pool-method', default='AVG',
                    help='pooling method (AVG|RRSVM|SoftRRSVM) (default: AVG)')

args = parser.parse_args()
args.cuda = cuda_model.ifUseCuda(args.gpu_id, args.multiGpu)


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


train_loader, test_loader =datasets.Cifar10.get_cifar10_datasets(args, train_portion=0.1) # using 10% of all data

model = ResNet.ResNet18_RRSVM(args.pool_method)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
loss_fn = torch.nn.CrossEntropyLoss()
if args.cuda:
    model = cuda_model.convertModel2Cuda(model, args)
    loss_fn = loss_fn.cuda()
print("Number of Params:\t{:d}".format(sum([p.data.nelement() for p in model.parameters()])))


def train():
    model.train()
    start_time = time.time()
    n_train_exs = 0
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        n_train_exs += len(data)
        total_loss += len(data) * loss.data[0]
        m_progressbar.progress_bar(n_train_exs, len(train_loader.sampler), start_time=start_time,
                        prefix_message='Training progress',
                        post_message='Ave loss {:.6f}'.format(total_loss / n_train_exs))


def test(data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    n_tst_exs = 0
    for data, target in data_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        loss = loss_fn(output, target)
        test_loss += len(data) * loss.data[0]
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        n_tst_exs += len(data)

    test_acc = 100. * correct / n_tst_exs
    test_loss /= n_tst_exs
    print('  Average loss: {:8.5f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, n_tst_exs, test_acc))
    return test_acc, test_loss


for epoch in range(1, args.epochs + 1):
    print('===> Epoch {}'.format(epoch))
    train()
    print("Testing on train data")
    test(train_loader)
    print("Testing on test data")
    test(test_loader)