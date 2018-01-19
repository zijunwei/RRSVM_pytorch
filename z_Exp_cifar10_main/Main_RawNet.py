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

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--train-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')

parser.add_argument("--gpu_id", default=0, type=str)
parser.add_argument('--multiGpu', '-m', action='store_true', help='positivity constraint')

parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--pool-method', default='R+M',
                    help='pooling method (max|RRSVM|SoftRRSVM|R+M | A+M | M+M ) (default: RRSVM)')

args = parser.parse_args()
args.cuda = torch.cuda.is_available() and (args.gpu_id is not None or args.multiGPU)


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


train_loader, test_loader =datasets.Cifar10.get_cifar10_datasets(args, train_portion=0.1) # using 10% of all data


class Net(nn.Module):
    def __init__(self, pool_method):
        ksize = 2
        psize = (ksize-1)/2
        d1 = 10
        d2 = 20
        d3 = 50
        self.d2b = 25*d2

        gksize = 7 # global pooling or bigger receptive field
        gpsize = (gksize-1)/2

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, d1, kernel_size=5)
        self.conv2 = nn.Conv2d(d1, d2, kernel_size=5)
        if pool_method == 'max':
            self.pool1 = torch.nn.MaxPool2d(kernel_size=ksize, stride=2, padding=psize)
            self.pool2 = torch.nn.MaxPool2d(kernel_size=ksize, stride=2, padding=psize)
        elif pool_method == 'RRSVM':
            self.pool1 = RRSVM_Module(in_channels=d1, init='eps_max', kernel_size=ksize, stride=2, padding=psize)
            self.pool2 = RRSVM_Module(in_channels=d2, init='eps_max', kernel_size=ksize, stride=2, padding=psize)
        elif pool_method == 'SoftRRSVM':
            self.pool1 = SoftRRSVM_Module(in_channels=d1, kernel_size=ksize, stride=2, padding=psize)
            self.pool2 = SoftRRSVM_Module(in_channels=d2, kernel_size=ksize, stride=2, padding=psize)
        elif pool_method == 'R+M':
            self.pool1 = torch.nn.MaxPool2d(kernel_size=ksize, stride=2, padding=psize)
            self.pool2 = torch.nn.MaxPool2d(kernel_size=ksize, stride=2, padding=psize)
            self.gpool1 = RRSVM_Module(in_channels=d1, init='eps_max', kernel_size=gksize, stride=2, padding=gpsize)
            self.gpool2 = RRSVM_Module(in_channels=d2, init='eps_max', kernel_size=gksize, stride=2, padding=gpsize)
        elif pool_method == 'A+M':
            self.pool1 = torch.nn.MaxPool2d(kernel_size=ksize, stride=2, padding=psize)
            self.pool2 = torch.nn.MaxPool2d(kernel_size=ksize, stride=2, padding=psize)
            self.gpool1 = torch.nn.AvgPool2d(kernel_size=gksize, stride=2, padding=gpsize)
            self.gpool2 = torch.nn.AvgPool2d(kernel_size=gksize, stride=2, padding=gpsize)
        elif pool_method == 'M+M':
            self.pool1 = torch.nn.MaxPool2d(kernel_size=ksize, stride=2, padding=psize)
            self.pool2 = torch.nn.MaxPool2d(kernel_size=ksize, stride=2, padding=psize)
            self.gpool1 = torch.nn.MaxPool2d(kernel_size=gksize, stride=2, padding=gpsize)
            self.gpool2 = torch.nn.MaxPool2d(kernel_size=gksize, stride=2, padding=gpsize)
        elif pool_method == 'GR+M':
            self.pool1 = torch.nn.MaxPool2d(kernel_size=ksize, stride=2, padding=psize)
            self.pool2 = torch.nn.MaxPool2d(kernel_size=ksize, stride=2, padding=psize)
            self.gpool1 = RRSVM_Module(in_channels=d1, init='eps_max', kernel_size=24, stride=24, padding=0)
            self.gpool2 = RRSVM_Module(in_channels=d2, init='eps_max', kernel_size=8, stride=8, padding=0)
        elif pool_method == 'GM+M':
            self.pool1 = torch.nn.MaxPool2d(kernel_size=ksize, stride=2, padding=psize)
            self.pool2 = torch.nn.MaxPool2d(kernel_size=ksize, stride=2, padding=psize)
            self.gpool1 = RRSVM_Module(in_channels=d1, init='eps_max', kernel_size=24, stride=24, padding=0)
            self.gpool2 = RRSVM_Module(in_channels=d2, init='eps_max', kernel_size=8, stride=8, padding=0)

        elif pool_method == 'GA+M':
            raise  NotImplementedError


        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(self.d2b, d3)
        self.fc2 = nn.Linear(d3, 10)
        self.pool_method = pool_method

    def forward(self, x):
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv1(x)
        if self.pool_method in {'R+M', 'A+M', "M+M", 'GR+M'}:
            x1 = self.pool1(x)
            x2 = self.gpool1(x)
            # x = torch.cat((x1, x2), 1)
            x = x1 + x2
        else:
            x = self.pool1(x)

        x = F.relu(x)

        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.conv2(x)
        x = self.conv2_drop(x)

        if self.pool_method in {'R+M', 'A+M', "M+M", 'GR+M'}:
            x1 = self.pool2(x)
            x2 = self.gpool2(x)
            # x = torch.cat((x1, x2), 1)
            x = x1 + x2
        else:
            x = self.pool1(x)

        x = F.relu(x)

        x = x.view(-1, self.d2b)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # return x
        return F.log_softmax(x)

model = Net(pool_method=args.pool_method)

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