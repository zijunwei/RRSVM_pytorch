"""
Experiment code for RRSVM modules

Adapted from Zijun's Mnist_main.py
By: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
Created: 29-Dec-2017
Last modified: 29-Dec-2017
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from RRSVM.SoftMax_RRSVM import RRSVM_Module as SoftRRSVM_Module  # SoftMax RRSVM
from RRSVM.RRSVM import RRSVM_Module as RRSVM_Module # RRSVM
from py_utils import dir_utils
import os
from torch.utils.data.sampler import SubsetRandomSampler
import time, sys

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--pool-method', default='RRSVM',
                    help='pooling method (max|RRSVM|SoftRRSVM) (default: RRSVM)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

dataset_root = dir_utils.get_dir(os.path.join(os.path.expanduser('~'), 'datasets', 'RRSVM_datasets'))
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


train_size = 6000  # use a subset of training data
train_set = datasets.MNIST(dataset_root, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
indices = torch.randperm(len(train_set))
train_indices = indices[:][:train_size or None]
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                           sampler=SubsetRandomSampler(train_indices), **kwargs)


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(dataset_root, train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)



def progress_bar(k,n, prefix_message='Progress', post_message='', start_time=None):
    """
    Show the progress bar
    k: current progress
    n: total

    Created: 23-Mar-2017, Last modified: 23-Mar-2017
    """

    n_digit = len("{:d}".format(n))

    str_format = "{:s} {{:{:d}d}}/{:d} ({{:6.2f}}%), {:s}".format(prefix_message, n_digit, n, post_message)
    if start_time:
        str_format += " elapse time: {:7.1f}s".format(time.time() - start_time)

    pre_str = '\r'
    post_str = ''
    if k == 1:
        pre_str = '\n'
    elif k == n:
        post_str = '\n'

    _ = sys.stdout.write(pre_str + str_format.format(k, 100 * k / n) + post_str)
    # sys.stdout.flush()


# class MaxPoolNet(nn.Module):
#     def __init__(self):
#         super(MaxPoolNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)
#
#     def forward(self, x):
#         ksize = 2
#         psize = (ksize-1)/2
#         x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=ksize, stride=2, padding=psize))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), kernel_size=ksize, stride=2,padding=psize))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x)

class Net(nn.Module):
    def __init__(self, pool_method):
        ksize = 2
        psize = (ksize-1)/2
        d1 = 10
        d2 = 10
        d3 = 50
        self.d2b = 16*d2

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, d1, kernel_size=5)
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

        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(self.d2b, d3)
        self.fc2 = nn.Linear(d3, 10)

    def forward(self, x):
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv1(x)
        x = self.pool1(x)
        x = F.relu(x)

        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = self.pool2(x)
        x = F.relu(x)

        x = x.view(-1, self.d2b)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # return x
        return F.log_softmax(x)


class Net2(nn.Module):
    def __init__(self, pool_method):
        ksize = 2
        psize = (ksize-1)/2
        d1 = 5
        d2 = 5
        d3 = 50
        self.d2b = 16*d2*2

        gksize = 5 # global pooling or bigger receptive field
        gpsize = (gksize-1)/2

        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, d1, kernel_size=5)
        self.conv2 = nn.Conv2d(2*d1, d2, kernel_size=5)
        if pool_method == 'max':
            self.pool1 = torch.nn.MaxPool2d(kernel_size=ksize, stride=2, padding=psize)
            self.pool2 = torch.nn.MaxPool2d(kernel_size=ksize, stride=2, padding=psize)
        elif pool_method == 'RRSVM':
            self.pool1 = RRSVM_Module(in_channels=d1, init='eps_max', kernel_size=ksize, stride=2, padding=psize)
            self.pool2 = RRSVM_Module(in_channels=d2, init='eps_max', kernel_size=ksize, stride=2, padding=psize)
        elif pool_method == 'SoftRRSVM':
            self.pool1 = SoftRRSVM_Module(in_channels=d1, kernel_size=ksize, stride=2, padding=psize)
            self.pool2 = SoftRRSVM_Module(in_channels=d2, kernel_size=ksize, stride=2, padding=psize)

        self.gpool1 = RRSVM_Module(in_channels=d1, init='eps_max', kernel_size=gksize, stride=2, padding=gpsize)
        self.gpool2 = RRSVM_Module(in_channels=d2, init='eps_max', kernel_size=gksize, stride=2, padding=gpsize)


        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(self.d2b, d3)
        self.fc2 = nn.Linear(d3, 10)

    def forward(self, x):
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv1(x)
        x1 = self.pool1(x)
        x2 = self.gpool1(x)
        #print(x1.size())
        #print(x2.size())
        x = torch.cat((x1, x2), 1)

        x = F.relu(x)

        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.conv2(x)
        x = self.conv2_drop(x)

        x1 = self.pool2(x)
        x2 = self.gpool2(x)
        x = torch.cat((x1, x2), 1)
        x = F.relu(x)

        x = x.view(-1, self.d2b)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # return x
        return F.log_softmax(x)

model = Net2(pool_method=args.pool_method)
if args.cuda:
    model.cuda()
print("Number of Params:\t{:d}".format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
loss_fn = torch.nn.CrossEntropyLoss()


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
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        n_train_exs += len(data)
        total_loss += len(data) * loss.data[0]
        progress_bar(n_train_exs, len(train_loader.sampler), start_time=start_time,
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
        loss = F.nll_loss(output, target)
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