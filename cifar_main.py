from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
# import torchvision.transforms as transforms
import models.cifar.transforms as transforms
import os, sys
import argparse

# from models import *
import progressbar
from torch.autograd import Variable
from models.cifar import vgg
from models.cifar import inception
from models.cifar import densenet
from py_utils import dir_utils
from pt_utils import t_sets
import RRSVM.RRSVM as RRSVM
parser = argparse.ArgumentParser(description='PyTorch CIFAR Training on VGG')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar/cifar100]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default='VGG16_F_O', help='Type of model')
parser.add_argument("--gpu_id", default=None, type=int)
parser.add_argument('--positive_constraint', '-p', action='store_true', help='positivity constraint')
parser.add_argument('--multiGpu', '-m', action='store_true', help='positivity constraint')
parser.add_argument('--mbatch_size', default=64, type=int, help='The batch size would be 64, but this can be fractions of 64 to fit into memory ')
parser.add_argument('--n_epochs', default=300, type=int)
parser.add_argument('--net', default='vgg', type=str, help='If DenseNet, please do DenseNet_L_k')

args = parser.parse_args()

use_cuda = torch.cuda.is_available() and (args.gpu_id is not None or args.multiGpu)


best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.transform_train
transform_test = transforms.transform_test

full_batch_size = 64  # Too large to fit a 64 into it

useMiniBatch = False
if args.mbatch_size != full_batch_size:
    useMiniBatch = True


train_batch_size = args.mbatch_size
update_rate = int(full_batch_size/train_batch_size) + 1

test_batch_size = 100


kwargs = {'num_workers': 4, 'pin_memory': True}
if args.dataset.lower() == 'cifar10':
    print("CIFAR10")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, **kwargs)
    n_traindata = len(trainset)
    print('# training: {:d}\t batch size: {:d}, # batch: {:d}'.format(n_traindata, train_batch_size, n_traindata//train_batch_size))

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, **kwargs)
    n_testdata = len(testset)
    print('# testing: {:d}\t batch size: {:d}, # batch: {:d}'.format(n_testdata, test_batch_size, n_testdata//test_batch_size))
    n_classes = 10
elif args.dataset.lower() == 'cifar100':
    print("CIFAR100")
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, **kwargs)
    n_traindata = len(trainset)
    print('# training: {:d}\t batch size: {:d}, # batch: {:d}'.format(n_traindata, train_batch_size,
                                                                      n_traindata // train_batch_size))

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, **kwargs)
    n_testdata = len(testset)
    print('# testing: {:d}\t batch size: {:d}, # batch: {:d}'.format(n_testdata, test_batch_size, n_testdata // test_batch_size))
    n_classes = 100
else:
    print('Dataset not recognized')
    raise NameError


print ("Net: {:s}\tModel:{:s}".format(args.net, args.model))


if args.net.lower() == 'vgg':
    net = vgg.VGG(args.model.upper(), n_classes=n_classes)
elif args.net.lower() == 'inception':
    if args.model.upper() == "O_Master".upper():
        net = inception.GoogLeNet(n_classes=n_classes, useRRSVM=True)
    elif args.model.upper() == 'Orig'.upper():
        net = inception.GoogLeNet(n_classes=n_classes, useRRSVM=False)
    else:
        raise NotImplemented
elif 'densenet' in args.net.lower():
    hyper_params = args.net.lower().split('_')
    depth = int(hyper_params[1])
    growth_rate = int(hyper_params[2])
    if args.model.upper() == "O_Master".upper():
        net = densenet.DenseNet3(depth=depth, growth_rate=growth_rate, n_classes=n_classes, useRRSVM=True)
    elif args.model.upper() == 'Orig'.upper():
        net = densenet.DenseNet3(depth=depth, growth_rate=growth_rate, n_classes=n_classes, useRRSVM=False)
    else:
        raise NotImplemented
else:
    raise  NotImplemented

print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(filter(lambda p:p.requires_grad,  net.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-4) # was 5e-4 before


p_constraint = False
if args.positive_constraint:
    p_constraint = True
    positive_clipper = RRSVM.RRSVM_PositiveClipper()


if use_cuda:
    if args.multiGpu:
        device_count = torch.cuda.device_count()
        print("Using {:d} GPUs".format(device_count))
        net.cuda()
        net = nn.DataParallel(net, device_ids=[i for i in range(device_count)])
    else:
        torch.cuda.set_device(args.gpu_id)
        net.cuda()

    criterion.cuda()


# Model


save_dir = './snapshots/{:s}_{:s}_{:s}'.format(args.dataset.lower(), args.net.lower(), args.model.upper())
if args.positive_constraint:
    save_dir = './snapshots/{:s}_{:s}_{:s}_p'.format(args.dataset.lower(), args.net.lower(), args.model.upper())
save_dir = dir_utils.get_dir(save_dir)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(os.path.join(save_dir, 'ckpt.t7')), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(save_dir, 'ckpt.t7'), map_location=lambda storage, loc: storage)
    state_dict = checkpoint['net']
    net.load_state_dict(state_dict)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Training from scratch..')
    if os.path.isfile(os.path.join(save_dir, 'log.txt')):
        os.remove(os.path.join(save_dir, 'log.txt'))


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    total_n = 0
    # before it was 50 for each
    lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    t_sets.set_lr(optimizer, lr)
    pbar = progressbar.ProgressBar(max_value=n_traindata//train_batch_size)

    # acc_batch_loss = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()


        if batch_idx % update_rate ==0:
            optimizer.zero_grad()
            # acc_batch_loss = 0

        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # acc_batch_loss += loss.cpu().numpy()[0]

        if batch_idx % update_rate ==0:
            optimizer.step()
        train_loss += loss.data[0]


        # print("Trainloss: {:.02f}".format(loss.data[0]))
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        pbar.update(batch_idx)
        total_n = batch_idx + 1
        if p_constraint and positive_clipper.frequency % total_n == 0:
            net.apply(positive_clipper)

    # Save checkpoint.
    t_acc = 100. * correct / total
    t_loss = train_loss / (total_n)

    w_line = '\nTrain:\tLoss: {:d}\tAcc: {:.04f}\t{:.04f}\tLR {:0.6f}'.format(epoch, t_loss, t_acc, lr)
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

    w_line = '\nVal:\{:d}\tLoss: {:.04f}\tAcc: {:.04f}'.format(epoch, t_loss, t_acc)
    print (w_line)
    # save_dir = dir_utils.get_dir('./snapshots/cifar10_vgg_{:s}'.format(args.model))
    log_file = os.path.join(save_dir, 'log.txt')
    if not os.path.isfile(log_file):
        with open(log_file, 'w') as f:
            f.write(w_line)
    else:
        with open(log_file, 'a') as f:
            f.write(w_line)
    sys.stdout.flush()

    if t_acc > best_acc:

        state = {
            'net': net.state_dict(),
            'acc': t_acc,
            'loss': t_loss,
            'epoch': epoch,
        }

        torch.save(state,  os.path.join(save_dir,'ckpt.t7'))
        best_acc = t_acc

for epoch in range(start_epoch, start_epoch+args.n_epochs):
    train(epoch)
    test(epoch)

