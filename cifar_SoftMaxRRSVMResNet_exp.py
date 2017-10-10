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
from models.cifar import SoftMaxRRSVM_ResNetXt
from py_utils import dir_utils
from pt_utils import t_sets
import RRSVM.RRSVM as RRSVM
parser = argparse.ArgumentParser(description='PyTorch CIFAR Training on Softmax RRSVM ResNetXt')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar/cifar100]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument("--gpu_id", default=None, type=str)
parser.add_argument('--positive_constraint', '-p', action='store_true', help='positivity constraint')
parser.add_argument('--multiGpu', '-m', action='store_true', help='positivity constraint')
parser.add_argument('--mbatch_size', default=64, type=int, help='The batch size would be 64, but this can be fractions of 64 to fit into memory ')
parser.add_argument('--n_epochs', default=350, type=int)
parser.add_argument('--id', default=None, type=str, help='The Id of the run')
parser.add_argument('--verbose', '-v', dest='verbose', action='store_true', help='verbose mode, if not, saved in log.txt')
args = parser.parse_args()
identifier = 'SoftMaxResNetXT'


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
# update_rate = int(full_batch_size/train_batch_size)

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


print ("Model:{:s}".format(identifier))


model = SoftMaxRRSVM_ResNetXt.ResNeXt29_2x64d(n_classes=n_classes, useRRSVM=True)


print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(filter(lambda p:p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-4) # was 5e-4 before


p_constraint = False
if args.positive_constraint:
    p_constraint = True
    positive_clipper = RRSVM.RRSVM_PositiveClipper()

use_cuda = torch.cuda.is_available() and (args.gpu_id is not None or args.multiGpu)

if use_cuda:
    if args.multiGpu:
        if args.gpu_id is None:  # using all the GPUs
            device_count = torch.cuda.device_count()
            print("Using ALL {:d} GPUs".format(device_count))
            model = nn.DataParallel(model, device_ids=[i for i in range(device_count)]).cuda()
        else:
            print("Using GPUs: {:s}".format(args.gpu_id))
            device_ids = [int(x) for x in args.gpu_id]
            model = nn.DataParallel(model, device_ids=device_ids).cuda()


    else:
        torch.cuda.set_device(int(args.gpu_id))
        model.cuda()

    criterion.cuda()
    cudnn.benchmark = True


# Model


save_dir = './snapshots/{:s}_{:s}'.format(args.dataset.lower(), identifier)
if args.positive_constraint:
    save_dir = './snapshots/{:s}_{:s}_p'.format(args.dataset.lower(), identifier)
if args.id is not None:
    save_dir = save_dir+args.id

save_dir = dir_utils.get_dir(save_dir)
if not args.verbose:
        log_file = os.path.join(save_dir, 'log.txt')
        sys.stdout = open(log_file, "w")

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(os.path.join(save_dir, 'ckpt.t7')), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(save_dir, 'ckpt.t7'), map_location=lambda storage, loc: storage)
    state_dict = checkpoint['net']
    model.load_state_dict(state_dict)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Training from scratch..')
    if os.path.isfile(os.path.join(save_dir, 'ckpt_result.txt')):
        os.remove(os.path.join(save_dir, 'ckpt_result.txt'))


def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    total_n = 0
    # before it was 150, 225 and 300
    lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    t_sets.set_lr(optimizer, lr)
    if args.verbose:
        pbar = progressbar.ProgressBar(max_value=n_traindata//train_batch_size)

    # acc_batch_loss = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()


        # if batch_idx % update_rate ==0:
        optimizer.zero_grad()
            # acc_batch_loss = 0

        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # acc_batch_loss += loss.cpu().numpy()[0]

        # if batch_idx % update_rate ==0:
        optimizer.step()
        train_loss += loss.data[0]


        # print("Trainloss: {:.02f}".format(loss.data[0]))
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        if args.verbose:
            pbar.update(batch_idx)
        total_n = batch_idx + 1
        if p_constraint and positive_clipper.frequency % total_n == 0:
            model.apply(positive_clipper)

    # Save checkpoint.
    t_acc = 100. * correct / total
    t_loss = train_loss / (total_n)

    w_line = '\nTrain:\t {:d}\tLoss: \tAcc: {:.04f}\t{:.04f}\tLR {:0.6f}'.format(epoch, t_loss, t_acc, lr)
    print (w_line)
    result_file = os.path.join(save_dir, 'ckpt_result.txt')
    if not os.path.isfile(result_file):
        with open(result_file, 'w') as f:
            f.write(w_line)
    else:
        with open(result_file, 'a') as f:
            f.write(w_line)
    sys.stdout.flush()

def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    total_n = 0
    if args.verbose:
        pbar = progressbar.ProgressBar(max_value=n_testdata//test_batch_size)
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        if args.verbose:
            pbar.update(batch_idx)
        total_n = batch_idx + 1
    # Save checkpoint.
    t_acc = 100.*correct/total
    t_loss = test_loss/(total_n)

    w_line = '\nVal:\t{:d}\tLoss: {:.04f}\tAcc: {:.04f}'.format(epoch, t_loss, t_acc)
    print (w_line)
    # save_dir = dir_utils.get_dir('./snapshots/cifar10_vgg_{:s}'.format(args.model))
    result_file = os.path.join(save_dir, 'ckpt_result.txt')
    if not os.path.isfile(result_file):
        with open(result_file, 'w') as f:
            f.write(w_line)
    else:
        with open(result_file, 'a') as f:
            f.write(w_line)
    sys.stdout.flush()

    if t_acc > best_acc:

        state = {
            'net': model.state_dict(),
            'acc': t_acc,
            'loss': t_loss,
            'epoch': epoch,
        }

        torch.save(state,  os.path.join(save_dir,'ckpt.t7'))
        best_acc = t_acc

for epoch in range(start_epoch, start_epoch+args.n_epochs):
    train(epoch)
    test(epoch)

