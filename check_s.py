from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os, sys
import argparse

# from models import *
import progressbar
from torch.autograd import Variable
from models.cifar import vgg
from py_utils import dir_utils
from pt_utils import t_sets
import RRSVM.RRSVM as RRSVM
parser = argparse.ArgumentParser(description='PyTorch CIFAR Training on VGG')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--dataset', default='cifar100', type=str, help='dataset = [cifar/cifar100]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default='VGG16_F_O', help='Type of model')
parser.add_argument("--gpu_id", default=None, type=int)
parser.add_argument('--positive_constraint', '-p', action='store_true', help='positivity constraint')
args = parser.parse_args()
use_cuda = torch.cuda.is_available() and args.gpu_id is not None


if args.dataset.lower() == 'cifar10':
    print("CIFAR10")
    n_classes = 10
elif args.dataset.lower() == 'cifar100':
    print("CIFAR100")
    n_classes = 100
else:
    print('Dataset not recognized')
    raise NameError



# Model
print ("Model:{:s}".format(args.model))

save_dir = './snapshots/{:s}_vgg_{:s}'.format(args.dataset.lower(), args.model.upper())
if args.positive_constraint:
    save_dir = './snapshots/{:s}_vgg_{:s}_p'.format(args.dataset.lower(), args.model.upper())
save_dir = dir_utils.get_dir(save_dir)
print('load from {:s}'.format(save_dir))
p_constraint = False
if args.positive_constraint:
    p_constraint = True
    positive_clipper = RRSVM.RRSVM_PositiveClipper()


net = vgg.VGG(args.model.upper(), n_classes=n_classes)

if use_cuda:
    torch.cuda.set_device(args.gpu_id)
    net.cuda()

    # cudnn.enabled = False



if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(os.path.join(save_dir, 'ckpt.t7')), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(save_dir, 'ckpt.t7'), map_location=lambda storage, loc: storage)
    state_dict = checkpoint['net']
    net.load_state_dict(state_dict)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print("Best ACC: {:.04f}".format(best_acc))
else:
    raise NotImplemented



s_values = []
for s_module in net.modules():
    if isinstance(s_module, (RRSVM.RRSVM_Module)):
        s_values.append(s_module.s.cpu().data)



print("Check S!")