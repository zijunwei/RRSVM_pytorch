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


target_dict_file = '/home/zwei/Dev/RRSVM_pytorch/snapshots/cifar100_ResVgg_0Init/ckpt.t7'
checkpoint = torch.load(target_dict_file, map_location=lambda storage, loc: storage)
state_dict = checkpoint['net']
s_s = []
for name, param in state_dict.items():
    if name[-2:] == '.s':
        s_s.append(param)
        print ("{:s}".format(name))
        print(param)

print("Check S!")