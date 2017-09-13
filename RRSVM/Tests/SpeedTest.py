import torchvision.models
import RRSVM.RRSVM as RRSVM
from torch.autograd import Variable
import torch
import numpy as np
import sys
import time


if __name__ == '__main__':
    # test_gradient()
    kernel_size = 2
    n_channel = 200
    feature_size = 224
    batch_size = 40
    input = (Variable(torch.FloatTensor(torch.randn(batch_size, n_channel, feature_size, feature_size)), requires_grad=True),
             Variable(torch.FloatTensor(torch.randn(n_channel, kernel_size**2)), requires_grad=True),)

    F = RRSVM.RRSVM_F(kernel_size, padding=0, stride=feature_size, dilation=1)

    useCuda = False
    if useCuda and  torch.cuda.is_available():
        input = [i.cuda() for i in input]

    start = time.time()
    analytical, analytical_indices = F(*input)
    end = time.time()
    print "CPU:{:0.10f}".format(end - start)


    useCuda = True
    if useCuda and  torch.cuda.is_available():
        input = [i.cuda() for i in input]

    start = time.time()
    analytical, analytical_indices = F(*input)
    end = time.time()
    print "GPU:{:0.10f}".format(end - start)

