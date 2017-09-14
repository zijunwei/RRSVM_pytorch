import torchvision.models
import RRSVM.RRSVM as RRSVM
from torch.autograd import Variable
import torch
import numpy as np
import sys
import time
import torch.nn.functional as F


if __name__ == '__main__':

    # TODO: a list of observations:
    # After optimization, the GPU version is 7 times faster, but it seems that the operation is faster than Max-pooling
    #

    kernel_size = 2
    n_channel = 100
    feature_size = 224
    batch_size = 40
    input = (Variable(torch.FloatTensor(torch.randn(batch_size, n_channel, feature_size, feature_size)), requires_grad=True),
             Variable(torch.FloatTensor(torch.randn(n_channel, kernel_size**2)), requires_grad=True),)

    RRSVM_f = RRSVM.RRSVM_F(kernel_size, padding=0, stride=feature_size, dilation=1)

    useCuda = False
    if useCuda and  torch.cuda.is_available():
        input = [i.cuda() for i in input]

    start = time.time()
    analytical, analytical_indices = RRSVM_f(*input)
    end = time.time()
    print "RRSVM CPU:{:0.10f}".format(end - start)


    useCuda = True

    input_cuda = (Variable(torch.FloatTensor(torch.randn(batch_size, n_channel, feature_size, feature_size)).cuda(), requires_grad=True),
             Variable(torch.FloatTensor(torch.randn(n_channel, kernel_size**2)).cuda(), requires_grad=True),)
    start = time.time()
    analytical, analytical_indices = RRSVM_f(*input_cuda)
    end = time.time()
    print "RRSVM GPU:{:0.10f}".format(end - start)

    print "MaxPooling Test"

    kernel_size = 2
    n_channel = 100
    feature_size = 224
    batch_size = 40
    input = (Variable(torch.FloatTensor(torch.randn(1, n_channel, feature_size, feature_size)), requires_grad=True), kernel_size)
    f_max_pooling = F.max_pool2d
    start = time.time()
    analytical = f_max_pooling(*input)
    end = time.time()
    print "Max Pooling CPU:{:0.10f}".format(end - start)


    kernel_size = 2
    n_channel = 100
    feature_size = 224
    batch_size = 40
    input = Variable(torch.FloatTensor(torch.randn(1, n_channel, feature_size, feature_size)).cuda(), requires_grad=True)
    f_max_pooling = F.max_pool2d
    start = time.time()
    analytical = f_max_pooling(input, 2)
    end = time.time()
    print "Max Pooling GPU:{:0.10f}".format(end - start)