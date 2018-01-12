# 1. Check the gradient type of Convolutional NN
# 2. See if the testgradient works fine
import torch.nn.functional as Functional
from torch.autograd import Variable
import torch
from torch.autograd import gradcheck
import numpy as np


def test_gradient(input, padding=0, stride=1, dilation=1):
    def conv_f(input, weight):
        return Functional.conv2d(input=input, weight=weight, bias=None, padding=padding, stride=stride, dilation=dilation)

    test = gradcheck(lambda i, s: conv_f(i,s), inputs=input, eps=1e-3, atol=1e-3, rtol=1e-3)
    if test == True:
        print("Passed. Gradient Check Passed!")
    else:
        print("Failed. Gradient Check Failed!")


def test3(case_id):
    # test if the sum_gradient is all correct
    if case_id == 0:
        n_im = 1
        kernel_size = 2
        n_channel = 1
        feature_size = 4
        stride = kernel_size

    # test if the sum_gradient of the parameter is correct over different images
    elif case_id == 1:
        n_im = 2
        kernel_size = 2
        n_channel = 1
        feature_size = 4
        stride = kernel_size

    # test larger scale
    elif case_id == 2:
        n_im = 2
        kernel_size = 2
        n_channel = 3
        feature_size = 4
        stride = kernel_size
    elif case_id == 3:
        n_im = 3
        kernel_size = 5
        n_channel = 10
        feature_size = 14
        stride = kernel_size
    elif case_id == 4:
        n_im = 5
        kernel_size = 5
        n_channel = 20
        feature_size = 10
        stride = kernel_size

    # if you change the double back to float, it sometimes gives errors
    A = torch.randperm(n_im * n_channel * feature_size * feature_size).double()
    A = A.view(n_im, n_channel, feature_size, feature_size)

    input = (Variable(torch.DoubleTensor(A), requires_grad=True),
             Variable(torch.DoubleTensor(torch.randn(n_channel, n_channel, kernel_size, kernel_size).double()), requires_grad=True))
    test_gradient(input, padding=0, stride=stride)


if __name__ == '__main__':

    for ii in range(5):
        print("---- Test 3, Case {}".format(ii+1))
        test3(ii)
