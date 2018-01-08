import torchvision.models
import torch.nn.functional as Functional
import torch.nn as nn
from torch.autograd import Variable
import torch
# from torch.autograd import gradcheck
from RRSVM.Tests.MyGradCheck import gradcheck
import numpy as np

# TODO: May be you need the S to be 2D ...
# TODO: Think about padding case with zero


def test_gradient(input, channels=3, kernel_size=3, padding=0, stride=1):

    F = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=1)

    test = gradcheck(lambda i, s: F(i), inputs=input, eps=1e-3, atol=1e-3, rtol=1e-3)
    if test == True:
        print("Passed. Gradient Check Passed!")
    else:
        print("Failed. Gradient Check Failed!")




# zwei: set up a new set of tests to test each situation specifically
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
        kernel_size = 7
        n_channel = 50
        feature_size = 10
        stride = 5

    # A = torch.randn(1, n_channel, feature_size, feature_size)
    # A[A < 0] = 0.0
    A = torch.randperm(n_im * n_channel * feature_size * feature_size).float()
    A = A.view(n_im, n_channel, feature_size, feature_size)

    input = (Variable(torch.FloatTensor(A), requires_grad=True),
             Variable(torch.FloatTensor(torch.randn(n_channel, n_channel, kernel_size, kernel_size)), requires_grad=True))
    # test_forward(input, kernel_size=kernel_size, padding=0, stride=stride, dilation=1)
    test_gradient(input, kernel_size=kernel_size, padding=0, stride=kernel_size)


if __name__ == '__main__':
    # for ii in range(5):
    #     print("---- Test 1, Case {}".format(ii+1))
    #     test1(ii+1)
    for ii in range(5):
        print("---- Test 3, Case {}".format(ii+1))
        test3(ii)
