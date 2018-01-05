import torchvision.models
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import gradcheck
from RRSVM.Tests.MyGradCheck import gradcheck
import numpy as np

# TODO: May be you need the S to be 2D ...
# TODO: Think about padding case with zero


def test_gradient(input, kernel_size=3, padding=0, stride=1):

    functional = nn.MaxPool2d(kernel_size=kernel_size, padding=padding, stride=stride, dilation=1)
    test = gradcheck(lambda i: functional(i), inputs=input, eps=1e-3, atol=1e-3, rtol=1e-3)
    if test == True:
        print("Passed. Gradient Check Passed!")
    else:
        print("Failed. Gradient Check Failed!")

# use random inputs
def test1(case_id):
    if case_id == 1:
        n_im = 1
        kernel_size = 2
        n_channel = 1
        feature_size = 2
    elif case_id == 2:
        n_im = 2
        kernel_size = 2
        n_channel = 3
        feature_size = 28
    elif case_id == 3:
        n_im = 1
        kernel_size = 5
        n_channel = 10
        feature_size = 14
    elif case_id == 4:
        n_im = 4
        kernel_size = 3
        n_channel = 2
        feature_size = 10
    elif case_id == 5:
        n_im = 8
        kernel_size = 2
        n_channel = 3
        feature_size = 14

    input = (Variable(torch.FloatTensor(torch.randn(n_im, n_channel, feature_size, feature_size)), requires_grad=True),)
    test_gradient(input, kernel_size=kernel_size, padding=0, stride=kernel_size)


# use random non-negative inputs
def test2(case_id):
    if case_id == 1:
        kernel_size = 2
        n_channel = 1
        feature_size = 2
    elif case_id == 2:
        kernel_size = 2
        n_channel = 3
        feature_size = 7
    elif case_id == 3:
        kernel_size = 5
        n_channel = 10
        feature_size = 14

    A = torch.randn(1, n_channel, feature_size, feature_size)
    A[A < 0] = 0.0
    input = (Variable(torch.FloatTensor(A), requires_grad=True),)
    test_gradient(input, kernel_size=kernel_size, padding=0, stride=kernel_size)


if __name__ == '__main__':
    for ii in range(5):
        print("---- Test 1, Case {}".format(ii+1))
        test1(ii+1)

    for ii in range(3):
        print("---- Test 2, Case {}".format(ii+1))
        test2(ii+1)
