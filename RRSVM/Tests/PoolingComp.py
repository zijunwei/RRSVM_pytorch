# compares maxpooling, average pooling and

import RRSVM.RRSVM as RRSVM
from torch.autograd import Variable
import torch.nn.functional as Funtional
import torch
from RRSVM.Tests.MyGradCheck import gradcheck

global stride
global padding
global dilation

n_im = 1
kernel_size = 2
n_channel = 1
feature_size =4
stride = kernel_size
padding = 0
dilation = 1

A = torch.randperm(n_im*n_channel*feature_size*feature_size).float()
A = A.view(n_im, n_channel, feature_size, feature_size)

weight = torch.randn(n_channel,kernel_size**2)

RRSVM_input = (Variable(torch.FloatTensor(A), requires_grad=True),
               Variable(torch.FloatTensor(weight), requires_grad=True),)

Max_input = Variable(torch.FloatTensor(A), requires_grad=True)
Avg_Input = Variable(torch.FloatTensor(A), requires_grad=True)


RRSVM_F = RRSVM.RRSVM_F(kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, return_indices=True)

def Max_F(input):
    return Funtional.max_pool2d(input, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, return_indices=True)


def Avg_F(input):
    return Funtional.avg_pool2d(input, kernel_size=kernel_size, padding=padding, stride=stride)


RRSVM_out, RRSVM_indices = RRSVM_F(*RRSVM_input)
RRSVM_out.backward(torch.ones(RRSVM_out.size()))

Max_out, Max_indices = Max_F(Max_input)
Max_out.backward(torch.ones(Max_out.size()))

Avg_out = Avg_F(Avg_Input)
Avg_out.backward(torch.ones(Avg_out.size()))

grad_RRSVM = RRSVM_input[0].grad
grad_Max = Max_input.grad
grad_Avg = Avg_Input.grad


print "DEBUG"





