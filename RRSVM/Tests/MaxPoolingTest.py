# This is a test among maxpooling, convolution and RRSVM

from torch.autograd import Variable
import torch
from torch.autograd import gradcheck
import numpy as np
import torch
import torch.nn.functional as F


if __name__ == '__main__':
    kernel_size = 2
    n_channel = 100
    feature_size = 6
    batch_size = 3
    input = (Variable(torch.DoubleTensor(torch.randn(1, n_channel, feature_size, feature_size).double()), requires_grad=True), kernel_size)
    f_max_pooling = F.max_pool2d
    print gradcheck(f_max_pooling, input, eps=1e-3)