import torchvision.models
import RRSVM
from torch.autograd import Variable
import torch
# from torch.autograd import gradcheck
from MyGradCheck import gradcheck
import numpy as np

# TODO: May be you need the S to be 2D ...
# TODO: Think about padding case with zero

def gradTest():
    input = (Variable(torch.FloatTensor(torch.randn(1, 1, 3, 3)), requires_grad=True),
             Variable(torch.FloatTensor(torch.randn(1, 3, 3)), requires_grad=True),)

    F = RRSVM.RRSVM_F(kernel_size=3, padding=0, stride=1, dilation=1)

    test = gradcheck(lambda i, s: F(i, s), inputs=input, eps=1e-6, atol=1e-4)
    print test

def TestOutput():
    input = (Variable(torch.FloatTensor(torch.randn(1, 1, 3, 3)), requires_grad=True),
             Variable(torch.FloatTensor(torch.randn(1, 3, 3)), requires_grad=True),)

    F = RRSVM.RRSVM_F(kernel_size=3, padding=0, stride=1, dilation=1)
    analytical, _ = F(*input)
    numerical, _ = get_numerical_output(*input, kernel_size=3, padding=0, stride=1, dilation=1)
    # test = gradcheck(lambda i, s: F(i, s), inputs=input, eps=1e-6, atol=1e-4)
    print "DONE"

def get_numerical_output(input, s, kernel_size=3, padding=0, stride=1, dilation=1):
    input = input.data.numpy()
    s = s.data.numpy()

    input_n = input.shape[0]
    input_d = input.shape[1]
    input_h = input.shape[2]
    input_w = input.shape[3]

    output_h = (input_h+ 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1
    output_w = (input_w+ 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1


    output_size = [input_n, input_d, output_h, output_w]
    output = np.zeros(output_size)
    output_indices_size = [input_n, input_d, output_h, output_w, kernel_size*kernel_size]
    output_indices = np.zeros(output_indices_size)

    for idx_n in range(input_n):
        n_input = input[idx_n]
        for idx_d in range(input_d):
            d_s = s[idx_d]
            d_s_flat = d_s.flatten()
            n_d_input = n_input[idx_d]
            for idx_h in range(output_h):
                for idx_w in range(output_w):
                    elements = np.zeros([kernel_size, kernel_size])
                    # indices = np.zeros(kernel_size*kernel_size)
                    for i in range(kernel_size):
                        for j in range(kernel_size):
                            elements[i, j] = n_d_input[idx_h+i, idx_w+j]

                    elements = elements.flatten()
                    sorted_elements = -np.sort(-elements)
                    sorted_indices = np.argsort(-elements)
                    output_indices[idx_n, idx_d, idx_h, idx_w] = sorted_indices
                    output[idx_n, idx_d, idx_h, idx_w] = np.sum(np.multiply(sorted_elements, d_s_flat))
                    #FIXMe: Not finished yet




    print "DB"


if __name__ == '__main__':
    gradTest()
    # TestOutput()
