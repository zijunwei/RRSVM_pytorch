import torchvision.models
import RRSVM.RRSVM.RRSVM
from torch.autograd import Variable
import torch
# from torch.autograd import gradcheck
from RRSVM.RRSVM.MyGradCheck import gradcheck
import numpy as np
import sys


def test_gradient(input, kernel_size=3, padding=0, stride=1):

    F = RRSVM.RRSVM.RRSVM_F(kernel_size=kernel_size, padding=padding, stride=stride, dilation=1)

    test = gradcheck(lambda i, s: F(i, s), inputs=input, eps=1e-3, atol=1e-3, rtol=1e-3)
    if test == True:
        print("Gradient Check Passed!")
    else:
        print("Gradient Check Failed!")


def test_forward(input, kernel_size=3, padding=1, stride=2, dilation=1):
    # input = (Variable(torch.FloatTensor(torch.randn(1, 1, 5, 5)), requires_grad=True),
    #          Variable(torch.FloatTensor(torch.randn(1, 9)), requires_grad=True),)

    F = RRSVM.RRSVM.RRSVM_F(kernel_size, padding, stride, dilation=1)

    numerical, numerical_indices = get_numerical_output(*input, kernel_size=kernel_size, padding=padding, stride=stride, dilation=1)

    if torch.cuda.is_available():
        input = [i.cuda() for i in input]
    else:
        print("Cuda device not detected on this device")
        sys.exit(-1)
    analytical, analytical_indices = F(*input)
    analytical = analytical.cpu().data.numpy()
    analytical_indices = analytical_indices.cpu().data.numpy()

    atol = 1e-5
    rtol = 1e-3


    if not (np.absolute(numerical - analytical) <= (atol + rtol * np.absolute(numerical))).all():
        print "Output Failed Foward Test"
    else:
        print "Ouput Pass Foward Test"

    relative_loss = (numerical - analytical) / (numerical + 1e-6)
    print "Max Diff: {:.04f}".format((np.abs(relative_loss).max()))


    if not (np.absolute(numerical_indices - analytical_indices) <= (atol + rtol * np.absolute(numerical_indices))).all():
        print "Indices Failed Foward Test"
    else:
        print "Indices Pass Foward Test"
    # test = gradcheck(lambda i, s: F(i, s), inputs=input, eps=1e-6, atol=1e-4)
    # print "DONE"


def get_numerical_output(input, s, kernel_size=3, padding=0, stride=1, dilation=1):

    if isinstance(input, (Variable, torch.Tensor)):
        input = input.data.numpy()
        s = s.data.numpy()

    input_n = input.shape[0]
    input_d = input.shape[1]
    input_h = input.shape[2]
    input_w = input.shape[3]

    output_h = (input_h+ 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1
    output_w = (input_w+ 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1

    output_size = [input_n, input_d, output_h, output_w]
    output = np.zeros(output_size, dtype=np.float32)
    output_indices_size = [input_n, input_d, output_h, output_w, kernel_size*kernel_size]
    output_indices = np.zeros(output_indices_size, dtype=np.long)

    for idx_n in range(input_n):
        n_input = input[idx_n]
        for idx_d in range(input_d):
            d_s = s[idx_d]
            # d_s_flat = d_s.flatten()
            n_d_input = n_input[idx_d]
            n_d_input = pad2d(n_d_input, padding)
            for idx_h in range(output_h):
                for idx_w in range(output_w):
                    elements = np.zeros([kernel_size, kernel_size])
                    # indices = np.zeros(kernel_size*kernel_size)

                    for i in range(kernel_size):
                        for j in range(kernel_size):
                            elements[i, j] = n_d_input[idx_h*stride+i, idx_w*stride+j]

                    elements = elements.flatten()
                    sorted_elements = -np.sort(-elements)
                    sorted_indices = np.argsort(-elements)
                    output_indices[idx_n, idx_d, idx_h, idx_w] = sorted_indices
                    output[idx_n, idx_d, idx_h, idx_w] = np.sum(np.multiply(sorted_elements, d_s))

    return output, output_indices


def pad2d(array2d, padding):
    if padding == 0:
        return array2d

    array_shape = array2d.shape
    if len(array_shape) != 2 :
        raise NotImplementedError('Only Supports 2D padding')
    new_array = np.zeros([array_shape[0]+ 2*padding, array_shape[1] + 2* padding])
    new_array_shape = new_array.shape
    new_array[padding:new_array_shape[0]-padding, padding:new_array_shape[1]-padding] = array2d
    return new_array

if __name__ == '__main__':
    # test_gradient()
    kernel_size = 2
    n_channel = 10
    feature_size = 20
    input = (Variable(torch.FloatTensor(torch.randn(1, n_channel, feature_size, feature_size)), requires_grad=True),
             Variable(torch.FloatTensor(torch.randn(n_channel, kernel_size**2)), requires_grad=True),)

    test_forward(input, kernel_size=kernel_size, padding=0, stride=kernel_size, dilation=1)
    test_gradient(input, kernel_size=kernel_size, padding=0, stride=kernel_size)