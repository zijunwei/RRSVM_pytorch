import torchvision.models
import RRSVM.RRSVM as RRSVM
from torch.autograd import Variable
import torch
# from torch.autograd import gradcheck
from RRSVM.Tests.MyGradCheck import gradcheck
import numpy as np

# TODO: May be you need the S to be 2D ...
# TODO: Think about padding case with zero


def test_gradient(input, kernel_size=3, padding=0, stride=1):

    F = RRSVM.RRSVM_F(kernel_size=kernel_size, padding=padding, stride=stride, dilation=1)

    test = gradcheck(lambda i, s: F(i, s), inputs=input, eps=1e-3, atol=1e-3, rtol=1e-3)
    if test == True:
        print("Passed. Gradient Check Passed!")
    else:
        print("Failed. Gradient Check Failed!")

#zwei: forward indices check may fail due to many entries of the same value occure in the receptive field. Now change
# the forwared indices check to sort and compare the values.
def check_forward_indices(input, numerical, analytical, kernel_size, padding, stride, dilation):
    if padding >0 or dilation>1:
        print("Foward indices check only supports padding=0 and dilation=1")
    input_shape = input.shape
    output_size_H =(input_shape[2] + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1
    output_size_W =(input_shape[3] + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1
    # Should be a list of asseration here, but I will go without it for now
    for batch_idx in range(input_shape[0]):
        for chl_idx in range(input_shape[1]):

            input_slice = input[batch_idx, chl_idx]
            for h_idx in range(output_size_H):
                for w_idx in range(output_size_W):
                    input_region = input_slice[h_idx*stride:h_idx*stride+kernel_size, w_idx*stride:w_idx*stride+kernel_size]
                    input_array = input_region.flatten()
                    s_analytical_indices = analytical[batch_idx, chl_idx, h_idx, w_idx]
                    s_numberical_indices = numerical[batch_idx, chl_idx, h_idx, w_idx]
                    s_analytical_arranges = input_array[s_analytical_indices]
                    s_numberical_arranges = input_array[s_numberical_indices]
                    if not (s_numberical_arranges == s_analytical_arranges).all():
                        return False

    return True




def test_forward(input, kernel_size=3, padding=0, stride=2, dilation=1):
    F = RRSVM.RRSVM_F(kernel_size, padding, stride, dilation=dilation, return_indices=True)
    analytical, analytical_indices = F(*input)
    analytical = analytical.data.cpu().numpy()
    analytical_indices = analytical_indices.data.cpu().numpy()
    numerical, numerical_indices = get_numerical_output(*input, kernel_size=kernel_size, padding=padding, stride=stride, dilation=1)

    atol = 1e-5
    rtol = 1e-3
    if not (np.absolute(numerical - analytical) <= (atol + rtol * np.absolute(numerical))).all():
        print "Failed. Output Failed Foward Test"
    else:
        print "Passed. Ouput Pass Foward Test"

    #FIXME Minh: Seems like a bug. This code does not test the indices
    # if not (np.absolute(numerical - analytical) <= (atol + rtol * np.absolute(numerical))).all():
    if not (numerical_indices == analytical_indices).all():
        print "Failed. Indices Failed Foward Test"
    else:
        print "Passed. Indices Pass Foward Test"


    # FIXME: Using Zwei's way
    # input_np = input[0].data.cpu().numpy()
    # if not check_forward_indices(input_np, numerical_indices, analytical_indices, kernel_size, padding, stride, dilation):
    #     print "Failed. Indices Failed Foward Test"
    # else:
    #     print "Passed. Indices Pass Foward Test"

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

    input = (Variable(torch.FloatTensor(torch.randn(n_im, n_channel, feature_size, feature_size)), requires_grad=True),
             Variable(torch.FloatTensor(torch.randn(n_channel, kernel_size ** 2)), requires_grad=True),)
    test_forward(input, kernel_size=kernel_size, padding=0, stride=kernel_size, dilation=1)
    test_gradient(input, kernel_size=kernel_size, padding=0, stride=kernel_size)


# use random non-negative inputs
def test2(case_id):
    if case_id == 1:
        kernel_size = 2
        n_channel = 1
        feature_size = 2
        stride = kernel_size
    elif case_id == 2:
        kernel_size = 2
        n_channel = 3
        feature_size = 7
        stride = kernel_size
    elif case_id == 3:
        kernel_size = 5
        n_channel = 10
        feature_size = 14
        stride = kernel_size
    elif case_id == 4:
        kernel_size = 7
        n_channel = 50
        feature_size = 10
        stride = 5

    # A = torch.randn(1, n_channel, feature_size, feature_size)
    # A[A < 0] = 0.0
    A = torch.randperm(1 * n_channel * feature_size * feature_size).float()
    A = A.view(1, n_channel, feature_size, feature_size)

    input = (Variable(torch.FloatTensor(A), requires_grad=True),
             Variable(torch.FloatTensor(torch.randn(n_channel, kernel_size ** 2)), requires_grad=True),)
    test_forward(input, kernel_size=kernel_size, padding=0, stride=stride, dilation=1)
    test_gradient(input, kernel_size=kernel_size, padding=0, stride=kernel_size)


if __name__ == '__main__':
    for ii in range(5):
        print("---- Test 1, Case {}".format(ii+1))
        test1(ii+1)
    for ii in range(4):
        print("---- Test 2, Case {}".format(ii+1))
        test2(ii+1)
