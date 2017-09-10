import torch
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
from _ext import RRSVM
import numpy as np
from torch.autograd import Variable


class RRSVM_F(torch.autograd.Function):
    def __init__(self, kernel_size=3, padding=0, stride=1, dilation=1):
        super(RRSVM_F, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        # TODO: GPU currently not implemented
        self.GPU_implemented = True

    def forward(self, input, s):

        assert self.kernel_size * self.kernel_size == s.size(1), 'Kernel size should be the same as s size'
        k_s = s.dim()
        if k_s != 2:
            raise RuntimeError('S should be 2D [Channel, kH * kW]')

        k_i = input.dim()
        if k_i != 4:
            raise RuntimeError('Currently RRSVM only supports 2D convolution')

        input = input.contiguous()
        s = s.contiguous()
        #TODO: This is the core function
        output, indices = self._update_output(input, s)
        self.intermediate = (input, s, indices,)
        # self.save_for_backward(input, s)
        self.mark_non_differentiable(indices)
        return output, indices

    def backward(self, grad_output, _indices_grad=None):
        k = grad_output.dim()
        if k != 4:
            raise RuntimeError('Currently RRSVM only supports 2D convolution')

        grad_output = grad_output.contiguous()

        input, s, indices = self.intermediate
        input = input.contiguous()

        grad_input = (self._grad_input(input, s, indices, grad_output) if self.needs_input_grad[0] else None)
        grad_s = (self._grad_params(input, s, indices, grad_output) if self.needs_input_grad[1] else None)
        return grad_input, grad_s

    def _update_output(self, input, s):
        output = input.new(*self._output_size(input, s))
        indices = input.new(*self._indices_size(input, s)).long()
        if not input.is_cuda:
            RRSVM.RRSVM_updateOutput(input, s, output, indices, self.kernel_size, self.kernel_size, self.stride, self.stride,
                                           self.padding, self.padding, self.dilation, self.dilation)
        else:
            RRSVM.RRSVM_updateOutput_cuda(input, s, output, indices, self.kernel_size, self.kernel_size, self.stride, self.stride,
                                           self.padding, self.padding, self.dilation, self.dilation)
            # raise NotImplementedError
        return output, indices

    def _grad_input(self, input, s, indices, grad_output):
        grad_input = input.new(*input.size())
        if not grad_output.is_cuda:
            RRSVM.RRSVM_updateGradInput(s, indices, grad_output, grad_input, input.size(3), input.size(2),
                                              self.kernel_size, self.kernel_size, self.stride, self.stride, self.padding, self.padding,
                                              self.dilation, self.dilation)
        else:
            RRSVM.RRSVM_updateGradInput_cuda(s, indices, grad_output, grad_input, input.size(3), input.size(2),
                                              self.kernel_size, self.kernel_size, self.stride, self.stride, self.padding, self.padding,
                                              self.dilation, self.dilation)
        return grad_input

    def _grad_params(self, input, s, indices, grad_output):
        #TODO: Core function for getting output value here
        grad_s = s.new(*s.size())
        if not grad_output.is_cuda:
            RRSVM.RRSVM_accGradParameters(input, indices, grad_output, grad_s, self.kernel_size, self.kernel_size,
                                                self.stride, self.stride, self.padding, self.padding,
                                                self.dilation, self.dilation)
        else:
            RRSVM.RRSVM_accGradParameters_cuda(input, indices, grad_output, grad_s, self.kernel_size, self.kernel_size,
                                                self.stride, self.stride, self.padding, self.padding,
                                                self.dilation, self.dilation)
        return grad_s

    def _output_size(self, input, s):
        bs = input.size(0)
        assert input.size(1) == s.size(0), "s channels is not the same size as input channels"
        channels = input.size(1)
        output_size = (bs, channels)
        outputHeight = (input.size(2) + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) / self.stride + 1
        output_size += (outputHeight,)

        outputWidth = (input.size(3) + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) / self.stride + 1
        output_size += (outputWidth,)

        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError("RRSVM input is too small (output would be {})".format(
                'x'.join(map(str, output_size))))
        return output_size

    def _indices_size(self, input, s):
        bs = input.size(0)
        assert input.size(1) == s.size(0), "s channels is not the same size as input channels"
        channels = input.size(1)
        indices_size = (bs, channels)
        outputHeight = (input.size(2) + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) / self.stride + 1
        indices_size += (outputHeight,)

        outputWidth = (input.size(3) + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) / self.stride + 1
        indices_size += (outputWidth,)
        indices_size += (self.kernel_size * self.kernel_size,)

        if not all(map(lambda s: s > 0, indices_size)):
            raise ValueError("RRSVM input is too small (output would be {})".format(
                'x'.join(map(str, indices_size))))
        return indices_size


class RRSVM_Module(torch.nn.Module):
    # comapred to convolution:
    # __init__(self, in_channels, out_channels, kernel_size, stride=1,
             # padding=0, dilation=1, groups=1, bias=True):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(RRSVM_Module, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.s = Parameter(torch.Tensor(in_channels, self.kernel_size * self.kernel_size))
        # initialize s:
        n_elt = in_channels * self.kernel_size * self.kernel_size
        init_val = 1. / n_elt
        self.s.data.fill_(init_val)

    def forward(self, input):
        F = RRSVM_F(self.kernel_size, self.padding, self.stride, self.dilation)
        return F(input, self.s)

    def toggleFreeze(self):
        for param in self.parameters():
            param.requires_grad = not param.requires_grad

    def param_loss(self, loss_fn):
        return loss_fn(self.s)

def RRSVM_Loss(net):
    loss = 0
    #TODO: The following types of losses:
    #1. Non-Negative
    #2. L2 norm might not be good, because we want different feature map to have different weights
    #3. Non-incremental
    for id, s_module in enumerate(net.modules()):
        if isinstance(s_module, RRSVM):
          loss += loss_fn(s_module.s)
    return loss


if __name__ == '__main__':
    #TODO: TEST Starting HERE!
    RRSVM_block = RRSVM_Module(in_channels=1, kernel_size=3, stride=1)
    x_data = np.array(np.random.randn(9))
    x_idx = np.argsort(-x_data)
    x = torch.FloatTensor(x_data).resize_([1, 1, 3, 3])
    y = Variable(torch.FloatTensor([1]), requires_grad=False)
    input_x = Variable(x)

    # forward:
    output, _= RRSVM_block(input_x)
    s_output = torch.squeeze(output)
    # backward:
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(s_output, y)
    loss.backward()

    print "DONE"

