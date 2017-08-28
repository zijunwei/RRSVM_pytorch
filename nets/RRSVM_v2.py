import torch
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
from _ext import RRSVM

class RRSVM_F_v2(torch.autograd.Function):
    def __init__(self, stride, padding):
        self.stride = stride
        self.padding = padding
        self.groups = 1

    def forward(self, input, s):
        k = input.dim()
        if k != 4:
            raise RuntimeError('Currently RRSVM only supports 2D convolution')
        input = input.contiguous()
        #TODO: This is the core function
        output, indices = self._update_output(input, s)
        self.save_for_backward(input, s, indices)
        self.mark_non_differentiable(indices)
        return output, indices

    def backward(self, grad_output, _indices_grad=None):
        k = grad_output.dim()
        if k != 4:
            raise RuntimeError('Currently RRSVM only supports 2D convolution')

        grad_output = grad_output.contiguous()

        input, s, indices = self.saved_tensors
        input = input.contiguous()

        grad_input = (self._grad_input(input, s, indices, grad_output) if self.needs_input_grad[0] else None)
        grad_s = (self._grad_params(input, s, indices, grad_output) if self.needs_input_grad[0] else None)
        return grad_input, grad_s

    def _update_output(self, input, s):
        #TODO: Core function for getting output value here
        output = input.new(*self._output_size(input, s))
        indices = input.new().long()
        if not input.is_cuda():
            RRSVM.c_forward(output, input, s, indices)
        else:
            RRSVM.c_forward_cuda(output, input, s, indices)
        return output

    def _grad_input(self, input, s, indices, grad_output):
        grad_input = input.new()
        if grad_output.is_cuda():
            #TODO cuda implementation
            pass
        else:
            raise NotImplementedError
        return grad_input


    def _grad_params(self, input, s, indices, grad_output):
        #TODO: Core function for getting output value here
        grad_s = s.new()
        if grad_output.is_cuda():
            pass
        else:
            raise  NotImplementedError
        return grad_s

    def _output_size(self, input, s):
        bs = input.size(0)
        assert input.size(1) == s.size(0), "s channels is not the same size as input channels"
        channels = input.size(1)
        output_size = (bs, channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = self.padding[d]
            kernel = s.size(d + 1)
            stride = self.stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride + 1,)
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError("RRSVM input is too small (output would be {})".format(
                'x'.join(map(str, output_size))))
        return output_size


class RRSVM_v2_M(torch.nn.Module):
    # comapred to convolution:
    # __init__(self, in_channels, out_channels, kernel_size, stride=1,
             # padding=0, dilation=1, groups=1, bias=True):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0):
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.s = Parameter(torch.Tensor(in_channels, *self.kernel_size))

    def forward(self, input):
        F = RRSVM_F_v2(self.stride, self.padding)
        return F(input, self.s)

