import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.nn import Parameter
import math


class RRSVM_Simple(nn.Module):
    def __init__(self, in_channels, gridsize, final=True, bias=True):
        super(RRSVM_Simple, self).__init__()
        self.in_channels = in_channels
        self.final = final
        self.w = Parameter(torch.Tensor(
                1, in_channels, 1, 1))

        self.s = Parameter(torch.Tensor(1, gridsize, gridsize))

        if bias:
            self.bias = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('bias', None)

        # initialization:
        n_s = gridsize * gridsize
        self.s.data.fill_(1. / n_s)
        self.w.data.normal_(0, math.sqrt(2. / self.in_channels))
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        bs, c, h, w = input.size()
        score = F.conv2d(input, self.w, self.bias).view(bs, -1)
        _, w_indices = torch.sort(score, descending=True)
        s = self.s.repeat(bs, 1, 1).view(bs, -1)
        sorted_s = torch.gather(s, dim=1, index=w_indices)
        sorted_s = sorted_s.view(bs, 1, h, w)
        s = sorted_s.expand_as(input)

        # output = F.conv2d(torch.mul(s, input), self.w, self.bias).view(bs, -1)

        output = torch.mul(s, input)

        # output = torch.sum(output, dim=1)

        return output


def RRSVM_L1Loss(net, loss_fn):
    loss = 0
    for id, s_module in enumerate(net.modules()):
        if isinstance(s_module, RRSVM_Simple):
          loss += loss_fn(s_module.s)
    return loss


if __name__ == '__main__':
    grid_size = 5
    rrsvm = RRSVM_Simple(10, gridsize=grid_size, bias=False)
    for i in range(10):
        input_tensor = torch.FloatTensor(torch.randn([3, 10, grid_size, grid_size]))

        input_v = Variable(input_tensor, requires_grad=True)

        output = rrsvm(input_v)