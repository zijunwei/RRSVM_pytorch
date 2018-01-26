import torch
import numpy as np
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.optim as optim

class SortNet(torch.nn.Module):
    def __init__(self, dim=5, return_indices=False):
        super(SortNet, self).__init__()
        self.w = Parameter(torch.FloatTensor(torch.randperm(dim).float()))
        self.return_indices = return_indices
    def forward(self, x):

        sorted_x, indice_x = torch.sort(x, descending=True)
        # x = torch.mm(sorted_x, self.w)
        x = torch.dot(sorted_x, self.w)
        if self.return_indices:
            return x, indice_x
        else:
            return x


x = torch.FloatTensor(torch.randperm(5).float())
sortNet = SortNet()
optimizer = optim.SGD(sortNet.parameters(), lr=0.1)
x = Variable(x, requires_grad=True)
optimizer.zero_grad()
y = sortNet(x)
gradient = y.data.new(*y.size()).fill_(2.0)
y.backward(gradient)
optimizer.step()

print 'DEBEU'