import torch
from torch.autograd import Variable
x = Variable(torch.ones(2, 2), requires_grad = True)
y = x ** 2
y.backward(torch.ones(2, 2), retain_graph=True)
print "first backward of x is:"
print x.grad
y.backward(2*torch.ones(2, 2), retain_graph=False)
print "second backward of x is:"
print x.grad