'''LeNet in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import RRSVM.RRSVM as RRSVM
from torch.autograd import Variable

modes ={'max': nn.MaxPool2d(kernel_size=2, stride=2), 'avg': nn.AvgPool2d(kernel_size=2, stride=2)}


class LeNet_Base(nn.Module):
    def __init__(self, mode='max'):

        super(LeNet_Base, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = modes[mode]
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = modes[mode]
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class LeNet_RRSVM(nn.Module):
    def __init__(self):
        super(LeNet_RRSVM, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = RRSVM.RRSVM_Module(6, 2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = RRSVM.RRSVM_Module(16, 2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out, _ = self.pool1(out)
        out = F.relu(self.conv2(out))
        out, _ = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

if __name__ == '__main__':
    image = Variable(torch.FloatTensor(torch.randn(1, 3, 32, 32)))
    RRSVM_LeNet = LeNet_RRSVM()
    RRSVM_output = RRSVM_LeNet(image)

    Max_LeNet = LeNet_Base('max')
    Max_output = Max_LeNet(image)


    print "DONE"