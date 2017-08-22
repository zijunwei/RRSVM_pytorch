import torch.nn as nn
import torch.nn.functional as F
from RRSVM import RRSVM
from pytorch_utils.t_sets import getOutputSize
import torch
from torch.autograd import Variable

class BaseNet(nn.Module):
    def __init__(self, input_size=None):
        super(BaseNet, self).__init__()
        if not input_size:
            input_size = torch.Size([1, 3, 32, 32])

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv1_outputsize = getOutputSize(input_size, self.conv1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool1_outputsize = getOutputSize(self.conv1_outputsize, self.pool1)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2_outputsize = getOutputSize(self.pool1_outputsize, self.conv2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool2_outputsize = getOutputSize(self.conv2_outputsize, self.pool2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc1_outputsize = getOutputSize(self.conv2_outputsize.view(), self.fc1)

        self.fc2 = nn.Linear(120, 84)
        # self.fc2_outputsize = getOutputSize(self.fc1_outputsize, self.fc2)

        self.fc3 = nn.Linear(84, 10)
        # self.fc3_outputsize = getOutputSize(self.fc2_outputsize, self.fc3)

        print "InitDone"

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RRSVMNetV1(nn.Module):
    def __init__(self, input_size=None):
        super(RRSVMNetV1, self).__init__()
        if not input_size:
            input_size = torch.Size([1, 3, 32, 32])

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv1_outputsize = getOutputSize(input_size, self.conv1)
        self.conv1_RRSVM = RRSVM(6, 28)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool1_outputsize = getOutputSize(self.conv1_outputsize, self.pool1)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2_outputsize = getOutputSize(self.pool1_outputsize, self.conv2)
        self.conv2_RRSVM = RRSVM(16, 10)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool2_outputsize = getOutputSize(self.conv2_outputsize, self.pool2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc1_outputsize = getOutputSize(self.conv2_outputsize.view(), self.fc1)

        self.fc2 = nn.Linear(120, 84)
        # self.fc2_outputsize = getOutputSize(self.fc1_outputsize, self.fc2)

        self.fc3 = nn.Linear(84, 10)
        # self.fc3_outputsize = getOutputSize(self.fc2_outputsize, self.fc3)

        print "InitDone"

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1_RRSVM(self.conv1(x))))
        x = self.pool2(F.relu(self.conv2_RRSVM(self.conv2(x))))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# class RRSVMNet(nn.Module):
#     def __init__(self, input_size=None):
#         super(RRSVMNet, self).__init__()
#         if not input_size:
#             input_size = torch.Size([1, 3, 32, 32])
#
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.conv1_outputsize = getOutputSize(input_size, self.conv1)
#         self.conv1_rrsvm = RRSVM(in_channels=6, gridsize=28)
#
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.pool1_outputsize = getOutputSize(self.conv1_outputsize, self.pool1)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.conv2_outputsize = getOutputSize(self.pool1_outputsize, self.conv2)
#
#         self.conv3 = nn.Conv2d(16, 32, 3)
#         self.conv3_outputsize = getOutputSize(self.conv2_outputsize, self.conv3)
#
#         self.conv4 = nn.Conv2d(32, 64, 3)
#         self.conv4_outputsize = getOutputSize(self.conv3_outputsize, self.conv4)
#
#         self.RRSVM = RRSVM(in_channels=64, gridsize=6)
#
#
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         # self.fc1_outputsize = getOutputSize(self.conv2_outputsize.view(), self.fc1)
#
#         self.fc2 = nn.Linear(120, 84)
#         # self.fc2_outputsize = getOutputSize(self.fc1_outputsize, self.fc2)
#
#         self.fc3 = nn.Linear(84, 10)
#         # self.fc3_outputsize = getOutputSize(self.fc2_outputsize, self.fc3)
#
#         print "InitDone"
#
#     def forward(self, x):
#         x = self.pool1(F.relu(self.conv1(x)))
#         x = self.pool2(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


if __name__ == '__main__':
    baseNet = RRSVMNetV1()

    for i in range(10):
        input_tensor = torch.FloatTensor(torch.randn([3, 3, 32, 32]))

        input_v = Variable(input_tensor, requires_grad=True)

        output = baseNet(input_v)
        print "DONE"

