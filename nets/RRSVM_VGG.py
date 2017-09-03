import torch
import torch.nn as nn
from torch.autograd import Variable

import VGGLayered_base
from RRSVM.RRSVM_v1 import RRSVM
from pytorch_utils.t_sets import getOutputSize


class RRSVMNet(nn.Module):
    def __init__(self, pretrained=True, isFreeze=False):
        super(RRSVMNet, self).__init__()
        basenet = VGGLayered_base.VGG16Conv(pretrained=pretrained, isFreeze=isFreeze)
        self.basenet = basenet

        self.upsample = nn.ConvTranspose2d(self.basenet.p5_size[1], self.basenet.p5_size[1]*2, 3, stride=2, padding=0)

        self.d_size = getOutputSize(self.basenet.p5_size, self.upsample)
        self.drop = nn.Dropout(p=0.5)
        self.RRSVM = RRSVM(self.d_size[1], self.d_size[2])
        self.output_size = getOutputSize(self.d_size, self.RRSVM)



    def forward(self, x):
        x, _, _, _, _ = self.basenet(x)
        x = self.upsample(x)
        x = self.RRSVM(x)
        return x

    def get_name(self):
        return self.__class__.__name__


if __name__ == '__main__':
    test_input = torch.randn([1, 3, 224, 224])
    compNet = RRSVMNet()
    test_input = Variable(test_input)
    output = compNet(test_input)
    print "DEBUG"




