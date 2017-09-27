'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import RRSVM.RRSVM as RRSVM


# M: MaxPooling, O: OrderedWeightedAveraging, A: AvgPooling
cfg = {
    'VGG16_F_M': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG16_F_O': [64, 64, 'O', 128, 128, 'O', 256, 256, 256, 'O', 512, 512, 512, 'O', 512, 512, 512, 'O'],
    'VGG16_F_A': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 'A', 512, 512, 512, 'A'],
    'VGG16_M_O': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'O'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, n_classes, p_constraint=False):  # three different types: OWA, MAX, AVG
        super(VGG, self).__init__()
        self.p_constraint = p_constraint
        self.n_classes = n_classes

        self.features = self._make_layers(cfg[vgg_name.upper()])
        self.classifier = nn.Linear(512, n_classes)
        self.image_size = [32, 32, 3]

    def forward(self, x):
        pool_indices = []
        for s_module in self.features:
            if isinstance(s_module, (nn.MaxPool2d, RRSVM.RRSVM_Module,)):
                x, p = s_module(x)
                pool_indices.append(p)
            else:
                x = s_module(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)]
            elif x == 'A':
                    layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            elif x == 'O':
                    layers += [RRSVM.RRSVM_Module(in_channels, kernel_size=2, stride=2, return_indices=True, p_constraint=self.p_constraint)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


if __name__ == '__main__':

    net = VGG('VGG16_F_O')
    x = torch.randn(2,3,32,32)
    print(net(Variable(x)))
    print(net(Variable(x)).size())