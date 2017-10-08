import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import RRSVM.RRSVM as RRSVM
from torch.nn import Parameter

__all__ = [ 'vgg16_bn',]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


def load_state_dict(model, state_dict):
    """Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. The keys of :attr:`state_dict` must
    exactly match the keys returned by this module's :func:`state_dict()`
    function.

    Arguments:
        state_dict (dict): A dict containing parameters and
            persistent buffers.
    """
    own_state = model.state_dict()
    # print own_state.keys()
    # print state_dict.keys()
    l_dec = 0
    for name in own_state.keys():
        name_split = name.split('.')
        if name_split[-1] == 's':
            l_dec += 1
            continue
        if name_split[0] == 'features':
            name_split[1] = str(int(name_split[1]) - l_dec)

        saved_name = '.'.join(name_split)
        param = state_dict[saved_name]
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            own_state[name].copy_(param)
        except:
            print('While copying the parameter named {}, whose dimensions in the model are'
                  ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                name, own_state[name].size(), param.size()))
            raise
    # for name, param in state_dict.items():
    #
    #     if name not in own_state:
    #         raise KeyError('unexpected key "{}" in state_dict'
    #                        .format(name))
    #     if isinstance(param, Parameter):
    #         # backwards compatibility for serialized parameters
    #         param = param.data
    #     try:
    #         own_state[name].copy_(param)
    #     except:
    #         print('While copying the parameter named {}, whose dimensions in the model are'
    #               ' {} and whose dimensions in the checkpoint are {}, ...'.format(
    #             name, own_state[name].size(), param.size()))
    #         raise

    # missing = list(set(own_state.keys()) - set(state_dict.keys()))
    # missing = sorted(missing)
    # if len(missing) > 0:
    #     for s_missing in missing:
    #         if s_missing[-1] == 's':
    #             # print "{:s} is NOT loaded from Orig Model".format(s_missing)
    #             continue
    #         else:
    #             raise KeyError('missing keys in state_dict: "{}"'.format(s_missing))

class VGG(nn.Module):

    def __init__(self, features, useRRSVM=False, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()
        self.useRRSVM= useRRSVM

    def forward(self, x):
        if self.useRRSVM:
            for i, s_module in enumerate(self.features):

                if isinstance(s_module, (nn.MaxPool2d)):
                    continue
                if isinstance(s_module, (RRSVM.RRSVM_Module)):
                    x1 = s_module(x)
                    x2 = self.features[i+1](x)
                    x = x1 + x2
                else:
                    x = s_module(x)
        else:
            x = self.features(x)


        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, useRRSVM=False, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            if useRRSVM:
                layers += [RRSVM.RRSVM_Module(in_channels=in_channels, kernel_size=2, stride=2)]
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}





def vgg16_bn(pretrained=False, useRRSVM=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D'], useRRSVM=useRRSVM, batch_norm=True), useRRSVM=useRRSVM, **kwargs)
    if pretrained:
        load_state_dict(model, model_zoo.load_url(model_urls['vgg16_bn']))
    return model



if __name__ == '__main__':
    model = vgg16_bn(pretrained=True, useRRSVM=True)
    print 'DEBUG'