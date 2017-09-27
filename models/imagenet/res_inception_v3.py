import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import RRSVM.RRSVM as RRSVM
from torch.nn import Parameter
__all__ = ['Inception3', 'inception_v3']


model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
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
    for name, param in state_dict.items():
        if name not in own_state:
            raise KeyError('unexpected key "{}" in state_dict'
                           .format(name))
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

    missing = list(set(own_state.keys()) - set(state_dict.keys()))
    missing = sorted(missing)
    if len(missing) > 0:
        for s_missing in missing:
            if s_missing[-1] == 's':
                # print "{:s} is NOT loaded from Orig Model".format(s_missing)
                continue
            else:
                raise KeyError('missing keys in state_dict: "{}"'.format(s_missing))


def inception_v3(pretrained=False, **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        model = Inception3(**kwargs)
        load_state_dict(model, model_zoo.load_url(model_urls['inception_v3_google']))
        return model

    return Inception3(**kwargs)


class Inception3(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False, useRRSVM=True):
        super(Inception3, self).__init__()
        self.useRRSVM = useRRSVM
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        if useRRSVM:
            self.pool2d_2b_r = RRSVM.RRSVM_Module(in_channels=64, kernel_size=3, stride=2)
            self.pool2d_2b = nn.MaxPool2d(kernel_size=3, stride=2)

        else:
            self.pool2d_2b = nn.MaxPool2d(kernel_size=3, stride=2)

        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        # 3 inception Fig 5
        if useRRSVM:
            self.pool2d_4a_r = RRSVM.RRSVM_Module(in_channels=192, kernel_size=3, stride=2)
            self.pool2d_4a = nn.MaxPool2d(kernel_size=3, stride=2)

        else:
            self.pool2d_4a = nn.MaxPool2d(kernel_size=3, stride=2)

        self.Mixed_5b = InceptionA(192, pool_features=32, useRRSVM=useRRSVM)
        self.Mixed_5c = InceptionA(256, pool_features=64, useRRSVM=useRRSVM)
        self.Mixed_5d = InceptionA(288, pool_features=64, useRRSVM=useRRSVM)
        self.Mixed_6a = InceptionB(288, useRRSVM=useRRSVM)
        self.Mixed_6b = InceptionC(768, channels_7x7=128, useRRSVM=useRRSVM)
        self.Mixed_6c = InceptionC(768, channels_7x7=160, useRRSVM=useRRSVM)
        self.Mixed_6d = InceptionC(768, channels_7x7=160, useRRSVM=useRRSVM)
        self.Mixed_6e = InceptionC(768, channels_7x7=192, useRRSVM=useRRSVM)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes, useRRSVM=useRRSVM)
        self.Mixed_7a = InceptionD(768, useRRSVM=useRRSVM)
        self.Mixed_7b = InceptionE(1280, useRRSVM=useRRSVM)
        self.Mixed_7c = InceptionE(2048, useRRSVM=useRRSVM)

        if useRRSVM:
            self.pool2d_8r = RRSVM.RRSVM_Module(in_channels=2048, kernel_size=8)
            self.pool2d_8 = nn.AvgPool2d(kernel_size=8)

        else:
            self.pool2d_8 = nn.AvgPool2d(kernel_size=8)

        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.data.numel()))
                values = values.view(m.weight.data.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        # x = F.max_pool2d(x, kernel_size=3, stride=2)

        if self.useRRSVM:
            x_r = self.pool2d_2b_r(x)
            x_0 = self.pool2d_2b(x)
            x = x_0 + x_r
        else:
            x = self.pool2d_2b(x)

        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        # x = F.max_pool2d(x, kernel_size=3, stride=2)
        if self.useRRSVM:
            x_r = self.pool2d_4a_r(x)
            x_0 = self.pool2d_4a(x)
            x = x_0 + x_r
        else:
            x = self.pool2d_4a(x)
        # 35 x 35 x 192

        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        if self.useRRSVM:
            x_r = self.pool2d_8r(x)
            x_0 = self.pool2d_8(x)
            x = x_0 + x_r
        else:
            x = self.pool2d_8(x)
        # x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        x = self.fc(x)
        # 1000 (num_classes)
        if self.training and self.aux_logits:
            return x, aux
        return x


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features, useRRSVM=True):
        super(InceptionA, self).__init__()
        self.useRRSVM = useRRSVM
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)
        if useRRSVM:
            self.branch_pool_0r = RRSVM.RRSVM_Module(in_channels=in_channels, kernel_size=3, stride=1, padding=1)
            self.branch_pool_0 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        else:
            self.branch_pool_0 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        if self.useRRSVM:
            branch_pool_r = self.branch_pool_0r(x)
            branch_pool_o = self.branch_pool_0(x)
            branch_pool = branch_pool_o + branch_pool_r
        else:
            branch_pool = self.branch_pool_0(x)

        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels, useRRSVM=True):
        super(InceptionB, self).__init__()
        self.useRRSVM = useRRSVM
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

        if useRRSVM:
            self.branch_pool_r = RRSVM.RRSVM_Module(in_channels=in_channels, kernel_size=3, stride=2)
            self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

        else:
            self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)


    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        if self.useRRSVM:
            branch_pool = self.branch_pool(x) + self.branch_pool_r(x)
        else:
            branch_pool = self.branch_pool(x)
        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7, useRRSVM=True):
        super(InceptionC, self).__init__()
        self.useRRSVM = useRRSVM
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        if useRRSVM:
            self.branch_pool_0r = RRSVM.RRSVM_Module(in_channels=in_channels, kernel_size=3, stride=1, padding=1)
            self.branch_pool_0 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        else:
            self.branch_pool_0 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        # branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        if self.useRRSVM:
            branch_pool = self.branch_pool_0(x) + self.branch_pool_0r(x)
        else:
            branch_pool = self.branch_pool_0(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels, useRRSVM=True):
        super(InceptionD, self).__init__()
        self.useRRSVM = useRRSVM
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)
        if useRRSVM:
            self.branch_pool_r =RRSVM.RRSVM_Module(in_channels, kernel_size=3, stride=2)
            self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

        else:
            self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        if self.useRRSVM:
            branch_pool =  self.branch_pool(x) + self.branch_pool_r(x)
        else:
            # branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
            branch_pool =self.branch_pool(x)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels, useRRSVM=True):
        super(InceptionE, self).__init__()
        self.useRRSVM = useRRSVM
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        if useRRSVM:
            self.branch_pool_0 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
            self.branch_pool_0r = RRSVM.RRSVM_Module(in_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.branch_pool_0 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        if self.useRRSVM:
            branch_pool = self.branch_pool_0(x) + self.branch_pool_0r(x)
        else:
            branch_pool = self.branch_pool_0(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes, useRRSVM=True):
        super(InceptionAux, self).__init__()
        self.useRRSVM = useRRSVM
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001
        if useRRSVM:
            self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
            self.pool_r = RRSVM.RRSVM_Module(in_channels, kernel_size=5, stride=3)
        else:
            self.pool = nn.AvgPool2d(kernel_size=5, stride=3)

    def forward(self, x):
        # 17 x 17 x 768
        if self.useRRSVM:
            x_o = self.pool(x)
            x_r = self.pool_r(x)
            x = x_o + x_r
        else:
        # x = F.avg_pool2d(x, kernel_size=5, stride=3)
            x = self.pool(x)
        # 5 x 5 x 768
        x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1(x)
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 1000
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

if __name__ == '__main__':
    from torch.autograd import Variable
    net = inception_v3(pretrained=True, useRRSVM=True)
    x = torch.randn(1, 3, 299, 299)
    y, y_aux = net(Variable(x))
    print y.size()
    print y_aux.size()
