import torchvision.models as models
import ResNet
import torch.utils.model_zoo as model_zoo
import torch

import PtUtils.cuda_model as cuda_model
import os
import shutil
from torch.nn.parameter import Parameter
from  torch.autograd import Variable
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def getRes101Model(eval=False, gpu_id=None, multiGpu=False, useRRSVM=False):
    #TODO: not specific layers to freeze
    model = ResNet.ResNet(ResNet.Bottleneck, [3, 4, 23, 3], num_classes=600, useRRSVM=useRRSVM)
    loaded_state_dict = model_zoo.load_url(model_urls['resnet101'])
    del loaded_state_dict['fc.weight']
    del loaded_state_dict['fc.bias']
    model.load_state_dict(loaded_state_dict, strict=False)
    model = cuda_model.convertModel2Cuda(model, gpu_id, multiGpu)
    if eval:
        model.eval()
    else:
        model.train()
    return model


# def weighted_binary_cross_entropy(output, target, weights=None):
#     if weights is not None:
#         assert len(weights) == 2
#
#         loss = weights[1] * (target * torch.log(output)) + \
#                weights[0] * ((1 - target) * torch.log(1 - output))
#     else:
#         loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
#
#     return torch.neg(torch.mean(loss))


class WeightedBCEWithLogitsLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, reduce=True):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        if self.weight is not None:
            return weighted_binary_cross_entropy_with_logits(input, target,
                                                      Variable(self.weight),
                                                      self.size_average,
                                                      reduce=self.reduce)
        else:
            return weighted_binary_cross_entropy_with_logits(input, target,
                                                      size_average=self.size_average,
                                                      reduce=self.reduce)



def weighted_binary_cross_entropy_with_logits(input, target, weights=None, size_average=True, reduce=True):
    if weights is not None:
        assert len(weights) == 2
    # w[0]: w_n, w[1]: w_p

    max_val = (-input).clamp(min=0)
    loss = weights[0] * (1- target)*(input + max_val) + weights[1] * max_val * target  + \
           (weights[1] * target + weights[0] * (1 - target))*(1 + (-max_val).exp() + (-input - max_val).exp()).log()

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()


if __name__ == '__main__':
    model = getRes101Model(eval=False)
    print "DEBUG"