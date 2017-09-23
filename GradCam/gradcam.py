from __future__ import print_function

from collections import OrderedDict

# import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from PIL import Image


class PropagatationBase(object):

    def __init__(self, model, target_layer, n_class, cuda=True):
        self.model = model
        self.model.eval()
        self.cuda = cuda
        if self.cuda:
            self.model.cuda()
        self.target_layer = target_layer
        self.n_class = n_class
        self.probs = None
        self.all_fmaps = OrderedDict()
        self.all_grads = OrderedDict()
        self.set_hook_func()

    def set_hook_func(self):
        raise NotImplementedError

    def encode_one_hot(self, idx):
        one_hot = torch.FloatTensor(1, self.n_class).zero_()
        one_hot[0][idx] = 1.0
        return one_hot

    def load_image(self, filename, transform):
        # self.raw_image = cv2.imread(filename)[:, :, ::-1]
        # self.raw_image = cv2.resize(self.raw_image, (224, 224))
        self.raw_image = Image.open(filename).convert('RGB')
        self.raw_image = self.raw_image.resize([224, 224])
        self.image = transform(self.raw_image).unsqueeze(0)
        if self.cuda:
            self.image = self.image.cuda()
        self.image = Variable(self.image, volatile=False, requires_grad=True)

    def forward(self):
        self.preds = self.model.forward(self.image)
        self.probs = F.softmax(self.preds)[0]
        self.prob, self.idx = self.probs.data.sort(0, True)

    def backward(self, idx):
        self.model.zero_grad()
        one_hot = self.encode_one_hot(idx)
        if self.cuda:
            one_hot = one_hot.cuda()
        self.preds.backward(gradient=one_hot, retain_graph=True)

    def find(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('invalid layer name: {}'.format(target_layer))


class GradCAM(PropagatationBase):

    def set_hook_func(self):

        def func_f(module, input, output):
            self.all_fmaps[id(module)] = output.data.cpu()

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_out[0].cpu()

        for module in self.model.named_modules():
            module[1].register_forward_hook(func_f)
            module[1].register_backward_hook(func_b)

    def normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm.data[0]

    def compute_grad_weights(self, grads):
        grads = self.normalize(grads)
        self.map_size = grads.size()[2:]
        return nn.AvgPool2d(self.map_size)(grads)

    def generate(self):
        fmaps = self.find(self.all_fmaps, self.target_layer)
        grads = self.find(self.all_grads, self.target_layer)
        weights = self.compute_grad_weights(grads)

        gcam = torch.FloatTensor(self.map_size).zero_()
        for fmap, weight in zip(fmaps[0], weights[0]):
            gcam += fmap * weight.data
        gcam = F.relu(Variable(gcam))

        return gcam.data.numpy()

    def save(self, filename, gcam):
        gcam -= gcam.min()
        gcam /= gcam.max()
        gcam_image = Image.fromarray(np.uint8(gcam*255))
        gcam_image = gcam_image.resize([224, 224])
        # gcam = cv2.resize(gcam * 255, (224, 224))
        # gcam = cv2.applyColorMap(np.uint8(gcam), cv2.COLORMAP_JET)
        gcam =np.array(gcam_image.convert('RGB')).astype(np.float) + np.array(self.raw_image).astype(np.float)
        gcam = 255 * gcam / np.max(gcam)
        gcam_image = Image.fromarray(np.uint8(gcam))
        gcam_image.save(filename)
        # cv2.imwrite(filename, gcam)


class BackPropagation(PropagatationBase):

    def set_hook_func(self):

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_in[0].cpu()

        for module in self.model.named_modules():
            module[1].register_backward_hook(func_b)

    def generate(self):
        output = self.find(self.all_grads, self.target_layer)
        return output.data.numpy()[0].transpose(1, 2, 0)

    def save(self, filename, data):
        data -= data.min()
        data /= data.max()

        image = Image.fromarray(np.uint8(data * 255))
        image.save(filename)
        # data = np.uint8(data * 255)
        # cv2.imwrite(filename, data)


class GuidedBackPropagation(BackPropagation):

    def set_hook_func(self):

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_in[0].cpu()

            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (F.threshold(grad_in[0], threshold=0.0, value=0.0),)

        for module in self.model.named_modules():
            module[1].register_backward_hook(func_b)