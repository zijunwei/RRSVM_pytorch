import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable

from pytorch_utils.t_sets import getOutputSize
# this is dividing VGG into smaller parts

#TODO: add loading from file
class VGG16Conv(nn.Module):
    def __init__(self, pretrained=True, isFreeze=True, input_size=None):
        super(VGG16Conv, self).__init__()
        if not input_size:
            input_size = torch.Size([1, 3, 224, 224])

        orig_model = models.vgg16(pretrained=pretrained)
        # set if free gradient (if True, use as only feature extractor, learns nothing)
        for param in orig_model.parameters():
            if isFreeze:
                param.requires_grad = False
            else:
                param.requires_grad = True

        components = list(orig_model.children()) #feature and classifier, total of 2

        feature = components[0]
        # abandon linear

        #divide feature into different parts
        self.p1 = nn.Sequential(*list(feature.children())[0:5])
        self.p1_size = getOutputSize(input_size, self.p1)

        self.p2 = nn.Sequential(*list(feature.children())[5:10])
        self.p2_size = getOutputSize(self.p1_size, self.p2)

        self.p3 = nn.Sequential(*list(feature.children())[10:17])
        self.p3_size = getOutputSize(self.p2_size, self.p3)

        self.p4 = nn.Sequential(*list(feature.children())[17:24])
        self.p4_size = getOutputSize(self.p3_size, self.p4)

        self.p5 = nn.Sequential(*list(feature.children())[24::])
        self.p5_size = getOutputSize(self.p4_size, self.p5)

    def forward(self, x):
        x1 = self.p1(x)
        x2 = self.p2(x1)
        x3 = self.p3(x2)
        x4 = self.p4(x3)
        x5 = self.p5(x4)
        return x5, x4, x3, x2, x1

    def name(self):
        return "VGG16LayerConv"



if __name__ == '__main__':

    VGG16net = VGG16Conv(pretrained=True, isFreeze=True)
    VGG16net.eval()

    image_file = '../image/img1.jpg'

    image = Image.open(image_file)

    series_transforms = transforms.Compose([transforms.Scale(256), transforms.CenterCrop(224), transforms.ToTensor()])

    tensor_image = series_transforms(image)
    # Test if this model is freezed
    # for param in ResNet50Conv.parameters():
    #     print  param.requires_grad
    tensor_image_batch = torch.unsqueeze(tensor_image, 0)
    tensor_image_batch = Variable(tensor_image_batch)
    output_batch = VGG16net(tensor_image_batch)



    print "DBUEG"