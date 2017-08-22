import torch
from torch.autograd import Variable


# def randImage(image_size=None):
#     if not image_size:
#         image_size = [1, 3, 224, 224]
#         return torch.FloatTensor(*image_size)
#     elif isinstance(image_size, torch.Size):
#         return torch.FloatTensor(image_size)
#     elif isinstance(image_size, (list, tuple)):
#         return torch.FloatTensor(*image_size)



# def randImageBatch(batch_size=1, image_size=None):
#     s_image = randImage(image_size)
#     image_batch = torch.unsqueeze(s_image, 0)
#     if batch_size > 1:
#         # TODO here
#         pass
#     return image_batch


def getOutputSize(input_size, module):
    # if first layer, it should be [1, 3, 224, 224]
    module.eval()
    input_image = torch.FloatTensor(torch.randn(*input_size))
    input_image = Variable(input_image)
    output_image = module(input_image)
    output_size = output_image.size()
    return output_size



if __name__ == '__main__':
    # rand_image = randImage()
    # rand_image_batch = randImageBatch()

    print "DBUG"