import os
import sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/RRSVM_pytorch')
sys.path.append(project_root)

import scipy.io as sio
import torch.utils.data as data

import glob
import numpy as np
import torchvision.datasets.folder as dataset_utils
import torch.utils.data
import torchvision.transforms as transforms
import pickle


dataset_directory = os.path.join(os.path.expanduser('~'), 'datasets/MPII')
image_directory = os.path.join(dataset_directory, 'images')
annotation_file = os.path.join(dataset_directory, 'annotation.pkl')

def MPII_val_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    return transforms.Compose([
            transforms.Resize(480),
            transforms.CenterCrop(450),
            transforms.ToTensor(),
            normalize,
        ])


def MPII_train_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    return transforms.Compose([
            transforms.Resize(480),
            transforms.RandomResizedCrop(450),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])



nClasses = 393

class MPIIDataset(data.Dataset):

    def __init__(self, split='train', transform=None, file_ext='jpg', default_loader=None):
        if default_loader is None:
            self.default_loader = dataset_utils.default_loader
        else:
            self.default_loader = default_loader

        annotations = pickle.load(open(annotation_file, 'rb'))
        image_path_list = glob.glob(os.path.join(image_directory, '*.{:s}'.format(file_ext)))
        image_path_list.sort()
        annotation_split = annotations[split]
        self.image_path_list = []
        self.labels = []
        for s_item in annotation_split:
            image_name = s_item[0]
            class_id = s_item[1]
            s_filepath = os.path.join(image_directory, image_name)
            # if s_filepath in image_path_list:
            # test passed
            self.image_path_list.append(s_filepath)
            self.labels.append(class_id)
            # else:
            #     print("{:s} Not Found\n".format(image_name))



        self.split = split
        self.transform = transform
        print "{:s}\t{:d} Images Found".format(self.split, len(self.image_path_list))

    def __getitem__(self, index):
        s_image_path = self.image_path_list[index]
        s_image = self.default_loader(s_image_path)

        target = self.labels[index]

        if self.transform is not None:
            s_image = self.transform(s_image)

        return s_image, target

    def __len__(self):
        return len(self.image_path_list)



if __name__ == '__main__':
    import torchvision.transforms as transforms
    import torchvision.models as models
    import torch.utils.model_zoo as model_zoo
    import torch

    import os
    import shutil


    def Res50Places_val_transform():
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])


    HicoDataset = MPIIDataset(split='val', transform=Res50Places_val_transform())
    HicoLoader = torch.utils.data.DataLoader(HicoDataset, batch_size=20, shuffle=False)
    for i, (image, label) in enumerate(HicoLoader):

        print "DEBUG"