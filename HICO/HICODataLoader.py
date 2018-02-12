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

dataset_directory = os.path.join(os.path.expanduser('~'), 'datasets/HICO')
#TODO: This is important to guarantee the train and val are not mixed!
np.random.seed(0)

def HICO_val_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])


def HICO_train_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])


class HICODataset(data.Dataset):

    def __init__(self, split='train', transform=None, file_ext='jpg', default_loader=None):
        if default_loader is None:
            self.default_loader = dataset_utils.default_loader
        else:
            self.default_loader = default_loader
        if split == 'train' or split == 'val' or split=='trainall':
            self.image_path_list = glob.glob(os.path.join(dataset_directory, 'images/train2015', '*.{:s}'.format(file_ext)))
            self.image_path_list.sort()
            annotation_mat = sio.loadmat(os.path.join(dataset_directory, 'train.mat'))
            self.annotations = annotation_mat['train_labels']
            perm_idx = np.random.permutation(len(self.image_path_list))
            if split == 'train':
                perm_idx = perm_idx[:32000]
            elif split=='val':
                perm_idx = perm_idx[32000:]
            else:
                pass
            self.image_path_list = [self.image_path_list[i] for i in perm_idx]
            self.annotations = self.annotations[:, perm_idx]

        elif split == 'test':
            self.image_path_list = glob.glob(os.path.join(dataset_directory, 'images/test2015', '*.{:s}'.format(file_ext)))
            self.image_path_list.sort()
            annotation_mat = sio.loadmat(os.path.join(dataset_directory, 'test.mat'))
            self.annotations = annotation_mat['test_labels']
        else:
            print "Unrecognized split\t{:s}".format(split)
            sys.exit(-1)
        self.split = split
        self.transform = transform
        print "{:s}\t{:d} Images Found".format(self.split, len(self.image_path_list))


    def __getitem__(self, index):
        s_image_path = self.image_path_list[index]
        s_image = self.default_loader(s_image_path)

        s_annotation = self.annotations[:, index]
        s_annotation[s_annotation!=1] = 0 # set all not clear to 0
        s_annotation = torch.FloatTensor(s_annotation)

        if self.transform is not None:
            s_image = self.transform(s_image)

        return s_image, s_annotation

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


    HicoDataset = HICODataset(split='test', transform=Res50Places_val_transform())
    HicoLoader = torch.utils.data.DataLoader(HicoDataset, batch_size=20, shuffle=True)
    for i, (image, label) in enumerate(HicoLoader):

        print "DEBUG"