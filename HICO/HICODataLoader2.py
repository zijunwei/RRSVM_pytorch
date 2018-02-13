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
            transforms.Resize(480),
            transforms.CenterCrop(450),
            transforms.ToTensor(),
            normalize,
        ])


def HICO_train_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    return transforms.Compose([
            transforms.Resize(480),
            transforms.RandomResizedCrop(450),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])


def _build_label_lookup(label_text):
  """Build lookup for label to object-verb description.
  Args:
    label_text: string, path to file containing mapping from
      label to object-verb description.
      Assumes each line of the file looks like:
        0    airplane board
        1	 airplane direct
        2	 airplane exit
      where each line corresponds to a unique mapping. Note that each line is
      formatted as <label> <object> <verb>.
  Returns:
    Dictionary of synset to human labels, such as:
      0 --> 'airplane board'
  """
  lines = open(label_text, 'r').readlines()
  label_to_text = {}
  for l in lines:
    if l:
      parts = l.strip().split(' ')
      assert len(parts) == 3
      label = int(parts[0])
      text = parts[1:]
      label_to_text[label] = text
  return label_to_text


def getLabels(labels_file):
    lines = open(labels_file, 'r').readlines()
    labels = []
    for l in lines:
        s_label = []
        parts = l.strip().split(' ')
        # Encode label to num_class-dim vectors
        for part in parts:
            s_label.append(int(part))
        labels.append(s_label)

    return labels

nClasses = 600

class HICODataset(data.Dataset):

    def __init__(self, split='train', transform=None, file_ext='jpg', default_loader=None):
        if default_loader is None:
            self.default_loader = dataset_utils.default_loader
        else:
            self.default_loader = default_loader


        annotation_file = os.path.join(dataset_directory, 'labels_{:s}.txt'.format(split))
        image_path_list = glob.glob(os.path.join(dataset_directory, 'images/{:s}2015'.format(split), '*.{:s}'.format(file_ext)))
        image_path_list.sort()
        self.labels = getLabels(annotation_file)
        self.image_path_list = image_path_list
        # clean up to lists
        # self.annotations = []
        # self.image_path_list = []
        # for idx, (s_labels, s_file_path) in enumerate(zip(labels, image_path_list)):
        #         self.image_path_list.append(s_file_path)
        #         self.annotations.append(s_labels)
        #

        self.split = split
        self.transform = transform
        print "{:s}\t{:d} Images Found".format(self.split, len(self.image_path_list))

    def __getitem__(self, index):
        s_image_path = self.image_path_list[index]
        s_image = self.default_loader(s_image_path)

        s_positive_idxes = self.labels[index]

        s_annotation = np.zeros(600)
        s_annotation[s_positive_idxes] = 1
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


    HicoDataset = HICODataset(split='train', transform=Res50Places_val_transform())
    HicoLoader = torch.utils.data.DataLoader(HicoDataset, batch_size=20, shuffle=False)
    for i, (image, label) in enumerate(HicoLoader):

        print "DEBUG"