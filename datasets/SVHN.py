from py_utils import dir_utils
import os
from torchvision import datasets, transforms
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import argparse
import numpy as np
# adopted from https://github.com/aaron-xichen/pytorch-playground/blob/master/svhn/dataset.py


transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])


def target_transform(target):
    return int(target[0]) - 1

def get_SVHN_datasets(args, train_portion=1.0):

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {'num_workers': 4}
    dataset_root = dir_utils.get_dir(os.path.join(os.path.expanduser('~'), 'datasets', 'RRSVM_datasets'))

    train_set = datasets.SVHN(root=dataset_root, split='train', download=True, transform=transform, target_transform=target_transform)

    if train_portion < 1.0:
        np.random.seed(args.seed or 0)
        n_samples = len(train_set)
        categorical_labels = list(set(train_set.labels.squeeze().tolist()))
        n_categories = len(categorical_labels)
        # evenly sample:
        selected_indices = []
        for idx in range(n_categories):
            categorical_idx = [i for i in range(n_samples) if train_set.labels[i,0] == categorical_labels[idx]]
            n_categorical_samples = len(categorical_idx)
            indices = np.random.permutation(n_categorical_samples)
            relative_indices = indices[:][: int(n_categorical_samples * train_portion)]

            s_selected_indices = [categorical_idx[i] for i in relative_indices]
            selected_indices.extend(s_selected_indices)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.train_batch_size,
                                                   sampler=SubsetRandomSampler(selected_indices), **kwargs)
    else:

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, **kwargs)

    testset = datasets.SVHN(root=dataset_root, split='test', download=True, transform=transform, target_transform=target_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args, shuffle=False, **kwargs)
    return train_loader, test_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Loader')
    args = parser.parse_args()
    args.cuda = False
    args.train_batch_size = 20
    args.test_batch_size = 20
    args.seed = 0
    train_loader, test_loader = get_SVHN_datasets(args, train_portion=0.1)
    print"DEBUG"