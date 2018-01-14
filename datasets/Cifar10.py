from py_utils import dir_utils
import os
from torchvision import datasets, transforms
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import argparse
import numpy as np


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def get_cifar10_datasets(args, train_portion=1.0):

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {'num_workers': 4}
    dataset_root = dir_utils.get_dir(os.path.join(os.path.expanduser('~'), 'datasets', 'RRSVM_datasets'))

    train_set = datasets.CIFAR10(root=dataset_root, train=True, download=True, transform=transform_train)

    if train_portion < 1.0:
        np.random.seed(args.seed or 0)
        n_samples = len(train_set)
        categorical_labels = list(set(train_set.train_labels))
        n_categories = len(categorical_labels)
        # evenly sample:
        selected_indices = []
        for idx in range(n_categories):
            categorical_idx = [i for i in range(n_samples) if train_set.train_labels[i] == categorical_labels[idx]]
            n_categorical_samples = len(categorical_idx)
            indices = np.random.permutation(n_categorical_samples)
            relative_indices = indices[:][: int(n_categorical_samples * train_portion)]

            s_selected_indices = [categorical_idx[i] for i in relative_indices]
            selected_indices.extend(s_selected_indices)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.train_batch_size,
                                                   sampler=SubsetRandomSampler(selected_indices), **kwargs)
    else:

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, **kwargs)

    testset = datasets.CIFAR10(root=dataset_root, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args, shuffle=False, **kwargs)
    return train_loader, test_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Loader')
    args = parser.parse_args()
    args.cuda = False
    args.train_batch_size = 20
    args.test_batch_size = 20
    train_loader, test_loader = get_cifar10_datasets(args, train_portion=0.1)
    print"DEBUG"