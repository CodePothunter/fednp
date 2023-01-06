import torch
from torch.functional import split
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from torchvision import datasets, transforms
import numpy as np
import os
import cv2
from PIL import Image

from dataset import mydatasets


mean_dict = {
    'cifar100': (0.5071, 0.4865, 0.4409),
    'cifar10': (0.4914, 0.4822, 0.4465),
    'mnist': (0.1307,),
    'tinyimagenet': (0.3975, 0.4481, 0.4802)
}
std_dict = {
    'cifar100': (0.2673, 0.2564, 0.2762),
    'cifar10': (0.2470, 0.2435, 0.2616),
    'mnist': (0.3081,),
    'tinyimagenet': (0.2816, 0.2689, 0.2764)
}
input_size_dict = {
    'cifar100': 32,
    'cifar10': 32,
    'mnist': 28,
    'tinyimagenet': 64
}
num_classes_dict = {
    'cifar100': 100,
    'cifar10': 10,
    'mnist': 10,
    'tinyimagenet': 200
}

builtin = ['cifar100', 'cifar10', 'mnist']

def clf_dataset(args):
    ds = args['dataset']
    mean = mean_dict[ds]
    std = std_dict[ds]
    size = input_size_dict[ds]
    num_classes = num_classes_dict[ds]
    in_channels = 1 if ds == 'mnist' else 3

    trans = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    transform_test = transforms.Compose(trans)

    if ds != 'mnist':
        trans.insert(0, transforms.RandomCrop(size, padding=4))
        trans.insert(1, transforms.RandomHorizontalFlip())

    transform_train = transforms.Compose(trans)

    if ds in builtin:
        dataset_class = datasets
    else:
        dataset_class = mydatasets

    train_dataset = getattr(dataset_class, ds.upper())(f'data/{ds}', train=True, download=True, transform=transform_train)
    test_dataset = getattr(dataset_class, ds.upper())(f'data/{ds}', train=False, download=True, transform=transform_test)
    return train_dataset, test_dataset, num_classes


def split_dataset(args, train_dataset):
    if args['K'] == 1:
        train_datasets = [train_dataset]
        return train_datasets
    if args['data_dist'] == 'random':
        splits = []
        size = len(train_dataset) // args['K']
        for _ in range(args['K'] - 1):
            splits.append(size)
        splits.append(len(train_dataset) - size * (args['K'] - 1))

        train_datasets = torch.utils.data.random_split(train_dataset, splits)
        return train_datasets
    
    train_targets = np.array(train_dataset.targets)
    ds = args['dataset']
    num_classes = num_classes_dict[ds]
    beta = args['beta']
    K = args['K']
    num_per_class = len(train_dataset) // num_classes
    client_idxes = [[] for i in range(K)]
    for k in range(num_classes):
        idxes = np.where(train_targets == k)[0]
        proportions = list(map(int, (np.random.dirichlet(np.repeat(beta, K)) * num_per_class).tolist()))
        proportions[-1] += num_per_class - sum(proportions)
        client_idxes.sort(key=lambda x: len(x))
        proportions.sort()
        for i, (client_idx, proportion) in enumerate(zip(client_idxes[::-1], proportions)):
            client_idx.extend(idxes[sum(proportions[:i]): sum(proportions[:i]) + proportion])

    train_datasets = []
    weights = np.empty((K, num_classes))
    for i in range(K):
        train_datasets.append(torch.utils.data.Subset(train_dataset, client_idxes[i]))
        for k in range(num_classes):
            weights[i, k] = (train_targets[client_idxes[i]] == k).sum()

    return train_datasets, weights
    