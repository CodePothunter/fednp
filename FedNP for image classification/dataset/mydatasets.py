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

# wget http://cs231n.stanford.edu/tiny-imagenet-200.zip

class TINYIMAGENET(Dataset):
    def __init__(self, root, train=True, download=False, transform=None):
        self.root = root
        self.train = train
        self.transform =transform
        self.data = []
        self.targets = []

        self.classes = []
        self.class_to_idx = {}

        with open(os.path.join(self.root, 'wnids.txt')) as wnid:
            for i, line in enumerate(wnid):
                c = line.strip('\n').split('\t')[0]
                self.classes.append(c)
                self.class_to_idx[c] = i

        if self.train:
            for c in self.classes:
                img_dir = os.path.join(self.root, 'train', c, 'images')
                for img in os.listdir(img_dir):
                    img_path = os.path.join(img_dir, img)
                    self.data.append(cv2.imread(img_path))
                    self.targets.append(self.class_to_idx[c])
            self.data = np.stack(self.data)
        else:
            with open(os.path.join(self.root, 'val', 'val_annotations.txt')) as f:
                for line in f:
                    img_file = line.strip('\n').split('\t')[0]
                    c = line.strip('\n').split('\t')[1]
                    img_path = os.path.join(self.root, 'val', 'images', img_file)
                    self.data.append(cv2.imread(img_path))
                    self.targets.append(self.class_to_idx[c])
        
    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        label = self.targets[index]
        if self.transform:
            image = self.transform(image)
        return image, label
        
    def __len__(self):
        return len(self.targets)
