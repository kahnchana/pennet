import math
import os
import random
from random import shuffle

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader

from core.utils import ZipReader

dataset_dict = {
    "cifar10": torchvision.datasets.CIFAR10,
    "stl10": torchvision.datasets.STL10,
    "pascal": torchvision.datasets.VOCDetection,
    "linneaus5": torchvision.datasets.ImageFolder,
}


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_args, split='train', level=None, debug=None):
        data_root = data_args['zip_root']
        if split == 'train':
            train = True
            if data_args["dataset"].startswith("linneaus5"):
                data_root = "{}/Linnaeus 5 64X64/train".format(data_root)
        else:
            train = False
            if data_args["dataset"].startswith("linneaus5"):
                data_root = "{}/Linnaeus 5 64X64/test".format(data_root)

        self.data = dataset_dict[data_args["dataset"]](root=data_root, image_set='train', download=False)
        # self.data = dataset_dict[data_args["dataset"]](root=data_root)

        self.split = split
        self.level = level
        self.w, self.h = data_args['w'], data_args['h']
        self.mask_type = data_args.get('mask', 'pconv')
        self.mask = [0] * len(self.data)

    def __len__(self):
        return len(self.data)

    def set_subset(self, start, end):
        self.mask = self.mask[start:end]
        self.data = self.data[start:end]

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)
        return item

    def load_item(self, index):
        # load image
        img, _ = self.data[index]
        img_name = "{:05d}.png".format(5)

        # load mask
        m = np.zeros((self.h, self.w)).astype(np.uint8)
        x1 = random.randint(5, 7)
        w1 = random.randint(20, 34)
        # w1 = random.randint(45, 50)
        y1 = random.randint(5, 7)
        h1 = random.randint(45, 50)
        m[x1: w1, y1: h1] = 255

        mask = Image.fromarray(m).convert('L')
        # augment
        if self.split == 'train':
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ColorJitter(0.05, 0.05, 0.05, 0.05)(img)
            mask = transforms.RandomHorizontalFlip()(mask)
            mask = mask.rotate(random.randint(0, 45), expand=True)
            mask = mask.filter(ImageFilter.MaxFilter(3))
        img = img.resize((self.w, self.h))
        mask = mask.resize((self.w, self.h), Image.NEAREST)
        return F.to_tensor(img) * 2 - 1., F.to_tensor(mask), img_name

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(dataset=self, batch_size=batch_size, drop_last=True)
            for item in sample_loader:
                yield item
