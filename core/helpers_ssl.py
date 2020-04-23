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


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_args=None, split='train', level=None, debug=None):
        data_root = "/home/kanchana/Downloads/public_datasets"
        self.data = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False)
        self.split = split
        self.level = level
        self.w, self.h = 64, 64
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
            print('loading error: ', self.data[index])
            item = self.load_item(0)
        return item

    def load_item(self, index):
        # load image
        img, label = self.data[index]
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
        return F.to_tensor(img) * 2 - 1., F.to_tensor(mask), img_name, label

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(dataset=self, batch_size=batch_size, drop_last=True)
            for item in sample_loader:
                yield item


class MultiClassHingeLoss(torch.nn.Module):
    """
    SVM loss
    Weston and Watkins version multiclass hinge loss @ https://en.wikipedia.org/wiki/Hinge_loss
    for each sample, given output (a vector of n_class values) and label y (an int \in [0,n_class-1])
    loss = sum_i(max(0, (margin - output[y] + output[i]))^p) where i=0 to n_class-1 and i!=y

    Note: hinge loss is not differentiable
          Let's denote hinge loss as h(x)=max(0,1-x). h'(x) does not exist when x=1,
          because the left and right limits do not converge to the same number, i.e.,
          h'(1-delta)=-1 but h'(1+delta)=0.

    To overcome this obstacle, people proposed squared hinge loss h2(x)=max(0,1-x)^2. In this case,
    h2'(1-delta)=h2'(1+delta)=0
    """

    def __init__(self, p=1, margin=1, weight=None, size_average=True):
        super(MultiClassHingeLoss, self).__init__()
        self.p = p
        self.margin = margin
        # weight for each class, size=n_class, variable containing FloatTensor, cuda, requires_grad=False
        self.weight = weight
        self.size_average = size_average

    def forward(self, output, y):  # output: batchsize*n_class
        # print(output.requires_grad)
        # print(y.requires_grad)
        output_y = output[torch.arange(0, y.size()[0]).long().cuda(), y.data.long().cuda()].view(-1, 1)  # transpose
        # margin - output[y] + output[i]
        loss = output - output_y + self.margin  # contains i=y
        # remove i=y items
        loss[torch.arange(0, y.size()[0]).long().cuda(), y.data.long().cuda()] = 0
        # max(0,_)
        loss[loss < 0] = 0
        # ^p
        if self.p != 1:
            loss = torch.pow(loss, self.p)
        # add weight
        if self.weight is not None:
            loss = loss * self.weight
        # sum up
        loss = torch.sum(loss)
        if self.size_average:
            loss /= output.size()[0]  # output.size()[0]
        return loss
