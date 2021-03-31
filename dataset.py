#!usr/bin/env python  
#-*- coding:utf-8 _*- 
"""
@version: python3.6
@author: ikkyu-wen
@contact: wenruichn@gmail.com
@time: 2019-09-07 20:27
公众号：AI成长社
知乎：https://www.zhihu.com/people/qlmx-61/columns
"""
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import pandas as pd
import six
import sys
from PIL import Image
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Dataset(Dataset):
    def __init__(self, root=None, transform=None, target_transform=None, to=None):
        if '.txt' in root:
            self.env = list(open(root))
        else:
            self.env = root

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        self.len = len(self.env) - 1

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1

        try:
            img_path, label = self.env[index].strip().split(',')
        except:
            img_path, a, label = self.env[index].strip().split(',')
            img_path = img_path + "," + a

        #img_path = self.env[index].strip()
        # print(img_path)
        # print(label)
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, int(label),img_path

class TestDataset(Dataset):
    def __init__(self, root=None, transform=None, transform1=None, transform2=None, transform3=None,
                 transform4=None, transform5=None, target_transform=None, to=None):
        if '.txt' in root:
            self.env = list(open(root))
        else:
            self.env = root

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        self.len = len(self.env) - 1

        self.transform = transform
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3
        self.transform4 = transform4
        self.transform5 = transform5
        self.target_transform = target_transform
        self.to = to

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        img_path, label = self.env[index].strip().split(',')

        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img1 = self.transform(img)
        if self.transform1 is not None:
            img2 = self.transform1(img)
        if self.transform2 is not None:
            img3 = self.transform2(img)
        if self.transform3 is not None:
            img4 = self.transform3(img)
        if self.transform4 is not None:
            img5 = self.transform4(img)
        if self.transform5 is not None:
            img6 = self.transform5(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img1, int(label))


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        # padding
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w+w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h+h_padding))

        # img.show()
        # resize
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img
