#!usr/bin/env python  
#-*- coding:utf-8 _*- 
"""
@version: python3.6
@author: ikkyu-wen
@contact: wenruichn@gmail.com
@time: 2019-09-07 18:54
公众号：AI成长社
知乎：https://www.zhihu.com/people/qlmx-61/columns
"""
import random
import math
import torch
from Random_erasing import RandomErasing
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms
from albumentations import *
import albumentations.augmentations.functional as F
import random
import numpy as np
import cv2



import random
from PIL import Image, ImageChops
class Cutout(object):
    def __init__(self, size=32, ratio=0.5):
        self.size = size
        self.p = ratio

    def __call__(self, img):
        if random.random() < self.p:
            x, y = img.size
            cx = random.randint(0, x)
            cy = random.randint(0, y)
            xmin, ymin = max(cx-self.size/2, 0), max(cy-self.size/2, 0)
            xmax, ymax = min(cx+self.size/2, x), min(cy+self.size/2, y)
            mask = Image.new("RGB", (int(xmax-xmin), int(ymax-ymin)))
            mask = mask.point(lambda _: random.randint(0, 255))
            img.paste(mask, (int(xmin), int(ymin)))
        return img
class ChannelDropoutCustom(ImageOnlyTransform):
    """Randomly Drop Channels in the input Image.
    Args:
        channel_drop_range (int, int): range from which we choose the number of channels to drop.
        fill_value : pixel value for the dropped channel.
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, uint16, unit32, float32
    """

    def __init__(self, channel_drop_range=(1, 1), fill_value=0, always_apply=False, p=0.5):
        super(ChannelDropoutCustom, self).__init__(always_apply, p)

        self.min_channels = channel_drop_range[0]
        self.max_channels = channel_drop_range[1]

        assert 1 <= self.min_channels <= self.max_channels

        self.fill_value = fill_value

    def apply(self, img, channels_to_drop=(0, ), **params):
        return F.channel_dropout(img, channels_to_drop, self.fill_value)

    def get_params_dependent_on_targets(self, params):
        img = params['image']

        num_channels = img.shape[-1]

        if len(img.shape) == 2 or num_channels == 1:
            raise NotImplementedError("Images has one channel. ChannelDropout is not defined.")

        if self.max_channels >= num_channels:
            raise ValueError("Can not drop all channels in ChannelDropout.")

        num_drop_channels = random.randint(self.min_channels, self.max_channels)

        drop_channels = list(set(range(num_channels)) - set([0]))

        channels_to_drop = random.choice(drop_channels, size=num_drop_channels, replace=False)

        return {'channels_to_drop': channels_to_drop}

    def get_transform_init_args_names(self):
        return ('channel_drop_range', 'fill_value')


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

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

        img = img.resize(self.size, self.interpolation)

        return img

class RandomRotate(object):
    def __call__(self, image):
        # if random.random() < 0.5:
        #     image = image[:, :, ::-1, :]
        # if random.random() < 0.5:
        #     image = image[:, ::-1, :, :]
        # if random.random() < 0.5:
        #     image = image.transpose([0, 2, 1, 3])
        image = np.ascontiguousarray(image)


        size = random.randint(round(448 * 0.5), 448)
        x = random.randint(0, 448 - size)
        y = random.randint(0, 448 - size)
        image = image[x:x + size, y:y + size]

        image = cv2.resize(image, (448, 448), interpolation=cv2.INTER_NEAREST)
        return image


# class RandomRotate(object):
#     def __init__(self, degree, p=0.5):
#         self.degree = degree
#         self.p = p
#
#     def __call__(self, img):
#         if random.random() < self.p:
#             rotate_degree = random.uniform(-1*self.degree, self.degree)
#             img = img.rotate(rotate_degree, Image.BILINEAR)
#         return img

class RandomGaussianBlur(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return img

def get_train_transform(mean, std, size):
    train_transform = transforms.Compose([
        #Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        #RandomRotate(),
        #RandomGaussianBlur(),
        #Cutout(),
        RandomErasing(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_transform

def get_test_transform(mean, std, size):
    return transforms.Compose([
        #Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_transforms(input_size=224, test_size=224, backbone=None):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transformations = {}
    transformations['val_train'] = get_train_transform(mean, std, input_size)
    transformations['val_test'] = get_test_transform(mean, std, test_size)
    return transformations

