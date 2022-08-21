

from torchvision import transforms
import torch
import skimage
import numpy as np
from PIL import Image


def AddGaussianNoise(img, mean=0.0, variance=1.0):
        img = np.array(img)
        img_noise = skimage.util.random_noise(img, mode='gaussian', seed=None, mean = mean, var=(variance / 255.0) ** 2)  # add  gaussian noise clip to [0, 1]
        img_noise = img_noise * 255  # limit the value in the range (0,255)
        img_noise = Image.fromarray((np.uint8(img_noise))).convert('RGB')
        return img_noise

