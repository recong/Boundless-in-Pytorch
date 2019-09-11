import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
# mean = np.array([0.485, 0.456, 0.406])
# std = np.array([0.229, 0.224, 0.225])

std = np.array([0.5, 0.5, 0.5])
mean = np.array([1, 1, 1])

def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].add_(mean[c]).mul_(std[c])
    return torch.clamp(tensors, 0, 255)


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape, ratio):
        self.hr_shape = hr_shape
        self.ratio = ratio
        # Transforms for low resolution images and high resolution images
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std),
             ]
        )

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        hr_shape = self.hr_shape
        ratio = self.ratio
        edge = int(hr_shape * ratio) + random.randint(-4, 4)
        a = Image.open(self.files[index % len(self.files)])
        width, height = a.size
        a = np.asarray(a).astype("f").transpose(2, 0, 1) / 127.5 - 1.0
        x = random.randint(0, height - hr_shape)
        y = random.randint(0, width - hr_shape)
        a = a[:, x:x + hr_shape, y:y + hr_shape]
        c = np.zeros((hr_shape, hr_shape))  # make mask
        c[:, :edge] = 1
        c = c[np.newaxis, :, :]
        b = a * (1 - c)  # apply mask

        c = torch.from_numpy(c)
        img_hr = torch.from_numpy(a)
        img_lr = torch.from_numpy(b)
        clip = torch.cat([img_lr, c])

        return {"lr": img_lr, "hr": img_hr, 'alpha': c, 'clip': clip}

    def __len__(self):
        return len(self.files)
