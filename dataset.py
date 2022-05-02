import glob
import random
import numpy as np

import torch
from torch.utils.data import Dataset
import h5py

class TrainDataset(Dataset):
    def __init__(self, h5_file, patch_size, scale):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file
        self.patch_size = patch_size
        self.scale = scale

    @staticmethod
    def random_crop(lr, hr, size, scale):
        lr_left = random.randint(0, lr.shape[3] - size)
        lr_right = lr_left + size
        lr_top = random.randint(0, lr.shape[2] - size)
        lr_bottom = lr_top + size
        hr_left = lr_left * scale
        hr_right = lr_right * scale
        hr_top = lr_top * scale
        hr_bottom = lr_bottom * scale
        lr = lr[:, :, lr_top:lr_bottom, lr_left:lr_right]
        hr = hr[:, :, hr_top:hr_bottom, hr_left:hr_right]
        return lr, hr

    @staticmethod
    def random_horizontal_flip(lr, hr):
        if random.random() < 0.5:
            lr = lr[:, :, :, ::-1].copy()
            hr = hr[:, :, :, ::-1].copy()
        return lr, hr

    @staticmethod
    def random_vertical_flip(lr, hr):
        if random.random() < 0.5:
            lr = lr[:, :, ::-1, :].copy()
            hr = hr[:, :, ::-1, :].copy()
        return lr, hr

    @staticmethod
    def random_rotate_90(lr, hr):
        if random.random() < 0.5:
            lr = np.rot90(lr, axes=(3, 2)).copy()
            hr = np.rot90(hr, axes=(3, 2)).copy()
        return lr, hr

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr = f['lr'][str(idx)]
            hr = f['hr'][str(idx)]
            lr, hr = self.random_crop(lr, hr, self.patch_size, self.scale)
            lr, hr = self.random_horizontal_flip(lr, hr)
            lr, hr = self.random_vertical_flip(lr, hr)
            lr, hr = self.random_rotate_90(lr, hr)
            gt = hr[3, :, :, :]
            return lr.astype(np.float32)/255.0, gt.astype(np.float32)/255.0

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])