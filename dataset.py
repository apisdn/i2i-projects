import os
import random
import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import Dataset
from glob import glob
from PIL import Image

'''
input_reader: function(filename: str, channels: int) -> np.ndarray | torch.Tensor
            Function to read the input data
            default: lambda x, channels: Image.open(x).convert("RGB" if channels == 3 else "L")
'''


class ImageDataset(Dataset):
    """dataset class for image data"""
    in_channels = 3
    out_channels = 3

    def __init__(self, input_globbing_pattern, target_globbing_pattern, transform=None):
        self.input_globbing_pattern = input_globbing_pattern
        self.target_globbing_pattern = target_globbing_pattern
        self.transform = transform

        # Store the paths to the images
        self._defaults = None
        self._filenames = {}
        
        self.images = sorted(glob(input_globbing_pattern, recursive=True))
        self.targets = sorted(glob(target_globbing_pattern, recursive=True))
        assert len(self.images) == len(self.targets), "Number of images and targets must be equal, is {} and {}".format(len(self.images), len(self.targets))

    def __len__(self):
        return len(self._filenames)
    
    def __getitem__(self, idx):
        input_tensor = Image.open(self.images[idx]).convert("RGB" if self.in_channels == 3 else "L")
        target_tensor = Image.open(self.targets[idx][1]).convert("RGB" if self.out_channels == 3 else "L")

        # note, implement this for real later
        if self.transform:
            input_tensor, target_tensor = self.transform(input_tensor, target_tensor)

        if not isinstance(input_tensor, torch.Tensor):
            input_tensor = torch.from_numpy(np.array(input_tensor))

        if not isinstance(target_tensor, torch.Tensor):
            target_tensor = torch.from_numpy(np.array(target_tensor))

        # For plotting purposes
        self._filenames[id(input_tensor)] = self.images[idx]
        self._filenames[id(target_tensor)] = self.targets[idx]

        return input_tensor, target_tensor

    def get_filenames(self, tensor, default=None):
        return self._filenames.get(id(tensor), default)