import torch
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
import torchvision.transforms.functional as TF
import os
import dask.dataframe as dd
import pandas as pd

KNOWN_MIN = -6.6
KNOWN_MAX = 101.9
KNOWN_MIN_MAX = True

"""
Custom Image dataset for image-to-image translation using pytorch

This is a custom dataset class for pytorch that reads images from a directory and returns them as tensors.
It also has the ability to apply transformations to the images before returning them.

As a note, the __getitem__ function currently returns the input, mask, and index of the image in the dataset
This is so the index can be used to get the filename of the image later on.
There is probably a better way to do this, but using memory location didn't work and I haven't found a better way yet.
"""
class ImageDataset(Dataset):
    """dataset class for image data"""
    in_channels = 3
    out_channels = 3

    def __init__(self, input_globbing_pattern: str, target_globbing_pattern: str, transform: callable = None) -> None:
        self.input_globbing_pattern = input_globbing_pattern
        self.target_globbing_pattern = target_globbing_pattern
        self.transform = transform
        
        self.images = sorted(glob(input_globbing_pattern, recursive=True))
        self.targets = sorted(glob(target_globbing_pattern, recursive=True))
        assert len(self.images) == len(self.targets), "Number of images and targets must be equal, is {} and {}".format(len(self.images), len(self.targets))

    """
    Returns the length of the dataset
    """
    def __len__(self) -> int:
        return len(self.images)
    
    """
    Returns the input and target tensors for the given index, as well as the index of the image in the dataset
    """
    def __getitem__(self, idx) -> tuple:
        input_tensor = Image.open(self.images[idx]).convert("RGB" if self.in_channels == 3 else "L")
        target_tensor = Image.open(self.targets[idx]).convert("RGB" if self.out_channels == 3 else "L")

        # note, implement this for real later
        if self.transform:
            input_tensor, target_tensor = self.transform(input_tensor, target_tensor)

        if not isinstance(input_tensor, torch.Tensor):
            #input_tensor = torch.from_numpy(np.array(input_tensor).astype(np.float32) / 255.0)
            input_tensor = TF.to_tensor(input_tensor)


        if not isinstance(target_tensor, torch.Tensor):
            #target_tensor = torch.from_numpy(np.array(target_tensor).astype(np.float32) / 255.0)
            target_tensor = TF.to_tensor(target_tensor)

        return input_tensor, target_tensor, idx

    """
    Returns the filename of the image at the given index
    """
    def get_filenames(self, idx, default=None) -> str:
        return self.images[idx]
    
"""
Custom Image dataset for image-to-image translation using pytorch for 3 image TIR

This is a custom dataset class for pytorch that reads images from a directory and returns them as tensors.
It also has the ability to apply transformations to the images before returning them.

As a note, the __getitem__ function currently returns the input, mask, and index of the image in the dataset
This is so the index can be used to get the filename of the image later on.
There is probably a better way to do this, but using memory location didn't work and I haven't found a better way yet.
"""
class ImageDataset3to1(Dataset):
    """dataset class for image data"""
    in_channels = 3
    out_channels = 3

    def __init__(self, input_globbing_pattern: str, target_globbing_pattern: str, transform: callable = None) -> None:
        self.input_globbing_pattern = input_globbing_pattern
        self.target_globbing_pattern = target_globbing_pattern
        self.transform = transform
        
        self.images0 = sorted(glob(os.path.join(input_globbing_pattern,'*_0*'), recursive=True))
        self.images1 = sorted(glob(os.path.join(input_globbing_pattern,'*_1*'), recursive=True))
        self.images2 = sorted(glob(os.path.join(input_globbing_pattern,'*_2*'), recursive=True))
        self.targets = sorted(glob(target_globbing_pattern, recursive=True))
        #assert len(self.images)/3 == len(self.targets), "Number of images and targets must be equal, is {} and {}".format(len(self.images), len(self.targets))

    """
    Returns the length of the dataset
    """
    def __len__(self) -> int:
        return len(self.targets)
    
    """
    Returns the input and target tensors for the given index, as well as the index of the image in the dataset
    """
    def __getitem__(self, idx) -> tuple:
        input_tensor1 = Image.open(self.images0[idx]).convert("L")
        input_tensor2 = Image.open(self.images1[idx]).convert("L")
        input_tensor3 = Image.open(self.images2[idx]).convert("L")

        input_tensor = Image.merge("RGB",(input_tensor1, input_tensor2, input_tensor3))

        target_tensor = Image.open(self.targets[idx]).convert("RGB" if self.out_channels == 3 else "L")

        # note, implement this for real later
        if self.transform:
            input_tensor, target_tensor = self.transform(input_tensor, target_tensor)

        if not isinstance(input_tensor, torch.Tensor):
            #input_tensor = torch.from_numpy(np.array(input_tensor).astype(np.float32) / 255.0)
            input_tensor = TF.to_tensor(input_tensor)


        if not isinstance(target_tensor, torch.Tensor):
            #target_tensor = torch.from_numpy(np.array(target_tensor).astype(np.float32) / 255.0)
            target_tensor = TF.to_tensor(target_tensor)

        return input_tensor, target_tensor, idx

    """
    Returns the filename of the image at the given index
    """
    def get_filenames(self, idx, default=None) -> str:
        return self.targets[idx]
    
class CSVImageDataset(Dataset):
    """dataset class for image data"""
    in_channels = 3
    out_channels = 1

    def __init__(self, input_globbing_pattern: str, target_globbing_pattern: str, transform: callable = None) -> None:
        self.input_globbing_pattern = input_globbing_pattern
        self.target_globbing_pattern = target_globbing_pattern
        self.transform = transform
        
        self.images = sorted(glob(input_globbing_pattern, recursive=True))
        self.targets = sorted(glob(target_globbing_pattern, recursive=True))

        assert len(self.images) == len(self.targets), "Number of images and targets must be equal, is {} and {}".format(len(self.images), len(self.targets))

        #self.global_min = float('inf')
        #self.global_max = float('-inf')

        #self.set_global_min_max()

    """
    Returns the length of the dataset
    """
    def __len__(self) -> int:
        return len(self.images)
    
    """
    Returns the input and target tensors for the given index, as well as the index of the image in the dataset
    """
    def __getitem__(self, idx) -> tuple:
        input_tensor = Image.open(self.images[idx]).convert("RGB" if self.in_channels == 3 else "L")

        # read in target from self.targets[idx] csv file
        target = pd.read_csv(self.targets[idx], header=None, delimiter=' ').iloc[1:, 20:260].values.astype('float32')
        
        #target = self.targets[idx].values
        #target = (target - self.global_min) / (self.global_max - self.global_min)
        target_tensor = torch.tensor(target).unsqueeze(0)

        # note, implement this for real later
        if self.transform:
            input_tensor, target_tensor = self.transform(input_tensor, target_tensor)

        if not isinstance(input_tensor, torch.Tensor):
            #input_tensor = torch.from_numpy(np.array(input_tensor).astype(np.float32) / 255.0)
            input_tensor = TF.to_tensor(input_tensor)


        if not isinstance(target_tensor, torch.Tensor):
            #target_tensor = torch.from_numpy(np.array(target_tensor).astype(np.float32) / 255.0)
            target_tensor = TF.to_tensor(target_tensor)

        return input_tensor, target_tensor, idx

    """
    Returns the filename of the image at the given index
    """
    def get_filenames(self, idx, default=None) -> str:
        return self.images[idx]
    
    def set_global_min_max(self):
        if KNOWN_MIN_MAX:
            self.global_min = KNOWN_MIN
            self.global_max = KNOWN_MAX
            return
        else:
            for target in self.targets:
                # extract number from target string (string is in form xxx//th_NUMBER.csv)
                # start by extracting last part of file path
                target = os.path.basename(target)
                idx = int(target.split('.')[0].split('_')[-1])

                target = pd.read_csv(self.targets[idx-1], header=None, delimiter=' ').iloc[1:260, 20:260].values
                min_val = target.min()
                max_val = target.max()
                if min_val < self.global_min:
                    self.global_min = min_val
                if max_val > self.global_max:
                    self.global_max = max_val

    def get_targets_from_csv(self, csv_path):
        df = dd.read_csv(csv_path,blocksize=1000000).iloc[:,list(range(20,260))].compute().astype('float32')
        #df = df.iloc[:, list(range(20, 260))]
        
        # Calculate global min and max
        self.global_min = df.values.min()
        self.global_max = df.values.max()

        # fit the scaler on all data
        #elf.scaler.fit(df.values)

        # split dataframe into a list of 240 row chunks
        chunks = [df.iloc[i:i+240] for i in range(0, len(df), 240)]
        return chunks