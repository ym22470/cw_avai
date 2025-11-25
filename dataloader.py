import os
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import Dataset

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class DIV2KDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, transform=None):
        """
        Args:
            hr_dir (string): Directory with high-res images.
            lr_dir (string): Directory with low-res images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.transform = transform

        # Get list of filenames and sort them to ensure alignment
        self.hr_files = sorted([f for f in os.listdir(hr_dir) if f.endswith('.png')])
        self.lr_files = sorted([f for f in os.listdir(lr_dir) if f.endswith('.png')])

        # Basic sanity check
        assert len(self.hr_files) == len(self.lr_files), "Mismatch in number of HR and LR images!"

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        # Load images
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])

        hr_image = Image.open(hr_path).convert("RGB")
        lr_image = Image.open(lr_path).convert("RGB")

        # Apply transforms (convert to Tensor)
        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)

        return lr_image, hr_image