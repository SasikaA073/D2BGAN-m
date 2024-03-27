import torch
from PIL import Image
import os
# import config
import numpy as np
from torch.utils.data import Dataset, DataLoader
from config import train_transforms

class LOLDataset(Dataset):
    def __init__(self, root_dark, root_bright, transform=train_transforms):
        super().__init__()

        self.root_dark = root_dark
        self.root_bright = root_bright
        self.transform = transform

        self.dark_images = os.listdir(root_dark)
        self.bright_images = os.listdir(root_bright)
        self.length_dataset = max(len(self.dark_images), len(self.bright_images))   # 485, 485 (I can add images as much as I want)
        self.dark_len = len(self.dark_images)
        self.bright_len = len(self.bright_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        dark_img = self.dark_images[index % self.dark_len]
        bright_img = self.bright_images[index % self.bright_len]

        dark_path = os.path.join(self.root_dark, dark_img)
        bright_path = os.path.join(self.root_bright, bright_img)

        # dark_img = np.array(Image.open(dark_path).convert("RGB"))
        # bright_img = np.array(Image.open(bright_path).convert("RGB"))

        dark_img = Image.open(dark_path).convert("RGB")
        bright_img = Image.open(bright_path).convert("RGB")

        if self.transform:
            dark_img = self.transform(dark_img)
            bright_img = self.transform(bright_img)
        return dark_img, bright_img
